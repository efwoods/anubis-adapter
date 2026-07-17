"""Auth0 API-key authentication, ported lean from the main Anubis API.

Kept from ``scaffold/security/auth.py``: the ``API-KEY`` header scheme, API-key
generation/hashing, the cached Auth0 Management token, the retrying httpx
wrapper, the hashed-key TTL cache, the Auth0 user lookup by hashed key (with a
clean 503 on transport failure), ``get_current_user``, signup, JWT
verification, and the ``/signup`` + ``/rotate_api_key`` routes.

Dropped: Supabase, Stripe/subscriptions, the anonymous-user path, LangGraph
client calls, and ``/delete_user`` — none apply to this service. Configuration
comes from :mod:`src.config` instead of the Anubis ``GlobalContext``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from jose import JWTError, jwt
from pydantic import BaseModel

from src.config import (
    AUTH0_AUDIENCE,
    AUTH0_CLIENT_ID,
    AUTH0_CLIENT_SECRET,
    AUTH0_CONNECTION,
    AUTH0_DOMAIN,
)

logger = logging.getLogger(__name__)

security_route = APIRouter()

api_key_scheme = APIKeyHeader(name="API-KEY")

ALGORITHMS = ["RS256"]

BASE_AUTH_URL = f"https://{AUTH0_DOMAIN}"

_api_key_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
_cache_lock = asyncio.Lock()


def generate_api_key() -> str:
    """Generates a secure, persistent API key."""
    return f"sk-{secrets.token_urlsafe(32)}"


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


async def retry_async_httpx_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> httpx.Response:
    """Async retry wrapper for httpx requests (exponential backoff on 429/5xx)."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    data=data,
                )

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"Retryable HTTP error: {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()
                return response

            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError,
            ) as exc:
                is_last_attempt = attempt == max_retries - 1

                if isinstance(exc, httpx.HTTPStatusError):
                    status_code = exc.response.status_code
                    if status_code not in {429, 500, 502, 503, 504}:
                        logger.exception("Non-retryable HTTP error")
                        raise

                if is_last_attempt:
                    logger.exception("Max retries exceeded")
                    raise

                delay = base_delay * (2**attempt)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.2fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    raise RuntimeError("Unexpected retry failure")


# ── Management API token (cached) ──────────────────────────────────────────
_management_token_cache: dict = {"token": None, "expires": 0}


async def _get_mgmt_token(request: Request) -> str:
    """Get a Management API token using client credentials."""
    now = time.monotonic()
    if _management_token_cache["token"] and now < _management_token_cache["expires"]:
        return _management_token_cache["token"]

    token_request_body = {
        "grant_type": "client_credentials",
        "client_id": AUTH0_CLIENT_ID,
        "client_secret": AUTH0_CLIENT_SECRET,
        "audience": f"{BASE_AUTH_URL}/api/v2/",
    }
    result = await retry_async_httpx_request(
        "POST", url=f"{BASE_AUTH_URL}/oauth/token", json=token_request_body
    )
    result.raise_for_status()
    data = result.json()
    _management_token_cache["token"] = data["access_token"]
    _management_token_cache["expires"] = now + data["expires_in"] - 60
    return _management_token_cache["token"]


async def _mgmt_headers(request: Request) -> dict:
    access_token = await _get_mgmt_token(request)
    return {"Authorization": f"Bearer {access_token}"}


async def signup_user(
    email: str, password: str, request: Request, name: Optional[str] = None
) -> dict:
    """Create an Auth0 user carrying a hashed API key; return the plain key once."""
    api_key = generate_api_key()
    api_key_hash = _hash_key(api_key)

    payload: Dict[str, Any] = {
        "email": email,
        "password": password,
        "connection": AUTH0_CONNECTION,
        "app_metadata": {
            "api_key": api_key_hash,
        },
    }
    if name:
        payload["name"] = name

    headers = await _mgmt_headers(request)
    response = await request.app.state.httpx_client.post(
        f"{BASE_AUTH_URL}/api/v2/users",
        json=payload,
        headers=headers,
    )
    if response.status_code == 400:
        raise HTTPException(
            detail=(
                "Invalid Password. Password requires a lower case and upper case "
                "character as well as at least 8 characters and a special "
                f"character: {response.json()}"
            ),
            status_code=response.status_code,
        )
    if response.status_code >= 400:
        raise HTTPException(
            detail=f"Error signing up user: {response.json()}",
            status_code=response.status_code,
        )
    return {
        "api_key": api_key,
        "message": (
            "Save this key. This key is shown only once and used for every "
            "api request."
        ),
    }


async def login_user(email: str, password: str, request: Request) -> httpx.Response:
    """Authenticate a user and return access/id/refresh tokens.

    Requires the Resource Owner Password Grant to be enabled on the Auth0
    application. Used by ``/rotate_api_key`` to verify credentials.
    """
    return await request.app.state.httpx_client.post(
        f"{BASE_AUTH_URL}/oauth/token",
        json={
            "grant_type": "password",
            "username": email,
            "password": password,
            "audience": AUTH0_AUDIENCE,
            "scope": "openid profile email offline_access",
            "client_id": AUTH0_CLIENT_ID,
            "client_secret": AUTH0_CLIENT_SECRET,
        },
    )


# ── Token verification ──────────────────────────────────────────────────────
_jwks_cache: dict = {"keys": None}


async def _get_jwks(request: Request) -> dict:
    if _jwks_cache["keys"] is not None:
        return _jwks_cache["keys"]
    response = await request.app.state.httpx_client.get(
        f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    )
    response.raise_for_status()
    _jwks_cache["keys"] = response.json()
    return _jwks_cache["keys"]


async def verify_token(token: str, request: Request) -> dict:
    """Decodes and validates an Auth0 JWT. Returns the payload."""
    jwks = await _get_jwks(request)
    unverified_header = jwt.get_unverified_header(token)

    rsa_key = next(
        (
            {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"],
            }
            for key in jwks["keys"]
            if key["kid"] == unverified_header["kid"]
        ),
        None,
    )
    if not rsa_key:
        raise JWTError("Unable to find matching key")

    return jwt.decode(
        token,
        rsa_key,
        algorithms=ALGORITHMS,
        audience=AUTH0_AUDIENCE,
        issuer=f"https://{AUTH0_DOMAIN}/",
    )


# ── Schemas ──────────────────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str | None = None


# ── Dependency: require valid API key ───────────────────────────────────────
async def get_user_with_api_key(api_key: str, request: Request) -> dict | None:
    """Look the caller up in Auth0 by hashed API key (TTL-cached)."""
    cache_key = _hash_key(api_key)

    async with _cache_lock:
        if cache_key in _api_key_cache:
            return _api_key_cache[cache_key]

    headers = await _mgmt_headers(request)
    try:
        result = await request.app.state.httpx_client.get(
            f"{BASE_AUTH_URL}/api/v2/users",
            params={"q": f'app_metadata.api_key:"{cache_key}"', "search_engine": "v3"},
            headers=headers,
        )
    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.TimeoutException,
        httpx.TransportError,
    ) as exc:
        # The identity provider is unreachable (DNS/egress/outage, or the event
        # loop was starved). Surface a clean 503 instead of a generic 500 so the
        # client can retry rather than treating it as a request error.
        logger.warning("Auth lookup to %s failed (transport): %s", BASE_AUTH_URL, exc)
        raise HTTPException(
            status_code=503, detail="Authentication service temporarily unreachable."
        ) from exc
    result.raise_for_status()
    users = result.json()
    if not users:
        return None
    user = users[0]

    if user["email_verified"] is not True:
        raise HTTPException(
            detail="Email is not yet verified. Please verify email to continue.",
            status_code=401,
        )

    async with _cache_lock:
        _api_key_cache[cache_key] = user

    user.update({"API_KEY": api_key})
    return user


async def get_current_user(
    request: Request, api_key: str | None = Depends(api_key_scheme)
) -> dict:
    """FastAPI dependency: the authenticated Auth0 user for the ``API-KEY`` header."""
    if not api_key:
        raise HTTPException(status_code=401, detail="Please send API-KEY in request.")

    user = await get_user_with_api_key(api_key, request)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user


# ── Routes ───────────────────────────────────────────────────────────────────
@security_route.post("/signup")
async def signup(body: SignupRequest, request: Request):
    user = await signup_user(body.email, body.password, name=body.name, request=request)
    return user


@security_route.post("/rotate_api_key")
async def rotate_api_key(request: Request, email: str, password: str):
    result = await login_user(email=email, password=password, request=request)
    id_token = result.json().get("id_token", None)
    if id_token:
        current_user = jwt.get_unverified_claims(id_token)
    else:
        raise HTTPException(detail="Invalid Credentials.", status_code=401)

    new_key = generate_api_key()
    new_key_hash = _hash_key(new_key)

    headers = await _mgmt_headers(request)
    user_id = current_user["sub"].split("|")[1]
    encoded_user_id = quote(current_user["sub"], safe="")
    response = await request.app.state.httpx_client.patch(
        f"{BASE_AUTH_URL}/api/v2/users/{encoded_user_id}",
        json={"app_metadata": {"api_key": new_key_hash}},
        headers=headers,
    )
    if response.status_code >= 400:
        raise HTTPException(
            detail=f"Error patching the new api_key: {response.json()}",
            status_code=response.status_code,
        )

    async with _cache_lock:
        stale_cache_keys = [
            cache_key
            for cache_key, cached_user in _api_key_cache.items()
            if cached_user["identities"][0]["user_id"] == user_id
        ]
        for cache_key in stale_cache_keys:
            del _api_key_cache[cache_key]

    return {
        "api_key": new_key,
        "message": (
            "Save this key. This key is shown only once and used on every "
            "api request."
        ),
    }
