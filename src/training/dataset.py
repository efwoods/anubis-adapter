"""Pull per-user/per-assistant adapter training datasets from the Anubis store.

The Anubis API's media -> identity pipeline writes prompt-completion datasets
into its LangGraph store (the PostgreSQL ``store`` table) under prefixes of the
form ``{user_id}.{assistant_id}.{dataset_type}.{source_uuid5}``. This module
runs the same query as ``anubis/sql/adapter_dataset_store_search.sql`` against
that table (read-only) and converts the rows into the Hugging Face ``Dataset``
GRPO trains on plus the reference-completion corpus the style rewards are
calibrated against.

Row shapes (verified against ``anubis/src/subgraphs/process_media_graph``):

* ``q_and_a_adapter`` — a JSON array of ``{"prompt": <str>, "completion": <str>}``.
* ``multi_turn_dataset_adapter`` — a JSON array where each element is one
  conversation ``{"messages": [{"role": ..., "content": ...}, ...]}``.
* ``langsmith_factual_q_and_a`` — factual question/answer pairs, EXCLUDED from
  training and held out for LangSmith evaluation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Dataset types that feed GRPO training. ``langsmith_factual_q_and_a`` is
# deliberately absent: those rows are the held-out factual evaluation set.
TRAINING_DATASET_TYPES = ("q_and_a_adapter", "multi_turn_dataset_adapter")

# The query from ``anubis/sql/adapter_dataset_store_search.sql``, with the
# named parameters converted to psycopg placeholders (``%%`` escapes the
# literal ``%`` of the LIKE wildcard).
ADAPTER_DATASET_STORE_QUERY = """
SELECT
    split_part(prefix, '.', 3)        AS dataset_type,
    split_part(prefix, '.', 4)        AS source_uuid5,
    key,
    value ->> 'source_filename'       AS source_filename,
    (value ->> 'row_count')::int      AS row_count,
    value ->> 'created_at'            AS created_at,
    value ->> 'value'                 AS dataset_json
FROM store
WHERE prefix LIKE %s || '.' || %s || '.%%'
  AND split_part(prefix, '.', 3) IN (
        'q_and_a_adapter',
        'langsmith_factual_q_and_a',
        'multi_turn_dataset_adapter'
  )
ORDER BY dataset_type, created_at;
"""


async def fetch_adapter_datasets(
    store_pool: Any, user_id: str, assistant_id: str
) -> List[Dict[str, Any]]:
    """Fetch every stored adapter dataset for one ``(user_id, assistant_id)`` pair.

    Returns one dict per store row:
    ``{"dataset_type", "source_uuid5", "key", "source_filename", "row_count",
    "created_at", "rows"}`` where ``rows`` is the parsed JSON array. Rows whose
    JSON fails to parse are logged and skipped rather than failing the fetch.
    """
    async with store_pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                ADAPTER_DATASET_STORE_QUERY, (user_id, assistant_id)
            )
            store_rows = await cursor.fetchall()

    datasets_found: List[Dict[str, Any]] = []
    for (
        dataset_type,
        source_uuid5,
        key,
        source_filename,
        row_count,
        created_at,
        dataset_json,
    ) in store_rows:
        try:
            parsed_rows = json.loads(dataset_json) if dataset_json else []
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Skipping unparseable %s dataset %s (%s): %s",
                dataset_type,
                source_uuid5,
                source_filename,
                exc,
            )
            continue
        if not isinstance(parsed_rows, list):
            logger.warning(
                "Skipping %s dataset %s: expected a JSON array, found %s",
                dataset_type,
                source_uuid5,
                type(parsed_rows).__name__,
            )
            continue
        datasets_found.append(
            {
                "dataset_type": dataset_type,
                "source_uuid5": source_uuid5,
                "key": key,
                "source_filename": source_filename,
                "row_count": row_count,
                "created_at": created_at,
                "rows": parsed_rows,
            }
        )
    return datasets_found


def _message_content_text(content: Any) -> str:
    """Coerce a message ``content`` field (string or content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                text_parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                text_parts.append(str(block))
        return "".join(text_parts)
    return "" if content is None else str(content)


def _prompt_message_list(prompt_value: Any) -> List[Dict[str, str]]:
    """Normalize a stored prompt (plain string or message list) to message form."""
    if isinstance(prompt_value, list):
        return [
            {
                "role": str(message.get("role", "user")),
                "content": _message_content_text(message.get("content")),
            }
            for message in prompt_value
            if isinstance(message, dict)
        ]
    return [{"role": "user", "content": _message_content_text(prompt_value)}]


def _completion_text(completion_value: Any) -> str:
    """Normalize a stored completion (plain string or message list) to text."""
    if isinstance(completion_value, list):
        return "".join(
            _message_content_text(message.get("content"))
            for message in completion_value
            if isinstance(message, dict)
        )
    return _message_content_text(completion_value)


def build_grpo_dataset(
    fetched_datasets: List[Dict[str, Any]],
) -> Tuple[Any, List[str]]:
    """Build the GRPO training dataset and the style-reference corpus.

    Returns ``(training_dataset, reference_completions)``:

    * ``training_dataset`` — a ``datasets.Dataset`` with a conversational
      ``prompt`` column (a list of ``{"role", "content"}`` messages per
      example). GRPO generates its own completions per prompt, so the stored
      real completions are NOT a training column.
    * ``reference_completions`` — the target's real completion texts, the
      corpus the per-feature style rewards are calibrated against.

    ``q_and_a_adapter`` rows become one single-turn example each.
    ``multi_turn_dataset_adapter`` conversations become one example per
    assistant turn, whose prompt is the message list up to (excluding) that
    turn. ``langsmith_factual_q_and_a`` rows are excluded (held out for
    evaluation).
    """
    prompt_examples: List[List[Dict[str, str]]] = []
    reference_completions: List[str] = []

    for fetched_dataset in fetched_datasets:
        dataset_type = fetched_dataset["dataset_type"]
        if dataset_type not in TRAINING_DATASET_TYPES:
            continue

        if dataset_type == "q_and_a_adapter":
            for row in fetched_dataset["rows"]:
                if not isinstance(row, dict):
                    continue
                prompt_messages = _prompt_message_list(row.get("prompt"))
                completion_text = _completion_text(row.get("completion")).strip()
                if not prompt_messages or not completion_text:
                    continue
                prompt_examples.append(prompt_messages)
                reference_completions.append(completion_text)

        elif dataset_type == "multi_turn_dataset_adapter":
            for conversation in fetched_dataset["rows"]:
                if not isinstance(conversation, dict):
                    continue
                messages = conversation.get("messages") or []
                normalized_messages = [
                    {
                        "role": str(message.get("role", "user")),
                        "content": _message_content_text(message.get("content")),
                    }
                    for message in messages
                    if isinstance(message, dict)
                ]
                for turn_index, message in enumerate(normalized_messages):
                    if message["role"] != "assistant":
                        continue
                    preceding_messages = normalized_messages[:turn_index]
                    completion_text = message["content"].strip()
                    if not preceding_messages or not completion_text:
                        continue
                    prompt_examples.append(preceding_messages)
                    reference_completions.append(completion_text)

    # datasets is a heavy import; keep the module import-cheap.
    from datasets import Dataset

    training_dataset = Dataset.from_dict({"prompt": prompt_examples})
    return training_dataset, reference_completions
