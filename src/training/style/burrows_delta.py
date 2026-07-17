"""Tokenizer shared by the vendored key-phrase and style-feature modules.

VENDORED from the main Anubis API
(``anubis/src/anubis/utils/dataset/burrows_delta.py``). Only :func:`tokenize`
is needed here — it is the dependency of
:mod:`src.training.style.key_phrases` — so the Burrows Delta scoring functions
were not carried over. The tokenisation must stay identical to the source so a
phrase discovered by the Anubis API matches token-for-token in this service.
"""

from __future__ import annotations

import re
from typing import List

_TOKEN_RE = re.compile(r"[A-Za-z']+")

# LLM and word-processor text uses the Unicode curly apostrophe family; the
# ASCII-only token pattern above would split "don’t" into "don" + "t" (which is
# how apostrophe-less phrases like "don t" leaked into the discovered
# key-phrase lists). Normalize every variant to the ASCII apostrophe BEFORE
# tokenising so "don't" / "don’t" produce the identical single token.
_APOSTROPHE_VARIANTS_RE = re.compile(r"[‘’ʼ`´]")


def tokenize(text: str) -> List[str]:
    """Lowercase alpha tokens; apostrophes preserved (function words like ``it's``)."""
    normalized = _APOSTROPHE_VARIANTS_RE.sub("'", text or "")
    return [t.lower() for t in _TOKEN_RE.findall(normalized)]
