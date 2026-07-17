"""Shared stylometric feature extractor — the single source of truth.

VENDORED from the main Anubis API
(``anubis/src/anubis/utils/dataset/style_features.py``) so the GRPO reward
functions here score completions with the SAME 28 features the Anubis API uses
to measure avatar authenticity. Only the internal import paths were changed and
the anubis-store-specific persistence/baseline helpers at the bottom were
dropped; keep the feature computations byte-identical to the source when
updating.

This module computes a flat dictionary of **28 scalar stylometric features** for
one text (feature-vector version 3; see :data:`STYLE_FEATURE_VECTOR_VERSION`).
The SAME function is called by the production authenticity evaluator
(``graph._attach_analyzed_features``), the per-avatar calibration
(``calibrate_ground_truth``), the bundled-baseline builder
(``data/build_baseline_features_arr.py``), and the validation notebook
(``style.ipynb``), so every path exercises the same code rather than a parallel
re-implementation that could drift.

Design constraints (from ``features/prompt_drafts/style/style.md``):

* **No spaCy** — part-of-speech information comes from ``nltk.pos_tag`` only.
  Features that would require a dependency parse (clause density, parse-tree
  depth, T-units) are intentionally omitted.
* **No VADER / sentiment** — sentiment lives elsewhere (Go-Emotions on the
  reply); style is measured purely from form.
* Every feature returns a finite ``float`` where possible and ``nan`` on texts
  too short to support the metric. No exception ever propagates out of
  :func:`extract_style_features`.

The feature names are deliberately self-commenting (``mean_sentence_length_words``
rather than ``MLS``) so downstream profile JSON and evaluator reports read
without a glossary.
"""

from __future__ import annotations

import html
import logging
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical, ORDERED feature list. The order is the column order of every
# feature matrix and feature vector in the pipeline, so it must stay stable:
# the stored covariance / inverse-covariance matrices are indexed by it.
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # ── Lexical diversity (3) ──────────────────────────────────────────────
    # Only the length-ROBUST diversity indices are kept. Raw TTR, Maas a², and
    # Yule's K were removed in vector version 3 as multicollinear: TTR is length-
    # biased and its two raw components (vocabulary size, total word count) were
    # also dropped, while Maas and Yule's K measure the same repetition signal
    # MATTR/MTLD/HD-D already carry length-robustly.
    "moving_average_ttr",                  # TTR averaged over a sliding window (length-robust)
    "mtld_lexical_diversity",              # Measure of Textual Lexical Diversity
    "hdd_lexical_diversity",               # Hypergeometric Distribution Diversity (HD-D)
    "lexical_density_content_word_ratio",  # content words / all words
    # ── Part-of-speech density via nltk.pos_tag (7) ────────────────────────
    # pos_sequence_compressibility was removed in v3 as redundant with
    # lexical_entropy_bits (both measure sequence predictability/variety).
    "noun_density",                        # share of tokens tagged noun
    "verb_density",                        # share of tokens tagged verb
    "adjective_density",                   # share of tokens tagged adjective
    "adverb_density",                      # share of tokens tagged adverb
    "pronoun_density",                     # share of tokens tagged pronoun
    "preposition_density",                 # share of tokens tagged preposition
    "noun_to_verb_ratio",                  # nominal (high) vs verbal/conversational (low) style
    # ── Sentence shape (4) ─────────────────────────────────────────────────
    "mean_sentence_length_words",          # average words per sentence
    "stdev_sentence_length_words",         # sentence-length variability (rhythm)
    "interrogative_sentence_ratio",        # share of sentences ending in '?'
    "exclamatory_sentence_ratio",          # share of sentences ending in '!'
    # ── Punctuation fingerprint, marks per word (7) ────────────────────────
    # Renamed from *_rate_per_1k in v4: the same counts normalized per WORD
    # (the per-1k value divided by 1,000) so every rate lives on a 0–1 scale.
    "comma_rate_per_word",
    "semicolon_rate_per_word",
    "colon_rate_per_word",
    "dash_rate_per_word",
    "ellipsis_rate_per_word",
    "exclamation_rate_per_word",
    "question_mark_rate_per_word",
    # ── Surface / flow (3) ─────────────────────────────────────────────────
    "all_caps_word_ratio",                 # SHOUTING / emphasis habit
    "words_per_paragraph",                 # internet writing = short paragraphs
    "transition_word_rate_per_word",       # logical-bridge words per word (however, therefore, …)
    # The readability composites (Flesch-Kincaid, Gunning Fog, SMOG) were removed
    # in v3: all three are deterministic functions of sentence length + syllable/
    # complex-word counts, so they are mutually collinear and add no signal beyond
    # the sentence-shape and word-length features already present.
    # ── Information theory (1) ─────────────────────────────────────────────
    "lexical_entropy_bits",                # Shannon entropy of the word distribution
    # ── Word & vocabulary shape (1) ────────────────────────────────────────
    # vocabulary_size_unique_words and total_word_count were removed in v3 (they
    # are the raw numerator/denominator of TTR — captured, and length-dependent).
    "average_word_length_characters",      # mean characters per word (orthographic length habit)
    # ── Signature key phrases (1) ──────────────────────────────────────────
    # Occurrences of the avatar's auto-discovered signature key-phrases per total
    # word. Unlike the character n-grams / function-word vectors (which were
    # capture-only and are now dropped), this collapses the key-phrase signal to a
    # single scalar so it can enter the Mahalanobis vector. The phrase set is
    # avatar-specific and passed into extract_style_features; see key_phrases.py.
    "key_phrase_rate",
]

# Bump whenever the composition or order of FEATURE_NAMES changes. The width of
# this vector is baked into persisted artifacts (the per-document ground-truth
# corpus rows, the bundled ChatGPT baseline matrix + IsolationForest, and any
# stored covariance matrices). Readers use `len(FEATURE_NAMES)` to detect and
# discard rows written under an older version — see deserialize_features_by_doc_id
# and the baseline staleness guards in graph._attach_analyzed_features /
# utility.load_baseline_features_explainer_model.
#   v1: the original 33 features.
#   v2: appended average_word_length_characters, vocabulary_size_unique_words,
#       total_word_count (width 33 -> 36).
#   v3: removed 9 multicollinear features (type_token_ratio, maas_lexical_diversity,
#       yule_characteristic_k, pos_sequence_compressibility, flesch_kincaid_grade,
#       gunning_fog_index, smog_index, vocabulary_size_unique_words,
#       total_word_count) and appended key_phrase_rate (width 36 -> 28).
#   v4: renamed the eight *_rate_per_1k features to *_rate_per_word and rescaled
#       their VALUES from marks-per-1,000-words to marks-per-word (divided by
#       1,000, a 0–1 scale). Width unchanged (28) — which is exactly why the
#       persisted corpus is now VERSION-TAGGED by serialize_features_by_doc_id:
#       a width check alone cannot tell v3 rows (per-1k scale) from v4 rows
#       (per-word scale), and mixing the two scales in one Mahalanobis /
#       IsolationForest corpus silently corrupts both comparisons.
STYLE_FEATURE_VECTOR_VERSION = 4

assert len(FEATURE_NAMES) == 28, f"expected 28 features, found {len(FEATURE_NAMES)}"


# Human-legible display title per feature. Keyed by the snake_case FEATURE_NAMES
# so build_style_profile_str() can render `LEGIBLE[name]`. The titles are what the
# LLM sees in the <STYLE> block, so they spell out acronyms (MATTR, HD-D, SMOG)
# rather than leaving the raw variable name.
FEATURE_NAMES_HUMAN_LEGIBLE: Dict[str, str] = {
    # ── Lexical diversity (3) ──────────────────────────────────────────────
    "moving_average_ttr": "Moving-Average Type-Token Ratio (MATTR)",
    "mtld_lexical_diversity": "Measure of Textual Lexical Diversity (MTLD)",
    "hdd_lexical_diversity": "Hypergeometric Distribution Diversity (HD-D)",
    "lexical_density_content_word_ratio": "Lexical Density (Content-Word Ratio)",
    # ── Part-of-speech density via nltk.pos_tag (7) ────────────────────────
    "noun_density": "Noun Density",
    "verb_density": "Verb Density",
    "adjective_density": "Adjective Density",
    "adverb_density": "Adverb Density",
    "pronoun_density": "Pronoun Density",
    "preposition_density": "Preposition Density",
    "noun_to_verb_ratio": "Noun-to-Verb Ratio",
    # ── Sentence shape (4) ─────────────────────────────────────────────────
    "mean_sentence_length_words": "Mean Sentence Length (words)",
    "stdev_sentence_length_words": "Sentence-Length Variability (std dev, words)",
    "interrogative_sentence_ratio": "Question-Sentence Ratio",
    "exclamatory_sentence_ratio": "Exclamation-Sentence Ratio",
    # ── Punctuation fingerprint, marks per word (7) ────────────────────────
    "comma_rate_per_word": "Commas per Word",
    "semicolon_rate_per_word": "Semicolons per Word",
    "colon_rate_per_word": "Colons per Word",
    "dash_rate_per_word": "Dashes per Word",
    "ellipsis_rate_per_word": "Ellipses per Word",
    "exclamation_rate_per_word": "Exclamation Marks per Word",
    "question_mark_rate_per_word": "Question Marks per Word",
    # ── Surface / flow (3) ─────────────────────────────────────────────────
    "all_caps_word_ratio": "ALL-CAPS Word Ratio",
    "words_per_paragraph": "Words per Paragraph",
    "transition_word_rate_per_word": "Transition Words per Word",
    # ── Information theory (1) ─────────────────────────────────────────────
    "lexical_entropy_bits": "Lexical Entropy (bits)",
    # ── Word & vocabulary shape (1) ────────────────────────────────────────
    "average_word_length_characters": "Average Word Length (characters)",
    # ── Signature key phrases (1) ──────────────────────────────────────────
    "key_phrase_rate": "Signature Key-Phrase Rate (per word)",
}

assert len(FEATURE_NAMES) == len(FEATURE_NAMES_HUMAN_LEGIBLE), (
    f"expected {len(FEATURE_NAMES)} features, found {len(FEATURE_NAMES_HUMAN_LEGIBLE)}"
)

# One-line plain-language description per feature, written for the LLM that reads
# the style profile. Each states the unit/range, the typical band, and crucially
# which DIRECTION means what — because the numbers alone are meaningless to the
# model (and several features are inverse: Maas and Yule's K go DOWN as vocabulary
# gets richer). Ranges are read straight off the computations in
# extract_style_features (POS/diversity shares are 0–1, punctuation is per-word
# on a 0–1 scale, etc.).
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # ── Lexical diversity (3) ──────────────────────────────────────────────
    "moving_average_ttr": "Type-token ratio averaged over a sliding ~50-word window. Ranges 0–1. Higher means richer vocabulary. Length-robust, so it stays comparable across short and long texts.",
    "mtld_lexical_diversity": "Mean length of word runs that stay above a 0.72 type-token threshold. Unbounded above; typically ~20–120 (most prose 40–100). Higher means sustained lexical variety, lower means vocabulary that repeats quickly.",
    "hdd_lexical_diversity": "Hypergeometric (HD-D) diversity: the type-token ratio a random fixed-size sample is expected to show. Ranges 0–1, typically ~0.70–0.90. Higher means more diverse word choice.",
    "lexical_density_content_word_ratio": "Content words (noun/verb/adjective/adverb) over all tagged tokens. Ranges 0–1, typically ~0.4–0.6. Higher means dense, informational, nominal writing; lower means more function words and a conversational feel.",
    # ── Part-of-speech density via nltk.pos_tag (7) ────────────────────────
    "noun_density": "Share of tokens tagged as nouns. Ranges 0–1, typically ~0.20–0.35. Higher means a nominal, topic-heavy style.",
    "verb_density": "Share of tokens tagged as verbs. Ranges 0–1, typically ~0.15–0.25. Higher means an active, event-driven style.",
    "adjective_density": "Share of tokens tagged as adjectives. Ranges 0–1, typically ~0.05–0.10. Higher means more descriptive, modifier-heavy writing.",
    "adverb_density": "Share of tokens tagged as adverbs. Ranges 0–1, typically ~0.03–0.08. Higher means more hedging/intensifying ('really', 'very', 'just').",
    "pronoun_density": "Share of tokens tagged as pronouns. Ranges 0–1, typically ~0.05–0.15. Higher means a personal, conversational voice (I/you/we).",
    "preposition_density": "Share of tokens tagged as prepositions (including 'to'). Ranges 0–1, typically ~0.10–0.15. Higher means more elaborated, phrase-stacked syntax.",
    "noun_to_verb_ratio": "Nouns divided by verbs (+1 smoothed so it stays finite). Greater than 0, typically ~1–3. Higher means a nominal, formal register; lower (near 1) means a verbal, conversational register.",
    # ── Sentence shape (4) ─────────────────────────────────────────────────
    "mean_sentence_length_words": "Average words per sentence. Greater than 0, typically ~10–25. Higher means longer, more complex sentences; lower means short, punchy ones.",
    "stdev_sentence_length_words": "Standard deviation of sentence length, in words. 0 or greater, typically ~4–15. Higher means rhythmic variety (mixing long and short sentences); 0 means uniform sentence length.",
    "interrogative_sentence_ratio": "Fraction of sentences ending in '?'. Ranges 0–1. Higher means a questioning, rhetorical, engaging style.",
    "exclamatory_sentence_ratio": "Fraction of sentences ending in '!'. Ranges 0–1. Higher means an emphatic, excited tone.",
    # ── Punctuation fingerprint, marks per word (7) ────────────────────────
    "comma_rate_per_word": "Commas per word (occurrences divided by total words). Ranges 0–1, typically ~0.04–0.08. Higher means more clause-chaining and parenthetical phrasing.",
    "semicolon_rate_per_word": "Semicolons per word (occurrences divided by total words). Ranges 0–1, usually ~0.0–0.005 (rare). Higher means deliberate, formal joining of independent clauses.",
    "colon_rate_per_word": "Colons per word (occurrences divided by total words). Ranges 0–1, typically ~0.0–0.01. Higher means frequent setups, lists, or explanatory pauses.",
    "dash_rate_per_word": "Em dashes, en dashes, and hyphens per word (occurrences divided by total words). Ranges 0–1, typically ~0.0–0.03. Higher means an interruptive, aside-heavy, informal rhythm.",
    "ellipsis_rate_per_word": "Ellipsis characters ('…') per word (occurrences divided by total words). Ranges 0–1, typically ~0.0–0.01. Higher means trailing-off, hesitant, or suspenseful phrasing. Counts only the single '…' glyph, not three dots.",
    "exclamation_rate_per_word": "Exclamation marks per word (occurrences divided by total words). Ranges 0–1. Higher means an emphatic, high-energy tone.",
    "question_mark_rate_per_word": "Question marks per word (occurrences divided by total words). Ranges 0–1. Higher means a more inquisitive, rhetorical style.",
    # ── Surface / flow (3) ─────────────────────────────────────────────────
    "all_caps_word_ratio": "Fraction of multi-letter tokens written in ALL CAPS. Ranges 0–1, usually near 0. Higher means a habit of SHOUTING or capitalized emphasis.",
    "words_per_paragraph": "Words divided by number of paragraphs (blank-line separated). Greater than 0. Higher means long, blocky paragraphs; lower means short, internet-style chunks.",
    "transition_word_rate_per_word": "Logical-bridge words (however, therefore, moreover, …) per word (occurrences divided by total words). Ranges 0–1, typically ~0.0–0.02. Higher means explicit, essayistic argument structure.",
    # ── Information theory (1) ─────────────────────────────────────────────
    "lexical_entropy_bits": "Shannon entropy of the word-frequency distribution, in bits. 0 or greater and grows with vocabulary size (~4–10+ bits common). Higher means less predictable, more varied word choice; lower means repetitive, predictable wording.",
    # ── Word & vocabulary shape (1) ────────────────────────────────────────
    "average_word_length_characters": "Mean number of characters per word (apostrophes counted, e.g. \"it's\" is 4). Greater than 0, typically ~4–5 for English prose. Higher means a preference for longer, often more formal or Latinate words; lower means shorter, plainer words.",
    # ── Signature key phrases (1) ──────────────────────────────────────────
    "key_phrase_rate": "Occurrences of the avatar's auto-discovered signature key-phrases (2–4 word recurring expressions like 'you know', 'got it') per total word in the text. 0 or greater, usually small (~0.0–0.1). Higher means the writing leans on the speaker's characteristic fixed phrasings; 0 means none of the signature phrases appear.",
}

assert len(FEATURE_NAMES) == len(FEATURE_DESCRIPTIONS), (
    f"expected {len(FEATURE_NAMES)} features, found {len(FEATURE_DESCRIPTIONS)}"
)


# ---------------------------------------------------------------------------
# Lexicons / regexes built once at import (cheap, no model downloads).
# ---------------------------------------------------------------------------

# Logical "bridge" words counted for transition density.
_TRANSITION_WORDS = frozenset(
    {
        "however",
        "therefore",
        "furthermore",
        "moreover",
        "nevertheless",
        "consequently",
        "meanwhile",
        "conversely",
        "thus",
        "hence",
        "accordingly",
        "additionally",
        "similarly",
        "instead",
        "otherwise",
        "subsequently",
    }
)

# Penn-Treebank tag prefixes -> coarse POS class. nltk.pos_tag emits PTB tags.
_NOUN_TAGS = ("NN",)  # NN, NNS, NNP, NNPS
_VERB_TAGS = ("VB",)  # VB, VBD, VBG, VBN, VBP, VBZ
_ADJECTIVE_TAGS = ("JJ",)  # JJ, JJR, JJS
_ADVERB_TAGS = ("RB",)  # RB, RBR, RBS
_PRONOUN_TAGS = ("PRP", "WP")  # PRP, PRP$, WP, WP$
_PREPOSITION_TAGS = ("IN", "TO")  # IN (prep/subord-conj), TO

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_WORD_RE = re.compile(r"[A-Za-z']+")
# Unicode curly-apostrophe family -> ASCII, so "don’t" stays ONE word token
# under the ASCII-only _WORD_RE instead of splitting into "don" + "t" (mirrors
# burrows_delta._APOSTROPHE_VARIANTS_RE — the two tokenisers must agree).
_APOSTROPHE_VARIANTS_RE = re.compile(r"[‘’ʼ`´]")
_SENTENCE_FALLBACK_RE = re.compile(r"(?<=[.!?])\s+")

# Punctuation fingerprint: label -> the character(s) that count toward it.
_PUNCTUATION_MARKS: Dict[str, str] = {
    "comma_rate_per_word": ",",
    "semicolon_rate_per_word": ";",
    "colon_rate_per_word": ":",
    "dash_rate_per_word": "—–-",   # em dash, en dash, hyphen-minus
    "ellipsis_rate_per_word": "…",
    "exclamation_rate_per_word": "!",
    "question_mark_rate_per_word": "?",
}


def _ensure_nltk_resources() -> None:
    """Lazily download the nltk data the extractor needs.

    Kept inside a function (not at import) so importing this module never pays a
    download/cold-start cost — consistent with the repo's lazy-import convention.
    Downloads are no-ops once the data is cached.
    """
    import nltk

    for resource, locator in (
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("stopwords", "corpora/stopwords"),
    ):
        try:
            nltk.data.find(locator)
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_text(text: str) -> str:
    """Light, stylometry-preserving normalisation.

    Ground-truth tweets carry HTML entities (``&amp;``), URLs and @mentions that
    would distort token and punctuation counts. We unescape entities and drop
    URLs + @mentions, but deliberately KEEP casing and punctuation because those
    ARE the stylistic signal we want to measure.
    """
    text = html.unescape(text or "")
    text = _APOSTROPHE_VARIANTS_RE.sub("'", text)
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    return text.strip()


def _word_tokens(text: str) -> List[str]:
    """Lowercased alphabetic tokens (apostrophes kept so ``it's`` stays whole)."""
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _sentences(text: str) -> List[str]:
    """Sentence split via nltk; regex fallback if the model is unavailable."""
    try:
        from nltk.tokenize import sent_tokenize

        sents = [s for s in sent_tokenize(text or "") if s.strip()]
    except Exception:
        sents = [
            s.strip() for s in _SENTENCE_FALLBACK_RE.split(text or "") if s.strip()
        ]
    return sents or ([text.strip()] if (text or "").strip() else [])


def _nan_features() -> Dict[str, float]:
    """All-NaN feature row for empty/degenerate input (callers impute later)."""
    return {name: math.nan for name in FEATURE_NAMES}

def extract_style_features(
    text: str,
    *,
    key_phrases: Sequence[str] | None = None,
    update_key_phrases_only: bool = False,
    features_dict: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return the 28 stylometric scalars for one document.

    The returned dict is keyed by :data:`FEATURE_NAMES`. Values are floats; a
    metric that cannot be computed on the given text yields ``nan`` rather than
    raising, so a single short document never breaks a batch.

    Args:
        text: The document to fingerprint.
        key_phrases: The avatar's auto-discovered signature phrases (from
            :func:`src.anubis.utils.dataset.key_phrases.discover_key_phrases`).
            Used ONLY to compute ``key_phrase_rate`` (occurrences per total word).
            When ``None`` or empty — e.g. a text with no calibrated phrase set —
            ``key_phrase_rate`` is ``0.0``. This is the one avatar-relative
            feature in the vector; every other feature depends on ``text`` alone.
        update_key_phrases_only: Recompute ONLY ``key_phrase_rate`` against the
            given ``key_phrases``, reusing every other value from
            ``features_dict``. Used when the same text must be scored against a
            second phrase set (e.g. baseline phrases first, then the avatar's)
            without paying for a full re-extraction. Returns a NEW dict; the
            passed ``features_dict`` is not mutated.
        features_dict: Pre-computed features dictionary the update-only path
            copies its non-key-phrase values from. Required when
            ``update_key_phrases_only`` is True.

    Coverage of the scalar features requested in
    ``features/statistical_significance.md``:

    * average sentence length (words per sentence) — ``mean_sentence_length_words``.
    * average word length (characters per word) — ``average_word_length_characters``.
    * punctuation frequencies — the seven ``*_rate_per_word`` marks above.
    * signature key-phrase reliance — ``key_phrase_rate``.

    The character n-gram and function-word VECTORS that used to be captured in the
    nested :mod:`src.anubis.utils.dataset.stylistic_profile` profile were dropped;
    the key-phrase signal is now carried here as the single ``key_phrase_rate``
    scalar and, separately, as the prompt-injected signature-phrase list.
    """

    if update_key_phrases_only:
        if not features_dict:
            raise ValueError(
                "Must send a pre-computed features_dict from which to update key_phrase_rate."
            )
        # Copy (never mutate the caller's dict) and re-measure only the
        # key-phrase rate against the new phrase set below.
        features: Dict[str, float] = dict(features_dict)
        cleaned = clean_text(text)
    else:
        _ensure_nltk_resources()

        cleaned = clean_text(text)
        if not cleaned:
            return _nan_features()

        from lexicalrichness import LexicalRichness
        from nltk.tokenize import word_tokenize

        features: Dict[str, float] = {}

        # Token views: `words` keeps punctuation as separate tokens (needed for
        # ALL-CAPS detection); `alpha_words` is the lowercased alphabetic stream
        # that most lexical metrics operate on.
        words = word_tokenize(cleaned)
        alpha_words = _word_tokens(cleaned)
        alpha_count = len(alpha_words) or 1            # guard divisions by zero
        per_word = 1.0 / alpha_count                   # marks-per-word (0–1 scale)
        sentences = _sentences(cleaned)
        sentence_count = len(sentences) or 1

        # ── A. LEXICAL DIVERSITY ───────────────────────────────────────────────
        # Only the length-ROBUST diversity indices are kept (v3). Raw TTR, Maas a²,
        # and Yule's K were removed as multicollinear with MATTR/MTLD/HD-D. Short
        # texts make several of these undefined, so each is guarded individually.
        lex = LexicalRichness(cleaned)
        features["moving_average_ttr"] = _safe(
            lambda: lex.mattr(window_size=min(50, max(1, lex.words)))
        )
        features["mtld_lexical_diversity"] = _safe(lambda: lex.mtld(threshold=0.72))
        features["hdd_lexical_diversity"] = _safe(
            lambda: lex.hdd(draws=min(42, max(1, lex.words)))
        )

        # Word-frequency table, reused below for lexical entropy. (Yule's K, which
        # also derived from this table, was removed in v3.)
        word_frequencies = Counter(alpha_words)

        # ── B. PART-OF-SPEECH DENSITY (nltk.pos_tag, Penn Treebank) ────────────
        # One tagging pass feeds every POS feature plus lexical density.
        pos_tags = [tag for _, tag in _safe_pos_tag(words)]
        pos_total = len(pos_tags) or 1
        noun_count = _count_tags(pos_tags, _NOUN_TAGS)
        verb_count = _count_tags(pos_tags, _VERB_TAGS)
        adjective_count = _count_tags(pos_tags, _ADJECTIVE_TAGS)
        adverb_count = _count_tags(pos_tags, _ADVERB_TAGS)
        pronoun_count = _count_tags(pos_tags, _PRONOUN_TAGS)
        preposition_count = _count_tags(pos_tags, _PREPOSITION_TAGS)

        features["noun_density"] = noun_count / pos_total
        features["verb_density"] = verb_count / pos_total
        features["adjective_density"] = adjective_count / pos_total
        features["adverb_density"] = adverb_count / pos_total
        features["pronoun_density"] = pronoun_count / pos_total
        features["preposition_density"] = preposition_count / pos_total
        # +1 smoothing keeps the ratio finite when a class is absent.
        features["noun_to_verb_ratio"] = (noun_count + 1) / (verb_count + 1)

        # Lexical density = content words (noun/verb/adj/adv) / all tagged tokens.
        content_word_count = noun_count + verb_count + adjective_count + adverb_count
        features["lexical_density_content_word_ratio"] = content_word_count / pos_total

        # ── C. SENTENCE SHAPE ──────────────────────────────────────────────────
        sentence_lengths = [len(_word_tokens(s)) for s in sentences]
        mean_sentence_length = sum(sentence_lengths) / sentence_count
        features["mean_sentence_length_words"] = mean_sentence_length
        features["stdev_sentence_length_words"] = _population_stdev(
            sentence_lengths, mean_sentence_length
        )
        features["interrogative_sentence_ratio"] = (
            sum(1 for s in sentences if s.rstrip().endswith("?")) / sentence_count
        )
        features["exclamatory_sentence_ratio"] = (
            sum(1 for s in sentences if s.rstrip().endswith("!")) / sentence_count
        )

        # ── D. PUNCTUATION FINGERPRINT (marks per word, 0–1 scale) ─────────────
        for feature_name, characters in _PUNCTUATION_MARKS.items():
            features[feature_name] = (
                sum(cleaned.count(ch) for ch in characters) * per_word
            )

        # ── E. SURFACE / FLOW ──────────────────────────────────────────────────
        features["all_caps_word_ratio"] = (
            sum(1 for w in words if w.isupper() and len(w) > 1) / (len(words) or 1)
        )
        paragraphs = [p for p in re.split(r"\n\s*\n", cleaned) if p.strip()] or [cleaned]
        features["words_per_paragraph"] = alpha_count / len(paragraphs)
        features["transition_word_rate_per_word"] = (
            sum(1 for w in alpha_words if w in _TRANSITION_WORDS) * per_word
        )

        # (Readability composites — Flesch-Kincaid, Gunning Fog, SMOG — were removed
        # in v3 as mutually collinear functions of sentence length + syllable counts.)

        # ── F. INFORMATION THEORY ──────────────────────────────────────────────
        # Shannon entropy of the unigram distribution, in bits: how unpredictable the
        # next word is. Computed from the word-frequency table built above.
        entropy_bits = 0.0
        for count in word_frequencies.values():
            probability = count / alpha_count
            entropy_bits -= probability * math.log2(probability)
        features["lexical_entropy_bits"] = entropy_bits

        # ── G. WORD SHAPE ──────────────────────────────────────────────────────
        # `total_words` is the TRUE token count (len(alpha_words)); `alpha_count`
        # above was floored to 1 only to guard divisions, so it must not be reused
        # here. Average word length divides total characters by that true count and
        # is NaN when there are no word tokens (all-punctuation input). (The raw
        # vocabulary-size and total-word-count features were removed in v3 as the
        # length-dependent components of TTR.)
        total_words = len(alpha_words)
        total_characters = sum(len(word) for word in alpha_words)
        features["average_word_length_characters"] = (
            total_characters / total_words if total_words else math.nan
        )

    # ── H. SIGNATURE KEY-PHRASE RATE ───────────────────────────────────────
    # Occurrences of the avatar's signature phrases per total word. The phrase set
    # is avatar-specific (passed in); with no set the rate is 0.0. Delegated to
    # key_phrases so the tokenisation matches how the phrases were discovered.
    from src.training.style.key_phrases import key_phrase_occurrence_rate

    features["key_phrase_rate"] = key_phrase_occurrence_rate(cleaned, key_phrases)

    # Guarantee exactly the declared keys, in the declared order.
    return {name: float(features.get(name, math.nan)) for name in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Small numeric helpers (kept module-private and self-documenting).
# ---------------------------------------------------------------------------


def _safe(metric_fn: Callable[[], Any], default: float = 0.0) -> float:
    """Run a metric, swallowing short-text / zero-division errors.

    Returns ``default`` on the ``ValueError`` / ``ZeroDivisionError`` that
    lexicalrichness raises on tiny inputs.
    """
    try:
        value = metric_fn()
    except (ValueError, ZeroDivisionError, IndexError, KeyError):
        return default
    if value is None:
        return default
    value = float(value)
    return default if math.isnan(value) else value


def _safe_pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    """``nltk.pos_tag`` with a defensive fallback to an empty tagging."""
    try:
        from nltk import pos_tag

        return list(pos_tag(tokens))
    except Exception:
        return []


def _count_tags(pos_tags: List[str], prefixes: Tuple[str, ...]) -> int:
    """Count tags whose Penn-Treebank label starts with any of ``prefixes``."""
    return sum(1 for tag in pos_tags if tag.startswith(prefixes))


def _population_stdev(values: Sequence[float], mean: float) -> float:
    """Return the population standard deviation (ddof=0); 0.0 if < 2 values."""
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
