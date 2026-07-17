"""Per-feature stylometric GRPO reward functions.

One asynchronous reward function per stylometric feature in
:data:`src.training.style.style_features.FEATURE_NAMES` (28 features, vector
version 4) — the SAME features the main Anubis API measures avatar authenticity
with, so a reward improvement here is by construction an improvement on the
production quality metric.

Each feature's reward for a generated completion is

    ``exp(-|feature(completion) - reference_median| / reference_scale)``

which lives in ``[0, 1]``: 1.0 when the completion matches the reference
corpus's median value for that feature exactly, decaying as the completion
drifts away, with the decay normalized by the corpus's own spread (the
interquartile range) so widely-varying features are judged loosely and tight
signature features are judged strictly. A feature that cannot be computed on a
completion (``nan``) rewards ``0.0``.

Feature extraction runs once per completion per training step: the first
reward function to see a batch populates a shared per-job cache and the other
27 read from it, so 28 reward functions cost one extraction pass.
"""

from __future__ import annotations

import asyncio
import math
import warnings
from typing import Any, Callable, Dict, List, Sequence

from src.training.style.key_phrases import discover_key_phrases
from src.training.style.style_features import FEATURE_NAMES, extract_style_features

# Floor for the per-feature scale so a zero-IQR feature (every reference
# completion has the identical value) never divides by zero — it instead
# becomes a near-exact-match requirement.
MINIMUM_FEATURE_SCALE = 1e-6

# The feature cache is cleared past this size so a long training run over a
# large completion space cannot grow memory without bound.
MAXIMUM_CACHED_COMPLETIONS = 8192


def build_reference_style_profile(
    reference_completions: Sequence[str],
) -> Dict[str, Any]:
    """Calibrate the reward target from the dataset's real completions.

    Computed once per training job. Returns::

        {
            "key_phrases": [<signature phrase strings>],
            "median": {feature_name: float},   # per-feature reference median
            "scale": {feature_name: float},    # per-feature IQR, floored
        }

    The signature key phrases are discovered from the reference corpus first so
    the ``key_phrase_rate`` feature is measured against the target's own
    phrases, exactly as the Anubis API does.
    """
    import numpy as np

    discovered_phrases = discover_key_phrases(list(reference_completions))
    key_phrases = [entry["phrase"] for entry in discovered_phrases]

    feature_rows = [
        extract_style_features(completion_text, key_phrases=key_phrases)
        for completion_text in reference_completions
    ]
    feature_matrix = np.array(
        [[row[feature_name] for feature_name in FEATURE_NAMES] for row in feature_rows],
        dtype=np.float64,
    )

    with warnings.catch_warnings():
        # A column that is NaN in every reference row raises an "All-NaN slice"
        # RuntimeWarning; its median/scale fall back to 0.0 / the floor below.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        column_medians = np.nanmedian(feature_matrix, axis=0)
        column_75th_percentiles = np.nanpercentile(feature_matrix, 75, axis=0)
        column_25th_percentiles = np.nanpercentile(feature_matrix, 25, axis=0)

    column_medians = np.nan_to_num(column_medians, nan=0.0)
    column_scales = np.nan_to_num(
        column_75th_percentiles - column_25th_percentiles, nan=0.0
    )
    column_scales = np.maximum(column_scales, MINIMUM_FEATURE_SCALE)

    return {
        "key_phrases": key_phrases,
        "median": {
            feature_name: float(column_medians[feature_index])
            for feature_index, feature_name in enumerate(FEATURE_NAMES)
        },
        "scale": {
            feature_name: float(column_scales[feature_index])
            for feature_index, feature_name in enumerate(FEATURE_NAMES)
        },
    }


class StyleFeatureCache:
    """Per-job cache mapping completion text -> its 28-feature dict.

    All 28 reward functions of one training step share one instance, so the
    (comparatively expensive) nltk/lexicalrichness extraction runs once per
    completion per step instead of 28 times.
    """

    def __init__(self, key_phrases: Sequence[str]):
        self.key_phrases = list(key_phrases)
        self._features_by_completion_text: Dict[str, Dict[str, float]] = {}

    async def features_for(
        self, completion_texts: Sequence[str]
    ) -> Dict[str, Dict[str, float]]:
        """Return the feature dict for every text, extracting the missing ones.

        Extraction is fanned out to worker threads (`asyncio.to_thread`) since
        ``extract_style_features`` is pure-CPU.
        """
        if len(self._features_by_completion_text) > MAXIMUM_CACHED_COMPLETIONS:
            self._features_by_completion_text.clear()

        missing_texts = list(
            dict.fromkeys(
                text
                for text in completion_texts
                if text not in self._features_by_completion_text
            )
        )
        if missing_texts:
            extracted_rows = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        extract_style_features, text, key_phrases=self.key_phrases
                    )
                    for text in missing_texts
                )
            )
            for text, feature_row in zip(missing_texts, extracted_rows):
                self._features_by_completion_text[text] = feature_row

        return {
            text: self._features_by_completion_text[text] for text in completion_texts
        }


async def compute_feature_reward(
    feature_name: str,
    completion_texts: Sequence[str],
    reference_style_profile: Dict[str, Any],
    style_feature_cache: StyleFeatureCache,
) -> List[float]:
    """Asynchronous reward for ONE stylometric feature over a completion batch.

    ``exp(-|feature(completion) - median| / scale)`` in ``[0, 1]``; ``nan``
    feature values reward ``0.0``.
    """
    features_by_text = await style_feature_cache.features_for(completion_texts)
    reference_median = reference_style_profile["median"][feature_name]
    reference_scale = reference_style_profile["scale"][feature_name]

    rewards: List[float] = []
    for completion_text in completion_texts:
        feature_value = features_by_text[completion_text].get(feature_name, math.nan)
        if math.isnan(feature_value):
            rewards.append(0.0)
        else:
            rewards.append(
                math.exp(-abs(feature_value - reference_median) / reference_scale)
            )
    return rewards


def _completion_to_text(completion: Any) -> str:
    """Extract the assistant text from a TRL completion (conversational or plain)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        text_parts: List[str] = []
        for message in completion:
            if isinstance(message, dict):
                content = message.get("content")
                text_parts.append(content if isinstance(content, str) else str(content))
            else:
                text_parts.append(str(message))
        return "".join(text_parts)
    return str(completion)


def build_reward_functions(
    reference_style_profile: Dict[str, Any],
) -> List[Callable[..., List[float]]]:
    """Return the 28 synchronous reward callables for ``GRPOTrainer``.

    Each callable follows the TRL reward signature
    ``(prompts, completions, **kwargs) -> list[float]`` and carries
    ``__name__ = feature_name`` so TRL logs a per-feature reward mean
    (``rewards/<feature_name>/mean``) every step.

    Each bridge runs its asynchronous implementation with ``asyncio.run``,
    which is safe here: ``GRPOTrainer`` executes reward functions inside the
    training executor thread, which has no running event loop.
    """
    style_feature_cache = StyleFeatureCache(reference_style_profile["key_phrases"])

    reward_functions: List[Callable[..., List[float]]] = []
    for feature_name in FEATURE_NAMES:

        def feature_reward_bridge(
            prompts: Any = None,
            completions: Any = None,
            feature_name: str = feature_name,
            **kwargs: Any,
        ) -> List[float]:
            completion_texts = [
                _completion_to_text(completion) for completion in (completions or [])
            ]
            return asyncio.run(
                compute_feature_reward(
                    feature_name,
                    completion_texts,
                    reference_style_profile,
                    style_feature_cache,
                )
            )

        feature_reward_bridge.__name__ = feature_name
        feature_reward_bridge.__qualname__ = feature_name
        reward_functions.append(feature_reward_bridge)

    return reward_functions
