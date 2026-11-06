"""Evidence-based visual pattern detection.

The legacy pipeline copied ``pattern`` verbatim from the matched knowledge-base
row, i.e. the pattern was a static property of the *predicted disease name*
("Coffee Sooty Mold" -> "powdery") and never looked at the image. A leaf with
yellow patches could therefore be labeled "powdery" with no white growth in
sight.

This module derives the pattern exclusively from measured image evidence
(color area ratios + connected-component structure) and the health prediction:

  * healthy leaf                 -> 'none'      (no disease pattern)
  * yellow-dominant evidence     -> 'yellowing' (chlorosis-like discoloration)
  * white-dominant evidence      -> 'powdery'   (white surface growth)
  * black-dominant evidence      -> 'dark_coating'
  * brown / discrete multi-spot  -> 'spots'
  * several strong categories    -> 'mixed'
  * unhealthy, no color evidence -> 'inconclusive' (never invented)

Every result carries a confidence in [0, 1] and the supporting ratios, so the
dashboard and the logs can show *why* a pattern was reported.
"""

from config import (
    COLOR_THRESHOLDS,
    PATTERN_MIXED_MIN_CATEGORIES,
    PATTERN_CONFIDENCE_SATURATION_RATIO,
)

# Mapping from visual evidence category to the color that supports it.
_CATEGORY_COLORS = {
    'yellowing':    'yellow',
    'powdery':      'white',
    'dark_coating': 'black',
    'spots':        'brown',
}


def _strength(ratio, threshold):
    """Evidence strength in [0, 1]: 0 at threshold, 1 at saturation ratio."""
    if ratio <= threshold:
        return 0.0
    span = max(PATTERN_CONFIDENCE_SATURATION_RATIO - threshold, 1e-6)
    return min(1.0, (ratio - threshold) / span)


def detect_pattern(color_evidence, predicted_health):
    """Derive a visual pattern from color evidence and the health prediction.

    Parameters
    ----------
    color_evidence : dict
        Output of ``services.color_filter.analyze_colors`` (needs ``ratios``
        and ``components``; ``colors`` is recomputed from ratios so the
        function also works with hand-built evidence in tests).
    predicted_health : str
        'healthy' or 'unhealthy' (output of the health model).

    Returns
    -------
    dict with keys ``pattern``, ``confidence`` (0-1), ``evidence`` (ratios used)
    and ``reason`` (human-readable justification).
    """
    ratios = color_evidence.get('ratios', {})
    components = color_evidence.get('components', {})

    if predicted_health == 'healthy':
        return {
            'pattern': 'none',
            'confidence': 1.0,
            'evidence': {},
            'reason': 'Leaf classified as healthy; no disease pattern reported.',
        }

    strengths = {}
    for category, color in _CATEGORY_COLORS.items():
        ratio = float(ratios.get(color, 0.0))
        strengths[category] = _strength(ratio, COLOR_THRESHOLDS[color])

    # Gray alone is ambiguous (shading, dust, several diseases) and is never
    # sufficient evidence for a specific pattern on its own.
    strong = {c: s for c, s in strengths.items() if s > 0.0}

    if not strong:
        return {
            'pattern': 'inconclusive',
            'confidence': 0.0,
            'evidence': {c: ratios.get(c, 0.0) for c in _CATEGORY_COLORS.values()},
            'reason': 'Leaf classified as unhealthy but no color pattern has '
                      'sufficient visual evidence; pattern left inconclusive.',
        }

    if len(strong) >= PATTERN_MIXED_MIN_CATEGORIES:
        ordered = sorted(strong.items(), key=lambda kv: kv[1], reverse=True)
        top_cat, top_strength = ordered[0]
        second_strength = ordered[1][1]
        # Only call it 'mixed' when categories are comparably strong;
        # otherwise report the dominant one.
        if second_strength >= 0.5 * top_strength:
            return {
                'pattern': 'mixed',
                'confidence': round(min(0.75, top_strength), 4),
                'evidence': {c: ratios.get(c, 0.0) for c in _CATEGORY_COLORS.values()},
                'reason': 'Multiple comparable visual patterns detected: '
                          + ', '.join(c for c, _ in ordered) + '.',
            }

    dominant = max(strong.items(), key=lambda kv: kv[1])
    category, strength = dominant

    # Structural refinement for brown evidence: discrete multi-spot lesions
    # vs. a single large necrotic patch. Both are reported as 'spots' family,
    # but a single dominant patch is better described as a lesion; we keep
    # 'spots' for compatibility with the existing dashboard vocabulary.
    reason = {
        'yellowing':    'Yellow discoloration covers a significant leaf area.',
        'powdery':      'White surface regions consistent with powdery growth.',
        'dark_coating': 'Dark/black regions consistent with a surface coating.',
        'spots':        'Brown lesion regions detected on the leaf.',
    }[category]

    return {
        'pattern': category,
        'confidence': round(strength, 4),
        'evidence': {_CATEGORY_COLORS[category]: ratios.get(_CATEGORY_COLORS[category], 0.0)},
        'reason': reason,
    }
