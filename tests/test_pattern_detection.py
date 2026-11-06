"""Evidence-based pattern detection tests (Issue 1 root cause).

The pattern must come from measured image evidence, never from a disease name.
"""

import pytest

from services.color_filter import segment_leaf, analyze_colors
from services.pattern_detection import detect_pattern
from conftest import (
    make_healthy_leaf, make_yellow_patch_leaf, make_white_powder_leaf,
    make_black_coating_leaf, make_brown_spot_leaf,
)


def _evidence(img):
    mask, _ = segment_leaf(img)
    return analyze_colors(img, mask)


class TestPatternFromEvidence:
    def test_healthy_pattern_none(self):
        ev = _evidence(make_healthy_leaf())
        res = detect_pattern(ev, 'healthy')
        assert res['pattern'] == 'none'
        assert res['confidence'] == 1.0

    def test_yellow_patches_not_powdery(self):
        """The reported bug: yellow patches labeled 'powdery'. Yellow evidence
        must yield yellowing, never powdery."""
        ev = _evidence(make_yellow_patch_leaf())
        res = detect_pattern(ev, 'unhealthy')
        assert res['pattern'] == 'yellowing'
        assert res['pattern'] != 'powdery'
        assert res['confidence'] > 0

    def test_white_powder_is_powdery(self):
        ev = _evidence(make_white_powder_leaf())
        res = detect_pattern(ev, 'unhealthy')
        assert res['pattern'] == 'powdery'

    def test_black_coating_is_dark_coating(self):
        ev = _evidence(make_black_coating_leaf())
        res = detect_pattern(ev, 'unhealthy')
        assert res['pattern'] == 'dark_coating'

    def test_brown_spots(self):
        ev = _evidence(make_brown_spot_leaf())
        res = detect_pattern(ev, 'unhealthy')
        assert res['pattern'] == 'spots'

    def test_unhealthy_no_evidence_inconclusive(self):
        """Unhealthy prediction but no color evidence must NOT invent a
        pattern from a disease name."""
        ev = _evidence(make_healthy_leaf())  # green, no disease colors
        res = detect_pattern(ev, 'unhealthy')
        assert res['pattern'] == 'inconclusive'
        assert res['confidence'] == 0.0

    def test_pattern_independent_of_disease_name(self):
        """detect_pattern takes only evidence + health, no disease argument —
        structurally guaranteeing patterns are not disease-name mappings."""
        import inspect
        sig = inspect.signature(detect_pattern)
        assert 'disease' not in sig.parameters
        assert set(sig.parameters) == {'color_evidence', 'predicted_health'}


class TestPatternHandBuiltEvidence:
    def test_mixed_patterns(self):
        ev = {
            'ratios': {'yellow': 0.20, 'white': 0.18, 'brown': 0.0,
                       'black': 0.0, 'gray': 0.0},
            'components': {},
        }
        res = detect_pattern(ev, 'unhealthy')
        assert res['pattern'] == 'mixed'

    def test_dominant_yellow_over_minor_white(self):
        ev = {
            'ratios': {'yellow': 0.30, 'white': 0.06, 'brown': 0.0,
                       'black': 0.0, 'gray': 0.0},
            'components': {},
        }
        res = detect_pattern(ev, 'unhealthy')
        assert res['pattern'] == 'yellowing'

    def test_gray_alone_inconclusive(self):
        ev = {
            'ratios': {'gray': 0.40, 'yellow': 0.0, 'white': 0.0,
                       'brown': 0.0, 'black': 0.0},
            'components': {},
        }
        res = detect_pattern(ev, 'unhealthy')
        assert res['pattern'] == 'inconclusive'

    def test_confidence_bounded(self):
        ev = {'ratios': {'yellow': 0.9}, 'components': {}}
        res = detect_pattern(ev, 'unhealthy')
        assert 0.0 <= res['confidence'] <= 1.0
