"""Color evidence + segmentation tests (Issue 1 & Issue 3 root causes).

Covers synthetic controlled images and the repository's real sample photos.
"""

import glob
import os

import cv2
import numpy as np
import pytest

from services.color_filter import (
    get_leaf_mask, segment_leaf, analyze_colors, detect_colors,
)
from conftest import (
    make_healthy_leaf, make_yellow_patch_leaf, make_white_powder_leaf,
    make_black_coating_leaf, make_brown_spot_leaf, make_shadow_leaf,
    make_non_leaf, make_blurry_leaf,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSegmentation:
    def test_healthy_leaf_segmented(self):
        mask, info = segment_leaf(make_healthy_leaf())
        assert info['is_leaf']
        assert info['coverage'] > 0.10
        assert info['green_fraction'] > 0.8

    def test_non_leaf_rejected(self):
        mask, info = segment_leaf(make_non_leaf())
        assert not info['is_leaf']
        assert np.count_nonzero(mask) == 0

    def test_no_background_fallback(self):
        """Legacy bug: low-green images fell back to a global threshold that
        admitted the whole background. The new mask must stay empty."""
        mask = get_leaf_mask(make_non_leaf())
        assert np.count_nonzero(mask) == 0

    def test_lesion_inside_leaf_counts_as_leaf_area(self):
        """Legacy bug: green-only mask excluded internal yellow lesions, so
        they were never counted as color evidence. Hole-filling must include
        them in the leaf area."""
        img = make_yellow_patch_leaf()
        mask, info = segment_leaf(img)
        # The yellow patch center must be inside the leaf mask.
        assert mask[120, 100] > 0
        assert info['is_leaf']

    def test_blurry_leaf_still_segmented_but_flagged(self):
        mask, info = segment_leaf(make_blurry_leaf())
        assert info['is_leaf']
        assert info['blur_advisory']


class TestColorEvidence:
    def test_healthy_leaf_no_disease_colors(self):
        img = make_healthy_leaf()
        mask, _ = segment_leaf(img)
        evidence = analyze_colors(img, mask)
        assert evidence['colors'] == []
        assert np.count_nonzero(evidence['disease_mask']) == 0

    def test_shadow_not_black_disease(self):
        img = make_shadow_leaf()
        mask, _ = segment_leaf(img)
        evidence = analyze_colors(img, mask)
        assert 'black' not in evidence['colors']

    def test_yellow_patches_detected(self):
        img = make_yellow_patch_leaf()
        mask, _ = segment_leaf(img)
        evidence = analyze_colors(img, mask)
        assert 'yellow' in evidence['colors']
        assert evidence['ratios']['yellow'] > 0.04
        # Yellow must NOT be reported as white/powdery.
        assert 'white' not in evidence['colors']

    def test_white_powder_detected(self):
        img = make_white_powder_leaf()
        mask, _ = segment_leaf(img)
        evidence = analyze_colors(img, mask)
        assert 'white' in evidence['colors']

    def test_black_coating_detected(self):
        img = make_black_coating_leaf()
        mask, _ = segment_leaf(img)
        evidence = analyze_colors(img, mask)
        assert 'black' in evidence['colors']

    def test_brown_spots_detected(self):
        img = make_brown_spot_leaf()
        mask, _ = segment_leaf(img)
        evidence = analyze_colors(img, mask)
        assert 'brown' in evidence['colors']

    def test_background_not_counted(self):
        """Gray concrete around the leaf must not leak into disease colors."""
        img = make_healthy_leaf()
        mask, _ = segment_leaf(img)
        evidence = analyze_colors(img, mask)
        assert 'gray' not in evidence['colors']
        assert 'white' not in evidence['colors']

    def test_detect_colors_backward_compatible_signature(self):
        img = make_healthy_leaf()
        mask = get_leaf_mask(img)
        colors, disease_mask, leaf_px = detect_colors(img, mask)
        assert isinstance(colors, list)
        assert disease_mask.shape == img.shape[:2]
        assert leaf_px > 0


class TestRealSamples:
    """The repository ships real photos: leaves on gray concrete. Healthy
    samples must not produce disease color indicators (Issue 3)."""

    @pytest.mark.parametrize('path', sorted(glob.glob(
        os.path.join(REPO_ROOT, 'samples', '*', 'healthy', '*.jpg'))))
    def test_healthy_samples_no_false_achromatic_indicators(self, path):
        """Issue 3 regression: healthy leaves must not produce false black /
        gray / white disease indicators from shadows, background, specular
        highlights or normal dark pigmentation.

        Note: a nominally 'healthy' sample may still contain *real* minor brown
        blemishes (the training notebook documented label noise in the healthy
        folders). Detecting genuine brown/yellow evidence is correct behavior;
        the route's healthy-consistency layer decides how it is displayed. So
        this test forbids only the false achromatic indicators that were the
        reported bug, not real chromatic lesions.
        """
        img = cv2.resize(cv2.imread(path), (256, 256))
        mask, info = segment_leaf(img)
        assert info['is_leaf'], f'leaf not segmented: {path}'
        evidence = analyze_colors(img, mask)
        false_indicators = set(evidence['colors']) & {'black', 'gray', 'white'}
        assert not false_indicators, (
            f'{os.path.basename(path)} healthy but got false achromatic '
            f'indicators {false_indicators} (ratios={evidence["ratios"]})'
        )

    @pytest.mark.parametrize('path', sorted(glob.glob(
        os.path.join(REPO_ROOT, 'samples', '*', 'unhealthy', '*.jpg'))))
    def test_unhealthy_samples_segmented(self, path):
        img = cv2.resize(cv2.imread(path), (256, 256))
        mask, info = segment_leaf(img)
        assert info['is_leaf'], f'leaf not segmented: {path}'
        # Unhealthy samples should produce *some* evidence (not a strict
        # requirement on which color, since labels are coarse).
        evidence = analyze_colors(img, mask)
        assert evidence['leaf_pixel_count'] > 0
