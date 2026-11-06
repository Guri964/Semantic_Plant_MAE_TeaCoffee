"""Leaf segmentation and disease-color evidence extraction.

Root causes fixed here (see docs/DIAGNOSIS_FIXES.md):

1. The legacy ``get_leaf_mask`` fell back to a global gray threshold whenever
   green coverage was low. On the project's photos (leaves on gray concrete)
   that fallback admitted the *entire background* as "leaf", so concrete and
   cast shadows were reported as black/gray disease regions.

2. The legacy mask was green-only, so yellow/brown/white lesions *on* the leaf
   were excluded from the leaf mask and therefore never counted as color
   evidence (``detect_colors`` ANDs color masks with the leaf mask).

3. Legacy HSV ranges matched shaded dark-green tissue ('black'), specular
   highlights ('white') and desaturated background ('gray') on healthy leaves.

The new pipeline:
  * segments the leaf from green tissue, keeps significant components and
    fills enclosed holes so internal lesions belong to the leaf area;
  * never falls back to "everything that is not black background";
  * extracts per-color evidence masks with shadow/highlight suppression and
    connected-component size filtering;
  * reports a color only when it covers a meaningful fraction of the leaf.

Public API (backward compatible):
  * ``get_leaf_mask(img_bgr) -> mask``
  * ``detect_colors(img_bgr, leaf_mask) -> (colors, disease_mask, leaf_px)``
New:
  * ``segment_leaf(img_bgr) -> (mask, info)``
  * ``analyze_colors(img_bgr, leaf_mask) -> evidence dict``
"""

import cv2
import numpy as np

from config import (
    COLOR_RANGES,
    COLOR_THRESHOLDS,
    COLOR_MIN_COMPONENT_ABS_PX,
    COLOR_MIN_COMPONENT_LEAF_FRACTION,
    LEAF_MIN_COVERAGE,
    LEAF_MIN_GREEN_FRACTION,
    LEAF_INTERIOR_ERODE_PX,
    SPECULAR_V_MIN,
    SPECULAR_S_MAX,
    SPECULAR_SUPPRESS_DILATE_PX,
    BLUR_ADVISORY_THRESHOLD,
)

# Vegetation hue band (OpenCV H is 0-179). Includes yellow-green senescent
# tissue but excludes gray concrete (S too low) and brown soil (H too low).
_GREEN_LOWER = np.array([18, 25, 25])
_GREEN_UPPER = np.array([95, 255, 255])


def _green_mask(hsv):
    return cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)


def segment_leaf(img_bgr):
    """Segment the leaf region and return ``(mask, info)``.

    The mask is built from green vegetation, cleaned morphologically, reduced
    to significant connected components, and hole-filled so that non-green
    lesions *inside* the leaf contour count as leaf area.

    ``info`` contains:
      * ``leaf_pixel_count`` – pixels in the returned mask
      * ``coverage``         – leaf area / frame area
      * ``green_fraction``   – green tissue pixels / leaf area
      * ``is_leaf``          – passes the minimum coverage / green checks
      * ``blur_score``       – Laplacian variance (advisory only)
    """
    frame_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green = _green_mask(hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaf_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        largest = max(areas)
        # Keep the dominant leaf blob(s); discard specks. The relative cutoff
        # allows a second leaf in frame while rejecting background noise.
        min_area = max(200.0, 0.20 * largest)
        kept = [c for c, a in zip(contours, areas) if a >= min_area]
        # RETR_EXTERNAL + FILLED fills enclosed holes (yellow/brown/white
        # lesions surrounded by green tissue become part of the leaf area).
        cv2.drawContours(leaf_mask, kept, -1, 255, thickness=cv2.FILLED)
        leaf_mask = cv2.morphologyEx(
            leaf_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

    leaf_pixel_count = int(np.count_nonzero(leaf_mask))
    coverage = leaf_pixel_count / frame_pixels if frame_pixels else 0.0
    green_in_leaf = int(np.count_nonzero(cv2.bitwise_and(green, leaf_mask)))
    green_fraction = green_in_leaf / leaf_pixel_count if leaf_pixel_count else 0.0

    is_leaf = (
        coverage >= LEAF_MIN_COVERAGE
        and green_fraction >= LEAF_MIN_GREEN_FRACTION
    )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    info = {
        'leaf_pixel_count': leaf_pixel_count,
        'coverage': round(coverage, 4),
        'green_fraction': round(green_fraction, 4),
        'is_leaf': bool(is_leaf),
        'blur_score': round(blur_score, 2),
        'blur_advisory': bool(blur_score < BLUR_ADVISORY_THRESHOLD),
    }
    return leaf_mask, info


def get_leaf_mask(img_bgr):
    """Backward-compatible wrapper returning only the leaf mask.

    Unlike the legacy version there is no gray-threshold fallback: if no
    credible leaf is found the mask is empty and callers must treat the
    image as invalid / unsupported rather than analyzing the background.
    """
    mask, _ = segment_leaf(img_bgr)
    return mask


def _color_raw_mask(hsv, color_name):
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_RANGES[color_name]:
        m = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined = cv2.bitwise_or(combined, m)
    return combined


def _interior_mask(leaf_mask):
    """Erode the segmentation boundary rim used for color analysis.

    The outer few pixels of a segmented leaf mix leaf edge, anti-aliased
    background and cast shadow; they are not reliable disease evidence.
    """
    if LEAF_INTERIOR_ERODE_PX <= 0:
        return leaf_mask
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * LEAF_INTERIOR_ERODE_PX + 1, 2 * LEAF_INTERIOR_ERODE_PX + 1),
    )
    return cv2.erode(leaf_mask, k)


def _suppress_specular(hsv, mask):
    """Remove white candidate pixels that are specular highlights.

    Glossy leaves reflect light as bright, desaturated bands with blown-out
    cores (V very high, S very low). Matte powdery growth lacks such cores.
    White pixels within a small neighbourhood of a specular core are treated
    as lighting artifacts.
    """
    core = cv2.inRange(
        hsv,
        np.array([0, 0, SPECULAR_V_MIN]),
        np.array([179, SPECULAR_S_MAX, 255]),
    )
    if not np.any(core):
        return mask
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * SPECULAR_SUPPRESS_DILATE_PX + 1, 2 * SPECULAR_SUPPRESS_DILATE_PX + 1),
    )
    zone = cv2.dilate(core, k)
    return cv2.bitwise_and(mask, cv2.bitwise_not(zone))


def _suppress_green_pigmentation(hsv, mask):
    """Remove green-hued tissue from an achromatic disease-color mask.

    Black coatings (sooty mold) and gray lesions are *achromatic*: they lack a
    green hue. Normal leaf pigmentation — including dark, desaturated foliage
    and shadows cast on green tissue — retains a green hue even when its
    saturation/value is low. Excluding the green hue band therefore separates
    true achromatic disease regions from pigmentation and shading, instead of
    flagging an entire dark healthy leaf as black/gray disease.
    """
    green_hue = cv2.inRange(hsv, np.array([22, 0, 0]), np.array([95, 255, 255]))
    return cv2.bitwise_and(mask, cv2.bitwise_not(green_hue))


def _suppress_shadows(hsv, mask):
    """Remove shaded *green tissue* from a dark-region mask.

    Kept for clarity/back-compat; green-hue exclusion (above) is the broader
    principle and subsumes this for black regions.
    """
    green_shadow = cv2.inRange(hsv, np.array([22, 60, 20]), np.array([95, 255, 90]))
    return cv2.bitwise_and(mask, cv2.bitwise_not(green_shadow))


def _filter_small_components(mask, leaf_pixel_count):
    """Drop connected components too small to be credible lesion evidence."""
    min_px = max(
        COLOR_MIN_COMPONENT_ABS_PX,
        int(COLOR_MIN_COMPONENT_LEAF_FRACTION * leaf_pixel_count),
    )
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept_mask = np.zeros_like(mask)
    kept_areas = []
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_px:
            kept_mask[labels == lbl] = 255
            kept_areas.append(area)
    return kept_mask, kept_areas


def analyze_colors(img_bgr, leaf_mask):
    """Extract disease-color evidence restricted to the leaf area.

    Returns a dict::

        {
          'colors':            [names passing area threshold],
          'disease_mask':      union of evidence masks (uint8),
          'leaf_pixel_count':  int,
          'ratios':            {color: fraction of leaf area},
          'masks':             {color: evidence mask},
          'components':        {color: {'count', 'largest_fraction', 'areas'}},
        }

    Ratios are computed on the *filtered* evidence masks, so speckle noise and
    sub-threshold components do not inflate them.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    interior = _interior_mask(leaf_mask)
    leaf_pixel_count = int(np.count_nonzero(interior))

    colors = []
    ratios = {}
    masks = {}
    components = {}
    disease_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    for color_name in COLOR_RANGES:
        raw = _color_raw_mask(hsv, color_name)
        mask = cv2.bitwise_and(raw, interior)

        if color_name in ('black', 'gray'):
            # Achromatic disease colors must not include green-hued foliage
            # (normal pigmentation) or shadows on green tissue.
            mask = _suppress_green_pigmentation(hsv, mask)
        elif color_name == 'white':
            mask = _suppress_specular(hsv, mask)

        mask, areas = _filter_small_components(mask, leaf_pixel_count)

        pixel_count = int(np.count_nonzero(mask))
        ratio = pixel_count / leaf_pixel_count if leaf_pixel_count else 0.0
        ratios[color_name] = round(ratio, 4)
        masks[color_name] = mask
        largest_fraction = (max(areas) / pixel_count) if pixel_count and areas else 0.0
        components[color_name] = {
            'count': len(areas),
            'largest_fraction': round(largest_fraction, 4),
            'areas': sorted(areas, reverse=True),
        }

        if ratio > COLOR_THRESHOLDS[color_name]:
            colors.append(color_name)
            disease_mask = cv2.bitwise_or(disease_mask, mask)

    return {
        'colors': colors,
        'disease_mask': disease_mask,
        'leaf_pixel_count': leaf_pixel_count,
        'ratios': ratios,
        'masks': masks,
        'components': components,
    }


def detect_colors(img_bgr, leaf_mask):
    """Backward-compatible API: ``(colors, disease_mask, leaf_pixel_count)``.

    Now returns only evidence-backed colors (area threshold + component
    filtering + shadow suppression) instead of raw HSV range hits.
    """
    evidence = analyze_colors(img_bgr, leaf_mask)
    return evidence['colors'], evidence['disease_mask'], evidence['leaf_pixel_count']
