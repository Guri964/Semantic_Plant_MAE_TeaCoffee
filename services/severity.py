import numpy as np
from config import SEVERITY_MILD_THRESHOLD, SEVERITY_MODERATE_THRESHOLD


def calculate_severity(disease_mask, leaf_pixel_count):
    if leaf_pixel_count == 0:
        return 0.0, 'Mild'

    diseased_pixels = np.sum(disease_mask > 0)
    severity_pct = (diseased_pixels / leaf_pixel_count) * 100

    if severity_pct < SEVERITY_MILD_THRESHOLD:
        severity_label = 'Mild'
    elif severity_pct < SEVERITY_MODERATE_THRESHOLD:
        severity_label = 'Moderate'
    else:
        severity_label = 'Severe'

    return round(severity_pct, 2), severity_label