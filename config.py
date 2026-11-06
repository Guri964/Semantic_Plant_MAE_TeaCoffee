import os
import secrets
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_KEY = secrets.token_hex(32)

DATABASE_PATH = os.path.join(BASE_DIR, 'tea_coffee.db')

CROP_MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'crop_model_best.pth')
HEALTH_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'health_model_best.pth')

UPLOAD_FOLDER      = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

CROP_CLASSES   = ['coffee', 'tea']
HEALTH_CLASSES = ['healthy', 'unhealthy']

# ---------------------------------------------------------------------------
# Crop identification / out-of-distribution (OOD) gate
# ---------------------------------------------------------------------------
# Legacy single softmax gate (kept for backward compatibility / fallback).
CROP_CONFIDENCE_THRESHOLD = 60.0

# The crop model is a *closed-set* binary classifier (coffee vs tea). Softmax
# confidence alone is NOT evidence that an image belongs to either class, so
# unsupported crops (e.g. avocado) can score >90%. The gate below combines:
#   * max-softmax probability (MSP)
#   * energy score  E(x) = -logsumexp(logits)   (lower = more in-distribution)
#   * kNN distance in the backbone feature space (optional, needs calibration)
# NOTE: for a 2-class model, prediction margin and normalized entropy are
# strictly monotone functions of the max-softmax probability, so they add no
# independent signal. They are computed for reporting only.
#
# Default MSP threshold is provisional and conservative. Reliable thresholds
# MUST be produced by scripts/calibrate_crop_ood.py on held-out validation
# data with the real trained weights (see docs/DIAGNOSIS_FIXES.md).
CROP_OOD_MSP_THRESHOLD = 0.90          # provisional; overridden by calibration
CROP_OOD_ENERGY_THRESHOLD = None       # None = disabled until calibrated
CROP_OOD_KNN_THRESHOLD = None          # None = disabled until calibrated
CROP_OOD_KNN_K = 5
CROP_OOD_CALIBRATION_PATH = os.path.join(BASE_DIR, 'models', 'crop_ood_calibration.json')

# ---------------------------------------------------------------------------
# Leaf segmentation / image validity gate
# ---------------------------------------------------------------------------
# Minimum fraction of the frame covered by the segmented leaf for the image
# to be considered a leaf photo at all (non-leaf rejection).
LEAF_MIN_COVERAGE = 0.02
# Minimum fraction of the segmented leaf that must be green leaf tissue.
# Prevents background objects / non-leaf images from passing the gate.
LEAF_MIN_GREEN_FRACTION = 0.20
# Advisory blur score (Laplacian variance). Reported, not used to reject.
BLUR_ADVISORY_THRESHOLD = 20.0

# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------
SEVERITY_MILD_THRESHOLD     = 15.0
SEVERITY_MODERATE_THRESHOLD = 40.0

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
KB_WEIGHTS   = {'crop': 0.5, 'health': 0.3, 'color': 0.2}
KB_MIN_SCORE = 0.4

# ---------------------------------------------------------------------------
# Disease-color evidence (HSV, OpenCV scale: H 0-179, S/V 0-255)
# ---------------------------------------------------------------------------
# Refined ranges. Key differences vs the legacy ranges:
#   * 'black' excludes green-hued shadows (see color_filter._exclude_shadows)
#   * 'white' excludes tiny specular highlights (see color_filter)
#   * 'gray' requires a higher area threshold because it is ambiguous
# Legacy COLOR_RANGES/COLOR_THRESHOLDS keys are preserved for compatibility.
COLOR_RANGES = {
    'brown': [([0,  50, 35],  [20, 255, 185]),
              ([18, 50, 60],  [25, 180, 160]),
              ([170, 50, 35], [179, 255, 185])],
    'black': [([0,  0,  0],   [179, 120, 80])],
    'yellow':[([18, 80, 120], [35, 255, 255])],
    'white': [([0,  0,  165], [179, 60,  240])],
    'gray':  [([0,  0,  70],  [179, 50,  195])],
}

# Minimum fraction of the *leaf area* a color must cover to be reported as
# disease evidence.
#
# Calibrated on the repository's shipped samples/ (6 healthy, 4 unhealthy) —
# the only validation images available in-repo (models/val.csv references a
# Kaggle dataset that is not bundled). Measured after segmentation + shadow/
# specular/pigmentation suppression:
#
#   color   healthy max   unhealthy max   chosen threshold
#   brown   0.0322*       0.0087          0.02   (*real blemish on a nominally
#                                                healthy leaf; see report)
#   black   0.0000        0.0120          0.01   (margin above healthy max 0)
#   yellow  0.0115        0.0107          0.04   (overlapping; conservative)
#   white   0.0054        0.0645          0.02   (~3.7x above healthy max)
#   gray    0.0101        0.0113          0.08   (ambiguous; kept high)
#
# IMPORTANT: 10 images is a small calibration set. These thresholds avoid the
# reported healthy-leaf false positives and improve sensitivity to clear
# white/black lesions, but they MUST be re-validated on a larger held-out set
# before being trusted broadly (see docs/DIAGNOSIS_FIXES.md).
COLOR_THRESHOLDS = {
    'brown':  0.02,
    'black':  0.01,
    'yellow': 0.04,
    'white':  0.02,
    'gray':   0.08,
}

# Legacy cap ratios (retained for compatibility; the evidence-based filter
# uses connected-component size filtering instead of arbitrary pixel caps).
COLOR_CAP_RATIOS = {
    'gray':  0.25,
    'black': 0.20,
}

# Minimum connected-component size for a color region to count as evidence:
# max(absolute px, fraction of leaf area). Removes speckle / noise.
COLOR_MIN_COMPONENT_ABS_PX = 40
COLOR_MIN_COMPONENT_LEAF_FRACTION = 0.002

# Color evidence is measured on the leaf *interior*: the segmentation boundary
# rim mixes leaf edge, anti-aliased background and cast shadow, which otherwise
# reads as gray/black disease. Erode this many pixels before color analysis.
LEAF_INTERIOR_ERODE_PX = 3

# Specular-highlight suppression for 'white': glossy leaves produce mirror-like
# reflections with blown-out cores (very high V, very low S). Matte powdery
# growth does not. White candidate pixels near such cores are treated as
# lighting artifacts, not disease.
SPECULAR_V_MIN = 240
SPECULAR_S_MAX = 30
SPECULAR_SUPPRESS_DILATE_PX = 7

# ---------------------------------------------------------------------------
# Visual pattern detection (evidence-based, decoupled from disease names)
# ---------------------------------------------------------------------------
# A pattern is reported only when its supporting color evidence exceeds the
# corresponding COLOR_THRESHOLDS entry. If two or more categories are strong,
# the pattern is 'mixed'. If the leaf is unhealthy but no category has enough
# evidence, the pattern is 'inconclusive' (never invented from a disease name).
PATTERN_MIXED_MIN_CATEGORIES = 2
PATTERN_CONFIDENCE_SATURATION_RATIO = 0.25   # ratio at which confidence -> 1.0

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL   = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')
