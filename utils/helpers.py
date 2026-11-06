import os
import uuid
from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file):
    ext         = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path   = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)
    return save_path, unique_name


def format_prediction_response(crop, crop_confidence, health, health_confidence,
                                pattern, colors, severity, severity_pct,
                                kb_result, recommendation,
                                pattern_confidence=None, pattern_reason=None,
                                color_evidence=None, leaf_info=None,
                                crop_ood=None, notes=None):
    """Build the /predict result payload.

    The original keys are unchanged (backward compatible). New optional keys
    expose the evidence behind the diagnosis: pattern confidence/reason,
    per-color area ratios, leaf segmentation quality, crop OOD scores, and
    advisory notes. Older clients simply ignore them.
    """
    result = {
        'predicted_crop':     crop,
        'crop_confidence':    crop_confidence,
        'predicted_health':   health,
        'health_confidence':  health_confidence,
        'predicted_pattern':  pattern,
        'detected_colors':    colors,
        'severity':           severity,
        'severity_pct':       severity_pct,
        'predicted_disease':  kb_result['disease'],
        'description':        kb_result['description'],
        'treatment':          kb_result['treatment'],
        'match_score':        kb_result['match_score'],
        'recommendation':     recommendation,
    }
    if pattern_confidence is not None:
        result['pattern_confidence'] = pattern_confidence
    if pattern_reason is not None:
        result['pattern_reason'] = pattern_reason
    if color_evidence is not None:
        result['color_evidence'] = color_evidence
    if leaf_info is not None:
        result['leaf_info'] = leaf_info
    if crop_ood is not None:
        result['crop_ood'] = crop_ood
    if notes:
        result['notes'] = notes
    return result