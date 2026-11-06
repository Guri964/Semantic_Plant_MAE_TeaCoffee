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
                                kb_result, recommendation):
    return {
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