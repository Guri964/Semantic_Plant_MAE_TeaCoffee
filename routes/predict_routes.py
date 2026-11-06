import torch
from flask import Blueprint, request, session, jsonify, render_template, redirect, url_for, current_app
from functools import wraps
from utils.helpers import allowed_file, save_uploaded_file, format_prediction_response
from services.preprocessing import load_image_bgr, prepare_tensor
from services.color_filter import get_leaf_mask, detect_colors
from services.severity import calculate_severity
from services.knowledge_base import match_disease
from services.llm_service import get_recommendation
from database import log_prediction
from config import CROP_CLASSES, HEALTH_CLASSES, CROP_CONFIDENCE_THRESHOLD

predict_bp = Blueprint('predict', __name__)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated


@predict_bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session.get('username'))


@predict_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file type. Use JPG or PNG.'}), 400

    image_path, _ = save_uploaded_file(file)

    crop_model   = current_app.config['CROP_MODEL']
    health_model = current_app.config['HEALTH_MODEL']
    device       = current_app.config['DEVICE']

    img_tensor = prepare_tensor(image_path, device)

    with torch.no_grad():
        crop_output = crop_model(img_tensor)
        crop_probs  = torch.softmax(crop_output, dim=1)
        crop_idx    = crop_output.argmax(dim=1).item()
        crop_conf   = round(crop_probs[0][crop_idx].item() * 100, 2)
        crop_pred   = CROP_CLASSES[crop_idx]

        if crop_conf < CROP_CONFIDENCE_THRESHOLD:
            return jsonify({
                'success': False,
                'message': 'Crop not supported or image not recognized. Please upload a clear tea or coffee leaf image.'
            }), 200

        health_output = health_model(img_tensor)
        health_probs  = torch.softmax(health_output, dim=1)
        health_idx    = health_output.argmax(dim=1).item()
        health_conf   = round(health_probs[0][health_idx].item() * 100, 2)
        health_pred   = HEALTH_CLASSES[health_idx]

    img_bgr = load_image_bgr(image_path)
    leaf_mask = get_leaf_mask(img_bgr)
    detected_colors, disease_mask, leaf_pixel_count = detect_colors(img_bgr, leaf_mask)
    severity_pct, severity_label = calculate_severity(disease_mask, leaf_pixel_count)

    kb_result = match_disease(crop_pred, health_pred, detected_colors)

    recommendation = get_recommendation(
        crop_pred, health_pred,
        kb_result['disease'],
        severity_label,
        detected_colors
    )

    log_prediction(
        user_id          = session['user_id'],
        image_path       = image_path,
        predicted_crop   = crop_pred,
        predicted_health = health_pred,
        predicted_pattern= kb_result['pattern'],
        detected_colors  = detected_colors,
        predicted_disease= kb_result['disease'],
        severity         = severity_label,
        severity_pct     = severity_pct,
        match_score      = kb_result['match_score'],
        recommendation   = recommendation
    )

    response = format_prediction_response(
        crop             = crop_pred,
        crop_confidence  = crop_conf,
        health           = health_pred,
        health_confidence= health_conf,
        pattern          = kb_result['pattern'],
        colors           = detected_colors,
        severity         = severity_label,
        severity_pct     = severity_pct,
        kb_result        = kb_result,
        recommendation   = recommendation
    )

    return jsonify({'success': True, 'result': response})