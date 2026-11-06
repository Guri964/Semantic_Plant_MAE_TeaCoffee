import torch
from flask import Blueprint, request, session, jsonify, render_template, redirect, url_for, current_app
from functools import wraps
from utils.helpers import allowed_file, save_uploaded_file, format_prediction_response
from services.preprocessing import load_image_bgr, prepare_tensor
from services.color_filter import segment_leaf, analyze_colors
from services.pattern_detection import detect_pattern
from services.crop_ood import compute_crop_scores, evaluate_crop_gate, infer_crop
from services.severity import calculate_severity
from services.knowledge_base import match_disease
from services.llm_service import get_recommendation
from database import log_prediction
from config import CROP_CLASSES, HEALTH_CLASSES

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
    calibration  = current_app.config.get('CROP_OOD_CALIBRATION')

    # ------------------------------------------------------------------
    # 1. Leaf validity gate (non-leaf / invalid image rejection).
    #    Segmentation no longer falls back to "analyze the whole frame",
    #    so images without a credible leaf are rejected here.
    # ------------------------------------------------------------------
    try:
        img_bgr = load_image_bgr(image_path)
    except ValueError:
        return jsonify({
            'success': False,
            'message': 'Invalid image file: could not be decoded. Use a valid JPG or PNG.',
            'reason':  'invalid_image',
        }), 400

    leaf_mask, leaf_info = segment_leaf(img_bgr)
    if not leaf_info['is_leaf']:
        return jsonify({
            'success': False,
            'message': 'Invalid image: no clear leaf detected. Please upload a '
                       'photo of a single tea or coffee leaf.',
            'reason':  'not_a_leaf',
        }), 200

    # ------------------------------------------------------------------
    # 2. Crop classification with out-of-distribution gating.
    #    A confident softmax alone is not proof of a supported crop.
    # ------------------------------------------------------------------
    img_tensor = prepare_tensor(image_path, device)

    feature_bank = (calibration or {}).get('feature_bank')
    need_features = feature_bank is not None
    crop_output, crop_features = infer_crop(crop_model, img_tensor, need_features)

    crop_probs = torch.softmax(crop_output, dim=1)
    crop_idx   = int(crop_output.argmax(dim=1).item())
    crop_conf  = round(crop_probs[0][crop_idx].item() * 100, 2)
    crop_pred  = CROP_CLASSES[crop_idx]

    knn_k = (calibration or {}).get('knn_k', 5)
    scores = compute_crop_scores(
        crop_output[0], features=crop_features,
        feature_bank=feature_bank, k=knn_k,
    )
    accepted, gate_reason, thresholds = evaluate_crop_gate(scores, calibration)
    if not accepted:
        return jsonify({
            'success': False,
            'message': 'Unsupported or unrecognized crop. This app only supports '
                       'tea and coffee leaves. Please upload a clear tea or '
                       'coffee leaf image.',
            'reason':  'unsupported_crop',
            'gate':    gate_reason,
        }), 200

    # ------------------------------------------------------------------
    # 3. Health classification.
    # ------------------------------------------------------------------
    with torch.no_grad():
        health_output = health_model(img_tensor)
        health_probs  = torch.softmax(health_output, dim=1)
        health_idx    = int(health_output.argmax(dim=1).item())
        health_conf   = round(health_probs[0][health_idx].item() * 100, 2)
        health_pred   = HEALTH_CLASSES[health_idx]

    # ------------------------------------------------------------------
    # 4. Color evidence + evidence-based pattern + severity.
    # ------------------------------------------------------------------
    evidence = analyze_colors(img_bgr, leaf_mask)
    pattern_result = detect_pattern(evidence, health_pred)

    notes = []
    if leaf_info.get('blur_advisory'):
        notes.append('Image appears blurry; results may be less reliable. '
                     'Consider retaking the photo in focus.')

    if health_pred == 'healthy':
        # Healthy-consistent output: no disease colors, no disease pattern,
        # no severity. Neutral visual evidence is still returned for
        # transparency, and strong contradictory color evidence is surfaced
        # as an advisory note rather than silently ignored.
        detected_colors = []
        severity_pct, severity_label = 0.0, 'None'
        strong = [c for c in evidence['colors'] if c in ('black', 'brown', 'yellow')]
        if strong:
            notes.append('Health model reports healthy, but some discolored '
                         'regions were observed (' + ', '.join(strong) +
                         '). Monitor the leaf and re-check if symptoms spread.')
        kb_result = match_disease(crop_pred, 'healthy', [])
        if kb_result['disease'] != 'Condition Not Identified':
            kb_result['disease'] = 'No Disease Detected'
        pattern = 'none'
    else:
        detected_colors = evidence['colors']
        severity_pct, severity_label = calculate_severity(
            evidence['disease_mask'], evidence['leaf_pixel_count']
        )
        kb_result = match_disease(crop_pred, health_pred, detected_colors)
        pattern = pattern_result['pattern']

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
        predicted_pattern= pattern,
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
        pattern          = pattern,
        colors           = detected_colors,
        severity         = severity_label,
        severity_pct     = severity_pct,
        kb_result        = kb_result,
        recommendation   = recommendation,
        pattern_confidence = pattern_result['confidence'],
        pattern_reason     = pattern_result['reason'],
        color_evidence     = evidence['ratios'],
        leaf_info          = leaf_info,
        crop_ood           = {'scores': scores, 'thresholds': thresholds},
        notes              = notes,
    )

    return jsonify({'success': True, 'result': response})
