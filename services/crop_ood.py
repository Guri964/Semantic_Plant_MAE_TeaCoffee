"""Out-of-distribution (OOD) gating for the closed-set crop classifier.

The trained crop model is a binary coffee/tea softmax classifier. A high
softmax probability does NOT prove the image is tea or coffee: unsupported
species (e.g. avocado) are routinely assigned >90% confidence by closed-set
networks. The legacy single 60% gate therefore could not reject them.

Signals used here (no retraining, no architecture changes):

* **MSP** – max softmax probability. Necessary but not sufficient.
* **Energy** – ``E(x) = -logsumexp(logits)``. Lower (more negative) means the
  model produced stronger overall evidence; OOD inputs tend to have higher
  energy. Requires calibration because logit scale is model-specific.
* **Feature-space kNN** – distance of the backbone embedding to its k nearest
  neighbours in a held-out *validation* feature bank. This is the strongest
  available signal: an avocado leaf embeds far from both tea and coffee
  clusters even when its softmax is confident. Requires a calibration artifact
  produced by ``scripts/calibrate_crop_ood.py``.

Note on margin/entropy: for a **2-class** softmax, the prediction margin
(``p1 - p2 = 2*MSP - 1``) and the normalized entropy are strictly monotone
functions of MSP, so thresholding them adds no independent information. They
are computed for reporting/transparency only; the gate decisions use MSP,
energy and kNN distance.

All thresholds come from a calibration JSON (see
``scripts/calibrate_crop_ood.py``). Uncalibrated fallbacks are conservative
and documented in ``docs/DIAGNOSIS_FIXES.md``.
"""

import json
import os

import numpy as np
import torch

from config import (
    CROP_OOD_MSP_THRESHOLD,
    CROP_OOD_ENERGY_THRESHOLD,
    CROP_OOD_KNN_THRESHOLD,
    CROP_OOD_KNN_K,
    CROP_OOD_CALIBRATION_PATH,
)


def compute_crop_scores(logits, features=None, feature_bank=None, k=CROP_OOD_KNN_K):
    """Compute OOD scores from raw crop-model logits.

    Parameters
    ----------
    logits : torch.Tensor or array-like, shape (C,) or (1, C)
    features : optional 1-D backbone embedding for the same image
    feature_bank : optional (N, D) array of L2-normalized validation embeddings
    k : neighbours for the kNN distance

    Returns dict with ``msp``, ``margin``, ``entropy`` (normalized 0-1),
    ``energy`` and ``knn_distance`` (None when no bank is available).
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)

    shifted = logits - logits.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()

    msp = float(probs.max())
    sorted_probs = np.sort(probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(len(probs))) if len(probs) > 1 else 0.0
    # energy = -logsumexp(logits), computed stably via the max-shift trick:
    energy = float(-(logits.max() + np.log(exp.sum())))

    knn_distance = None
    if features is not None and feature_bank is not None and len(feature_bank) > 0:
        feat = np.asarray(features, dtype=np.float64).reshape(-1)
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        bank = np.asarray(feature_bank, dtype=np.float64)
        sims = bank @ feat
        kk = min(k, len(bank))
        top = np.sort(sims)[::-1][:kk]
        knn_distance = float(1.0 - top.mean())  # mean cosine distance to k-NN

    return {
        'msp': round(msp, 6),
        'margin': round(margin, 6),
        'entropy': round(entropy, 6),
        'energy': round(energy, 6),
        'knn_distance': None if knn_distance is None else round(knn_distance, 6),
    }


def load_calibration(path=CROP_OOD_CALIBRATION_PATH):
    """Load a calibration artifact, or return None when absent/unreadable.

    Expected JSON schema (produced by scripts/calibrate_crop_ood.py)::

        {
          "msp_threshold": 0.987,
          "energy_threshold": -3.2,
          "knn_threshold": 0.41,
          "knn_k": 5,
          "feature_bank_path": "models/crop_ood_features.npy",
          "calibrated_on": "models/val.csv",
          "id_retention": 0.99
        }
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            calib = json.load(f)
    except (OSError, ValueError):
        return None

    bank_path = calib.get('feature_bank_path')
    if bank_path:
        if not os.path.isabs(bank_path):
            # Relative paths are resolved against the calibration file's own
            # directory (the script writes them that way).
            bank_path = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(path)), bank_path)
            )
        if os.path.exists(bank_path):
            try:
                calib['feature_bank'] = np.load(bank_path)
            except (OSError, ValueError):
                calib['feature_bank'] = None
        else:
            calib['feature_bank'] = None
    return calib


def evaluate_crop_gate(scores, calibration=None):
    """Decide whether the crop prediction may be trusted.

    Returns ``(accepted, reason, thresholds)``. ``reason`` is a machine-readable
    string: ``ok``, ``low_confidence``, ``high_energy`` or ``ood_features``.
    Thresholds actually applied are returned for logging/transparency.
    """
    calib = calibration or {}
    msp_thr = float(calib.get('msp_threshold', CROP_OOD_MSP_THRESHOLD))
    energy_thr = calib.get('energy_threshold', CROP_OOD_ENERGY_THRESHOLD)
    knn_thr = calib.get('knn_threshold', CROP_OOD_KNN_THRESHOLD)

    thresholds = {
        'msp_threshold': msp_thr,
        'energy_threshold': energy_thr,
        'knn_threshold': knn_thr,
        'calibrated': bool(calib),
    }

    if scores['msp'] < msp_thr:
        return False, 'low_confidence', thresholds
    if energy_thr is not None and scores['energy'] > float(energy_thr):
        return False, 'high_energy', thresholds
    if knn_thr is not None and scores.get('knn_distance') is not None:
        if scores['knn_distance'] > float(knn_thr):
            return False, 'ood_features', thresholds
    return True, 'ok', thresholds


def extract_features(crop_model, img_tensor):
    """Backbone embedding used for feature-space OOD (no classifier head)."""
    with torch.no_grad():
        feats = crop_model.backbone(img_tensor)
    return feats.detach().cpu().numpy().reshape(-1)


def infer_crop(crop_model, img_tensor, need_features=False):
    """Run the crop model once, returning ``(logits, features_np_or_None)``.

    ``CropModel.forward`` is exactly ``backbone`` followed by ``classifier``,
    so calling them separately in eval mode is numerically identical while
    avoiding a second backbone pass when features are needed for OOD.
    """
    with torch.no_grad():
        feats = crop_model.backbone(img_tensor)
        logits = crop_model.classifier(feats)
    features_np = feats.detach().cpu().numpy().reshape(-1) if need_features else None
    return logits, features_np
