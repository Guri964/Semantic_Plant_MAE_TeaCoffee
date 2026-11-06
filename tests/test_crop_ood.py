"""Crop OOD gate tests (Issue 2 root cause).

These test the gating logic with controlled logits/features. Real-model
rejection of avocado/unsupported species requires the trained weights and
representative OOD images; see tests/test_model_integration.py (skipped when
weights are LFS pointers) and scripts/calibrate_crop_ood.py.
"""

import json
import os

import numpy as np
import pytest
import torch

from services.crop_ood import (
    compute_crop_scores, evaluate_crop_gate, load_calibration, infer_crop,
)
from conftest import MockCropModelWithFeatures


class TestScores:
    def test_confident_logits_high_msp(self):
        s = compute_crop_scores(torch.tensor([6.0, -6.0]))
        assert s['msp'] > 0.99
        assert s['margin'] > 0.99
        assert s['entropy'] < 0.01

    def test_ambiguous_logits_low_msp_high_entropy(self):
        s = compute_crop_scores(torch.tensor([0.1, -0.1]))
        assert s['msp'] < 0.6
        assert s['entropy'] > 0.9

    def test_binary_margin_entropy_monotone_with_msp(self):
        """For 2 classes, margin and entropy are determined by MSP — they add
        no independent OOD signal (documented design note)."""
        prev_msp, prev_margin, prev_entropy = 0.5, 0.0, 1.0
        for t in np.linspace(0, 8, 20):
            s = compute_crop_scores([t, -t])
            assert s['msp'] >= prev_msp - 1e-6
            assert s['margin'] >= prev_margin - 1e-6
            assert s['entropy'] <= prev_entropy + 1e-6
            # margin == 2*msp - 1 for binary
            assert abs(s['margin'] - (2 * s['msp'] - 1)) < 1e-4
            prev_msp, prev_margin, prev_entropy = s['msp'], s['margin'], s['entropy']

    def test_energy_lower_for_stronger_logits(self):
        weak = compute_crop_scores([0.5, -0.5])['energy']
        strong = compute_crop_scores([6.0, -6.0])['energy']
        assert strong < weak  # more negative = more in-distribution evidence

    def test_knn_distance(self):
        bank = np.eye(4, dtype=np.float64)
        feat = np.array([1.0, 0.0, 0.0, 0.0])
        s = compute_crop_scores([6.0, -6.0], features=feat, feature_bank=bank, k=1)
        assert s['knn_distance'] == pytest.approx(0.0, abs=1e-6)
        far = np.array([0.0, 0.0, 0.0, 1.0])
        s2 = compute_crop_scores([6.0, -6.0], features=far, feature_bank=bank[:1], k=1)
        assert s2['knn_distance'] > 0.9


class TestGate:
    def test_confident_accepted_uncalibrated(self):
        s = compute_crop_scores([6.0, -6.0])
        ok, reason, thr = evaluate_crop_gate(s, None)
        assert ok and reason == 'ok'
        assert thr['calibrated'] is False

    def test_low_confidence_rejected(self):
        s = compute_crop_scores([0.3, -0.3])
        ok, reason, _ = evaluate_crop_gate(s, None)
        assert not ok and reason == 'low_confidence'

    def test_confident_but_ood_features_rejected(self):
        """The avocado case: softmax is confident, but the embedding is far
        from the validation feature bank -> rejected by the kNN signal."""
        bank = np.zeros((10, 4)); bank[:, 0] = 1.0
        feat = np.array([0.0, 0.0, 0.0, 1.0])  # orthogonal to bank
        s = compute_crop_scores([6.0, -6.0], features=feat, feature_bank=bank, k=3)
        assert s['msp'] > 0.99  # confident softmax...
        calib = {'msp_threshold': 0.90, 'knn_threshold': 0.5}
        ok, reason, _ = evaluate_crop_gate(s, calib)
        assert not ok and reason == 'ood_features'  # ...but rejected

    def test_high_energy_rejected(self):
        s = compute_crop_scores([6.0, -6.0])
        calib = {'msp_threshold': 0.90, 'energy_threshold': s['energy'] - 1.0}
        ok, reason, _ = evaluate_crop_gate(s, calib)
        assert not ok and reason == 'high_energy'

    def test_calibration_overrides_default(self):
        s = compute_crop_scores([2.0, -2.0])  # msp ~0.98
        ok_default, _, _ = evaluate_crop_gate(s, None)        # default 0.90 -> ok
        ok_strict, _, _ = evaluate_crop_gate(s, {'msp_threshold': 0.999})
        assert ok_default and not ok_strict


class TestCalibrationLoading:
    def test_missing_returns_none(self, tmp_path):
        assert load_calibration(os.path.join(str(tmp_path), 'nope.json')) is None

    def test_load_with_feature_bank(self, tmp_path):
        bank = np.eye(3, dtype=np.float32)
        np.save(os.path.join(str(tmp_path), 'bank.npy'), bank)
        calib = {'msp_threshold': 0.95, 'knn_threshold': 0.4,
                 'feature_bank_path': 'bank.npy'}
        p = os.path.join(str(tmp_path), 'calib.json')
        json.dump(calib, open(p, 'w'))
        loaded = load_calibration(p)
        assert loaded['msp_threshold'] == 0.95
        assert loaded['feature_bank'].shape == (3, 3)

    def test_corrupt_json_returns_none(self, tmp_path):
        p = os.path.join(str(tmp_path), 'bad.json')
        open(p, 'w').write('{not json')
        assert load_calibration(p) is None


class TestInferCrop:
    def test_single_pass_matches_forward(self):
        model = MockCropModelWithFeatures(logits=(4.0, -4.0))
        x = torch.zeros(1, 3, 224, 224)
        logits, feats = infer_crop(model, x, need_features=True)
        assert torch.allclose(logits, model(x))
        assert feats is not None and feats.shape == (8,)

    def test_no_features_when_not_needed(self):
        model = MockCropModelWithFeatures()
        logits, feats = infer_crop(model, torch.zeros(1, 3, 224, 224), need_features=False)
        assert feats is None
