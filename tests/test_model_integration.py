"""Real-model integration tests.

Skipped automatically when the trained checkpoints are Git LFS pointers (no
real weights in the environment). Run `git lfs pull` to enable them. These
verify model loading, preprocessing consistency and inference on the shipped
sample images. They do NOT assert avocado rejection reliability — that needs
representative unsupported-species images and calibrated thresholds (see
scripts/calibrate_crop_ood.py and docs/DIAGNOSIS_FIXES.md).
"""

import glob
import os

import pytest
import torch

import config
from conftest import requires_real_weights, REAL_WEIGHTS_AVAILABLE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestModelLoaderGuard:
    def test_lfs_pointer_raises_helpful_error(self, tmp_path):
        from services.model_loader import _check_lfs_pointer
        p = os.path.join(str(tmp_path), 'fake.pth')
        with open(p, 'wb') as f:
            f.write(b'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n')
        with pytest.raises(RuntimeError, match='Git LFS pointer'):
            _check_lfs_pointer(p)

    def test_missing_file_raises(self, tmp_path):
        from services.model_loader import _check_lfs_pointer
        with pytest.raises(FileNotFoundError):
            _check_lfs_pointer(os.path.join(str(tmp_path), 'nope.pth'))

    def test_health_model_architecture_no_download(self):
        """HealthModel must construct offline (weights=None) with the same
        state_dict keys as the trained checkpoint expects."""
        from services.model_loader import HealthModel, CropModel
        hm = HealthModel(num_classes=2)
        cm = CropModel(num_classes=2)
        keys = set(hm.state_dict().keys())
        assert any(k.startswith('efficientnet.') for k in keys)
        assert any(k.startswith('vit.') for k in keys)
        assert any(k.startswith('fusion.') for k in keys)
        assert any(k.startswith('backbone.') for k in set(cm.state_dict().keys()))


@requires_real_weights
class TestRealModels:
    @pytest.fixture(scope='class')
    def models(self):
        from services.model_loader import load_models
        return load_models()

    def test_models_load_and_eval(self, models):
        crop, health, device = models
        assert not crop.training and not health.training

    def test_preprocessing_shape(self, models):
        from services.preprocessing import prepare_tensor
        crop, health, device = models
        sample = glob.glob(os.path.join(REPO_ROOT, 'samples', '*', '*', '*.jpg'))[0]
        t = prepare_tensor(sample, device)
        assert t.shape == (1, 3, 224, 224)

    @pytest.mark.parametrize('path', sorted(glob.glob(
        os.path.join(REPO_ROOT, 'samples', '*', '*', '*.jpg'))))
    def test_crop_prediction_on_samples(self, models, path):
        from services.preprocessing import prepare_tensor
        from services.crop_ood import infer_crop, compute_crop_scores
        crop, health, device = models
        t = prepare_tensor(path, device)
        logits, _ = infer_crop(crop, t, need_features=False)
        assert logits.shape == (1, 2)
        s = compute_crop_scores(logits[0])
        assert 0.0 <= s['msp'] <= 1.0
        # The shipped samples are tea/coffee; record the predicted class.
        idx = int(logits.argmax(dim=1).item())
        assert config.CROP_CLASSES[idx] in ('coffee', 'tea')

    @pytest.mark.parametrize('path', sorted(glob.glob(
        os.path.join(REPO_ROOT, 'samples', '*', '*', '*.jpg'))))
    def test_health_prediction_on_samples(self, models, path):
        from services.preprocessing import prepare_tensor
        crop, health, device = models
        t = prepare_tensor(path, device)
        with torch.no_grad():
            logits = health(t)
        assert logits.shape == (1, 2)
        idx = int(logits.argmax(dim=1).item())
        assert config.HEALTH_CLASSES[idx] in ('healthy', 'unhealthy')
