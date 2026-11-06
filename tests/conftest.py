"""Shared pytest fixtures.

The trained checkpoints are Git LFS objects. In environments where only the
LFS pointers are present (CI without LFS), model-dependent integration tests
are skipped and application regression tests run against lightweight mock
models that expose the same interface (``backbone`` + ``classifier`` for the
crop model, callable for the health model).
"""

import importlib
import os
import sys

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic image generators (256x256, matching the color pipeline input)
# ---------------------------------------------------------------------------
SZ = 256


def _base_background():
    """Gray concrete-like background (low saturation)."""
    img = np.full((SZ, SZ, 3), 160, dtype=np.uint8)
    noise = np.random.default_rng(0).integers(-8, 8, (SZ, SZ, 1))
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _leaf_ellipse(img, color=(0, 128, 0)):
    cv2.ellipse(img, (128, 128), (95, 55), 0, 0, 360, color, -1)
    return img


def make_healthy_leaf():
    return _leaf_ellipse(_base_background(), (0, 128, 0))


def make_yellow_patch_leaf():
    img = make_healthy_leaf()
    cv2.ellipse(img, (100, 120), (28, 20), 0, 0, 360, (0, 200, 200), -1)
    cv2.ellipse(img, (160, 140), (20, 15), 0, 0, 360, (0, 210, 210), -1)
    return img


def make_white_powder_leaf():
    img = make_healthy_leaf()
    cv2.ellipse(img, (110, 120), (26, 18), 0, 0, 360, (225, 225, 225), -1)
    cv2.ellipse(img, (160, 135), (18, 14), 0, 0, 360, (230, 230, 230), -1)
    return img


def make_black_coating_leaf():
    img = make_healthy_leaf()
    cv2.ellipse(img, (120, 125), (35, 25), 0, 0, 360, (18, 18, 18), -1)
    return img


def make_brown_spot_leaf():
    img = make_healthy_leaf()
    for cx, cy in [(90, 115), (130, 140), (165, 115), (110, 150)]:
        cv2.circle(img, (cx, cy), 11, (40, 80, 140), -1)
    return img


def make_shadow_leaf():
    """Healthy green leaf with a dark desaturated shadow band across it."""
    img = make_healthy_leaf()
    overlay = img.copy()
    cv2.ellipse(overlay, (128, 150), (90, 22), 0, 0, 360, (35, 55, 40), -1)
    return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)


def make_non_leaf():
    return _base_background()


def make_blurry_leaf():
    img = make_healthy_leaf()
    return cv2.GaussianBlur(img, (31, 31), 0)


def save_img(tmp_path, img, name='leaf.png'):
    path = os.path.join(str(tmp_path), name)
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------------------
# Model availability
# ---------------------------------------------------------------------------
def _is_lfs_pointer(path):
    try:
        with open(path, 'rb') as f:
            return f.read(32).startswith(b'version https://git-lfs')
    except OSError:
        return True


REAL_WEIGHTS_AVAILABLE = (
    not _is_lfs_pointer(config.CROP_MODEL_PATH)
    and not _is_lfs_pointer(config.HEALTH_MODEL_PATH)
)

requires_real_weights = pytest.mark.skipif(
    not REAL_WEIGHTS_AVAILABLE,
    reason='Trained weights are Git LFS pointers in this environment; '
           'run `git lfs pull` to enable real-model integration tests.',
)


# ---------------------------------------------------------------------------
# Mock models (same interface as the real ones)
# ---------------------------------------------------------------------------
class MockCropModel(nn.Module):
    def __init__(self, logits=(6.0, -6.0)):
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self.backbone = nn.Identity()
        self.classifier = nn.Identity()

    def forward(self, x):
        # x is (1,3,224,224); produce a fixed feature then fixed logits.
        return self._logits.unsqueeze(0)


class MockCropModelWithFeatures(nn.Module):
    """Crop model whose backbone/classifier split matches infer_crop()."""
    def __init__(self, logits=(6.0, -6.0), feat_dim=8):
        super().__init__()
        self.register_buffer('_logits', torch.tensor(logits, dtype=torch.float32))
        self.register_buffer('_feat', torch.ones(feat_dim, dtype=torch.float32))

    def backbone(self, x):
        return self._feat.unsqueeze(0)

    def classifier(self, feats):
        return self._logits.unsqueeze(0)

    def forward(self, x):
        return self.classifier(self.backbone(x))


class MockHealthModel(nn.Module):
    def __init__(self, logits=(6.0, -6.0)):
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        return self._logits.unsqueeze(0)


# ---------------------------------------------------------------------------
# Flask app fixture with isolated DB/uploads and mock models
# ---------------------------------------------------------------------------
@pytest.fixture
def app(tmp_path, monkeypatch):
    db_path = os.path.join(str(tmp_path), 'test.db')
    upload_dir = os.path.join(str(tmp_path), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    monkeypatch.setattr(config, 'DATABASE_PATH', db_path, raising=False)
    monkeypatch.setattr(config, 'UPLOAD_FOLDER', upload_dir, raising=False)

    import database
    importlib.reload(database)
    monkeypatch.setattr(database, 'DATABASE_PATH', db_path, raising=False)

    import services.knowledge_base as kb
    monkeypatch.setattr(kb, 'DATABASE_PATH', db_path, raising=False)

    import utils.helpers as helpers
    monkeypatch.setattr(helpers, 'UPLOAD_FOLDER', upload_dir, raising=False)

    import app as app_module
    importlib.reload(app_module)

    crop = MockCropModelWithFeatures(logits=(6.0, -6.0))   # coffee, very confident
    health = MockHealthModel(logits=(6.0, -6.0))          # healthy
    monkeypatch.setattr(
        app_module, 'load_models',
        lambda: (crop, health, torch.device('cpu')),
    )
    monkeypatch.setattr(app_module, 'load_calibration', lambda: None)

    application = app_module.create_app()
    application.config['TESTING'] = True
    application.config['UPLOAD_FOLDER'] = upload_dir
    yield application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def logged_in_client(client):
    client.post('/signup', json={
        'username': 'tester', 'password': 'secret123', 'confirm_password': 'secret123',
    })
    client.post('/login', json={'username': 'tester', 'password': 'secret123'})
    return client


@pytest.fixture
def healthy_image_path(tmp_path):
    return save_img(tmp_path, make_healthy_leaf(), 'healthy.png')


@pytest.fixture
def yellow_image_path(tmp_path):
    return save_img(tmp_path, make_yellow_patch_leaf(), 'yellow.png')
