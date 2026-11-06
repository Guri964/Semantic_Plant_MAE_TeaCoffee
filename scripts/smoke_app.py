#!/usr/bin/env python3
"""Application smoke-test entry point.

Boots the real Flask app (routes, DB, templates, color pipeline, KB) and
exercises the full upload -> prediction flow over HTTP.

Model handling:
  * If the trained checkpoints are real files, they are loaded normally.
  * If they are Git LFS pointers (no weights fetched), lightweight MOCK models
    are injected so the application plumbing can still be smoke-tested. A
    prominent warning is printed; mock predictions are NOT real diagnoses.

Run:  python scripts/smoke_app.py [--port 5000]
This is a development/smoke tool, not a production server.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def _weights_available():
    from config import CROP_MODEL_PATH, HEALTH_MODEL_PATH
    for p in (CROP_MODEL_PATH, HEALTH_MODEL_PATH):
        try:
            with open(p, 'rb') as f:
                if f.read(32).startswith(b'version https://git-lfs'):
                    return False
        except OSError:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    import app as app_module

    if not _weights_available():
        print('=' * 70)
        print('WARNING: trained weights are Git LFS pointers. Injecting MOCK')
        print('models for plumbing smoke-test only. Predictions are NOT real.')
        print('=' * 70)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'tests'))
        from conftest import MockCropModelWithFeatures, MockHealthModel
        crop = MockCropModelWithFeatures(logits=(6.0, -6.0))
        health = MockHealthModel(logits=(6.0, -6.0))
        app_module.load_models = lambda: (crop, health, torch.device('cpu'))
        app_module.load_calibration = lambda: None

    application = app_module.create_app()
    application.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
