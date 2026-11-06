#!/usr/bin/env python3
"""Calibrate the crop OOD gate on held-out validation data.

The crop model is a closed-set binary (coffee/tea) classifier. This script
measures how the *real trained model* scores in-distribution validation leaves
and derives thresholds that retain a chosen fraction of them (default 99%):

  * msp_threshold    = quantile(val MSP, 1 - id_retention)
  * energy_threshold = quantile(val energy, id_retention)
  * knn_threshold    = quantile(leave-one-out kNN distance, id_retention)

It also stores the L2-normalized validation feature bank used by the kNN
signal at inference time.

IMPORTANT LIMITATIONS (read before trusting the gate):
  * This calibrates ID *retention*, not OOD *rejection*. Reliable rejection
    rates can only be measured with representative unsupported-crop images
    (avocado and other species). Pass --ood-dir to measure rejection on such
    a folder if you have one; nothing is downloaded by this script.
  * Thresholds are only valid for the exact checkpoint + preprocessing they
    were computed with. Re-run after any model/preprocessing change.

Usage (from the repository root, with real LFS weights fetched):

    python scripts/calibrate_crop_ood.py \
        --val-csv models/val.csv \
        --dataset-root /path/to/coffee-and-tea-dataset \
        [--ood-dir /path/to/unsupported_leaf_images] \
        [--id-retention 0.99] [--k 5]

The dataset root must contain COFFEE/COFFEE/{HEALTHY,UNHEALTHY} and
TEA/TEA/{HEALTHY,UNHEALTHY} so that the Kaggle paths in val.csv can be
remapped by basename. Alternatively pass --images-dir pointing at a flat
folder containing the validation images.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    CROP_CLASSES,
    CROP_OOD_CALIBRATION_PATH,
)
from services.model_loader import load_models  # noqa: E402
from services.preprocessing import prepare_tensor  # noqa: E402
from services.crop_ood import compute_crop_scores, extract_features  # noqa: E402


def _resolve_path(csv_path, dataset_root, images_dir):
    base = os.path.basename(csv_path)
    if images_dir:
        candidate = os.path.join(images_dir, base)
        return candidate if os.path.exists(candidate) else None
    if dataset_root:
        # csv path: .../COFFEE/COFFEE/HEALTHY/<name>.jpg
        parts = csv_path.replace('\\', '/').split('/')
        try:
            i = parts.index('COFFEE') if 'COFFEE' in parts else parts.index('TEA')
            rel = os.path.join(*parts[i:])
            candidate = os.path.join(dataset_root, rel)
            if os.path.exists(candidate):
                return candidate
        except ValueError:
            pass
        # fallback: search by basename
        for root, _dirs, files in os.walk(dataset_root):
            if base in files:
                return os.path.join(root, base)
    return csv_path if os.path.exists(csv_path) else None


def _knn_distances(bank, k):
    """Leave-one-out mean cosine distance to k nearest neighbours."""
    sims = bank @ bank.T
    np.fill_diagonal(sims, -np.inf)
    dists = []
    for i in range(len(bank)):
        top = np.sort(sims[i])[::-1][:k]
        dists.append(1.0 - top.mean())
    return np.array(dists)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--val-csv', default='models/val.csv')
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--images-dir', default=None)
    ap.add_argument('--ood-dir', default=None,
                    help='Optional folder of unsupported-crop images to measure rejection.')
    ap.add_argument('--id-retention', type=float, default=0.99)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default=CROP_OOD_CALIBRATION_PATH)
    args = ap.parse_args()

    crop_model, _health, device = load_models()
    crop_model.eval()

    rows = list(csv.DictReader(open(args.val_csv)))
    msps, energies, features, labels = [], [], [], []
    missing = 0
    for row in rows:
        path = _resolve_path(row['path'], args.dataset_root, args.images_dir)
        if path is None:
            missing += 1
            continue
        tensor = prepare_tensor(path, device)
        with torch.no_grad():
            logits = crop_model(tensor)
        feats = extract_features(crop_model, tensor)
        scores = compute_crop_scores(logits[0])
        msps.append(scores['msp'])
        energies.append(scores['energy'])
        features.append(feats)
        labels.append(row['crop'])

    if len(msps) < 20:
        print(f'ERROR: only {len(msps)} validation images resolved '
              f'({missing} missing). Provide --dataset-root or --images-dir.')
        sys.exit(1)

    msps = np.array(msps)
    energies = np.array(energies)
    bank = np.array(features, dtype=np.float64)
    bank = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)

    msp_thr = float(np.quantile(msps, 1.0 - args.id_retention))
    energy_thr = float(np.quantile(energies, args.id_retention))
    knn_dists = _knn_distances(bank, args.k)
    knn_thr = float(np.quantile(knn_dists, args.id_retention))

    bank_path = os.path.join(os.path.dirname(args.out), 'crop_ood_features.npy')
    np.save(bank_path, bank.astype(np.float32))

    calib = {
        'msp_threshold': round(msp_thr, 6),
        'energy_threshold': round(energy_thr, 6),
        'knn_threshold': round(knn_thr, 6),
        'knn_k': args.k,
        'feature_bank_path': os.path.relpath(bank_path, os.path.dirname(args.out)),
        'calibrated_on': args.val_csv,
        'id_retention': args.id_retention,
        'n_validation': len(msps),
        'class_order': CROP_CLASSES,
    }

    print(f'Validation images scored : {len(msps)} (missing: {missing})')
    print(f'MSP    min/median/max    : {msps.min():.4f} / {np.median(msps):.4f} / {msps.max():.4f}')
    print(f'Energy min/median/max    : {energies.min():.3f} / {np.median(energies):.3f} / {energies.max():.3f}')
    print(f'kNN dist median/max      : {np.median(knn_dists):.4f} / {knn_dists.max():.4f}')
    print(f'Thresholds @ {args.id_retention:.0%} ID retention:')
    print(f'  msp >= {msp_thr:.4f}, energy <= {energy_thr:.3f}, knn <= {knn_thr:.4f}')

    if args.ood_dir:
        rejected = total = 0
        for name in sorted(os.listdir(args.ood_dir)):
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(args.ood_dir, name)
            tensor = prepare_tensor(path, device)
            with torch.no_grad():
                logits = crop_model(tensor)
            feats = extract_features(crop_model, tensor)
            s = compute_crop_scores(logits[0], features=feats, feature_bank=bank, k=args.k)
            ok = (s['msp'] >= msp_thr and s['energy'] <= energy_thr
                  and s['knn_distance'] <= knn_thr)
            total += 1
            rejected += (not ok)
            print(f'  OOD {name}: msp={s["msp"]:.3f} energy={s["energy"]:.2f} '
                  f'knn={s["knn_distance"]:.3f} -> {"ACCEPTED (miss!)" if ok else "rejected"}')
        if total:
            print(f'OOD rejection rate: {rejected}/{total} = {rejected/total:.1%}')
        calib['ood_rejection_measured'] = {'rejected': rejected, 'total': total}

    with open(args.out, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f'Calibration written to {args.out}')
    print(f'Feature bank written to {bank_path}')


if __name__ == '__main__':
    main()
