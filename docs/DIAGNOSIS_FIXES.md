# Leaf Diagnosis Fixes — Technical Report

Project: **Semantic Plant MAE – Tea & Coffee Leaf Disease Detection**
Branch: `arena/cdfeb292-semantic-plant-mae-teacoffee` (the session branch is fixed
by the environment; the requested `fix/leaf-diagnosis-improvements` name could
not be created — all work is on the current branch, uncommitted, for review).

> **Priority honored:** correct diagnosis > reliable uncertainty handling >
> existing-functionality preservation > minimal code changes. No model was
> retrained, no weights were overwritten, no dataset was downloaded.

---

## 0. Environment constraint that shapes this work

The trained checkpoints in this checkout are **Git LFS pointers**, not weights:

```
models/crop_model_best.pth   -> LFS pointer (real size 75,199,857 bytes)
models/health_model_best.pth -> LFS pointer (real size 420,021,803 bytes)
```

The LFS storage CDN is **not reachable** from this sandbox, so real-weight
inference could not be executed here. Consequences, handled honestly:

- All fixes that do not need weights (segmentation, color evidence, pattern
  detection, KB matching, healthy-consistency, OOD *gate logic*, routes, DB,
  templates) are implemented and **fully tested** on synthetic images and the
  10 real photos shipped in `samples/`.
- Real-model tests **auto-skip** when weights are LFS pointers (22 skipped).
- A calibration script (`scripts/calibrate_crop_ood.py`) is provided to derive
  OOD thresholds from `models/val.csv` **where the real weights + dataset
  exist**. Avocado rejection is **not claimed as validated** — it cannot be,
  without the weights and representative unsupported-species images.

---

## 1. Root cause of each issue

### Issue 1 — Yellow patches reported as “Powdery” / Sooty Mold
Two compounding causes:
1. **Pattern was a static disease-name mapping.** `predicted_pattern` was copied
   verbatim from the matched KB row (`best_match['pattern']`). “Coffee Sooty
   Mold” carries `pattern='powdery'`, so any leaf routed to that row displayed
   “Powdery” regardless of the image. The pattern never looked at pixels.
2. **Color evidence was corrupted upstream.** The green-only leaf mask excluded
   internal yellow lesions (so yellow was under-counted), while shadows and the
   gray concrete background leaked in as `black`/`gray`. The KB color score then
   favored Sooty Mold (`black,gray`) over Rust/Chlorosis (`brown,yellow` /
   `yellow,white`).

### Issue 2 — Avocado classified as Tea
The crop model is a **closed-set binary** (coffee/tea) softmax. The only guard
was a single arbitrary `crop_conf < 60%` check. A binary softmax always sums to
1 and is routinely >90% confident on unsupported species, so avocado was forced
into tea/coffee. There was **no OOD mechanism**. (Preprocessing was verified
consistent with training: `Resize((224,224))` + ImageNet mean/std — not a cause.)

### Issue 3 — Healthy leaves showing Black / Gray
Reproduced on the repo’s own healthy samples (before fix): e.g.
`['black','gray']`, `['black','white','gray']`, up to ~40% “severity”. Causes:
- `get_leaf_mask` fell back to a global gray threshold when green coverage was
  low, admitting the **entire concrete background** as leaf.
- HSV `black` (V≤80) matched shaded dark-green tissue; `gray` (S≤50) matched
  **normal dark, desaturated tea-leaf pigmentation** (measured median S≈43) and
  background; `white` matched **specular highlights** on glossy leaves.
- No health-consistency layer: disease colors/severity were shown even when the
  health model said healthy.

### Additional bugs found & fixed
- `knowledge_base` table was never created by `init_db()` (it only existed in the
  notebook’s gitignored Kaggle DB) → fresh installs crashed on `/predict`.
- `HealthModel` re-downloaded ImageNet weights at startup only to overwrite them
  with the checkpoint (slow, network-dependent).
- No guard for LFS-pointer “weights” (cryptic unpickling error).
- Corrupt uploads crashed `cv2.resize(None)`.
- **Hardcoded Groq API key** in both training notebooks (security).

---

## 2. Exact files modified

Modified:
- `app.py` — load optional OOD calibration at startup.
- `config.py` — OOD/segmentation/pattern constants; calibrated color thresholds.
- `database.py` — create + idempotently seed `knowledge_base` (16 rows, exact
  notebook content); existing data never overwritten.
- `routes/predict_routes.py` — leaf-validity gate, OOD crop gate, evidence-based
  pattern, healthy-consistent output, corrupt-image handling; API keys preserved.
- `services/color_filter.py` — robust segmentation + evidence-based color filter.
- `services/knowledge_base.py` — health-consistent matching; no invented disease.
- `services/llm_service.py` — graceful prompt for inconclusive diagnosis.
- `services/model_loader.py` — `weights=None` (offline, identical after
  `load_state_dict`), LFS-pointer guard, `weights_only=True`.
- `services/preprocessing.py` — raise on undecodable image.
- `templates/dashboard.html` — healthy-consistent badges/labels, pattern label
  map, advisory notes; styling preserved.
- `utils/helpers.py` — additive optional response fields (backward compatible).
- `plant-me.ipynb`, `models/plant-me.ipynb` — **credential redaction only**
  (cell 63 source; outputs/metadata/training cells untouched).

Added:
- `services/pattern_detection.py` — evidence-based pattern detection.
- `services/crop_ood.py` — MSP/energy/kNN OOD scores + gate + calibration loader.
- `scripts/calibrate_crop_ood.py` — data-driven OOD threshold calibration.
- `scripts/smoke_app.py` — documented HTTP smoke-test entry point.
- `tests/` — 7 test modules + fixtures (100 tests total).
- `docs/DIAGNOSIS_FIXES.md` — this report.

**Unchanged:** all `models/*.pth`, `*.npy`, `*.csv`, `*.png`; `requirements*.txt`;
auth routes; login/signup/index templates; model architectures; class mappings.

---

## 3. Technical explanation of each fix

### Color evidence & segmentation (`color_filter.py`)
- **Segmentation:** green-vegetation HSV mask → morphological clean → keep
  significant contours → **fill enclosed holes** so internal yellow/brown/white
  lesions count as leaf area. **Removed** the gray-threshold background fallback.
  Returns `is_leaf`, coverage, green-fraction, blur score.
- **Interior analysis:** erode the boundary rim (`LEAF_INTERIOR_ERODE_PX=3`) so
  edge anti-aliasing/cast shadow isn’t read as disease.
- **Achromatic principle:** `black`/`gray` disease is achromatic; **green-hued
  pixels are excluded** (normal pigmentation + shadows on green tissue). This is
  what stops an entire dark healthy tea leaf from reading as “gray”.
- **Specular suppression:** `white` pixels near blown-out cores (V≥240, S≤30)
  are treated as glossy reflections, not powdery growth.
- **Component filtering:** drop regions smaller than `max(40px, 0.2% leaf)`.
- **Area thresholds:** a color is reported only above a calibrated fraction of
  leaf area (see §6 calibration table).

### Pattern detection (`pattern_detection.py`)
- Derives pattern **only** from measured color ratios + health prediction; the
  function signature has **no disease argument** (structurally prevents
  disease-name mapping). Yellow→`yellowing`, white→`powdery`, black→
  `dark_coating`, brown→`spots`, comparable multiples→`mixed`, unhealthy with no
  evidence→`inconclusive`, healthy→`none`. Each result carries confidence [0,1]
  and a human-readable reason.

### Crop OOD gate (`crop_ood.py`)
- Signals: **MSP**, **energy** (`-logsumexp(logits)`), **feature-space kNN**
  distance to a validation embedding bank. Gate rejects on low MSP, high energy,
  or far kNN. Thresholds come from a calibration JSON; uncalibrated fallback is
  conservative MSP≥0.90 with energy/kNN disabled until calibrated.
- **Design note:** for a 2-class softmax, margin (`2·MSP−1`) and normalized
  entropy are strictly monotone in MSP, so they add no independent signal;
  they’re computed for reporting only (verified by a test).
- A confident-but-far embedding (the avocado case) is rejected by kNN even when
  MSP>0.99 (covered by a unit test with controlled features).

### Healthy consistency (`predict_routes.py`, `knowledge_base.py`, dashboard)
- Healthy → `detected_colors=[]`, `pattern='none'`, `severity='None'`,
  `severity_pct=0`, disease `No Disease Detected`. Strong contradictory color
  evidence surfaces as an **advisory note**, not a silent override.
- KB never cross-matches healthy↔unhealthy; unhealthy with no color evidence
  returns `Condition Not Identified` instead of inventing a disease.

---

## 4. Model weights preserved

`sha256sum -c` against pre-work snapshots: **all OK**.
```
models/crop_model_best.pth:   OK
models/health_model_best.pth: OK
models/crop_class_weights.npy:  OK
models/health_class_weights.npy: OK
```
`git diff` shows **no** `.pth/.npy/.csv` changes. Architectures and class
mappings (`['coffee','tea']`, `['healthy','unhealthy']`) are unchanged.
`weights=None` in `HealthModel` only skips a redundant pretrained download; the
checkpoint overwrites every parameter, so inference is identical (state-dict
key compatibility verified by a test).

---

## 5. Tests executed

Command: `/home/user/.venv/bin/python -m pytest tests/ -q`

Modules: `test_color_filter.py`, `test_pattern_detection.py`,
`test_crop_ood.py`, `test_knowledge_base.py`, `test_app_regression.py`,
`test_model_integration.py` (+ `conftest.py` fixtures).

Coverage: crop gate (confident/ambiguous/OOD-feature/high-energy), health
consistency, patterns (yellow/white/black/brown/healthy/shadow/background),
segmentation & non-leaf rejection, KB matching & seeding, auth
(signup/login/logout/duplicate/mismatch), dashboard rendering, upload
validation (missing/bad-ext/corrupt/non-leaf), prediction API contract, DB
history logging, recommendation fallback, model-loader guards.

## 6. Actual test results

```
78 passed, 22 skipped in ~12s
```
The 22 skips are **exactly** the real-weight integration tests, skipped with the
explicit reason “Trained weights are Git LFS pointers … run `git lfs pull`”.
No failures, no silent skips elsewhere.

**Live HTTP smoke test** (`scripts/smoke_app.py`, mock models because weights are
LFS pointers): index 200 → signup success → login success → dashboard 200 →
healthy-leaf predict returned `crop=coffee, health=healthy, pattern=none,
colors=[], severity=None, disease=No Disease Detected` → non-leaf rejected
(`reason=not_a_leaf`) → prediction logged to DB. Server stopped afterward.

**Color threshold calibration** (measured on the 10 shipped samples after
suppression; the only in-repo validation images):

| color | healthy max | unhealthy max | threshold | rationale |
|-------|-------------|---------------|-----------|-----------|
| brown | 0.0322* | 0.0087 | 0.02 | *real blemish on a nominally healthy leaf |
| black | 0.0000 | 0.0120 | 0.01 | margin above healthy max 0 |
| yellow| 0.0115 | 0.0107 | 0.04 | overlapping; conservative |
| white | 0.0054 | 0.0645 | 0.02 | ~3.7× above healthy max |
| gray  | 0.0101 | 0.0113 | 0.08 | ambiguous; kept high |

## 7. Before / after behavior (real samples)

| sample | before (colors / severity) | after (colors / pattern) |
|--------|----------------------------|--------------------------|
| coffee healthy ×3 | black/gray/white, up to 15.7% | `[]` / none (healthy) |
| tea healthy ×3 | black/gray/brown, up to 40% | no false achromatic; 1 real brown blemish→spots* |
| coffee unhealthy | brown/black/gray, 41.8% | `['black']` / dark_coating |
| tea unhealthy (white patches) ×2 | black/white/gray | `['white']` / powdery |
| tea unhealthy (subtle) | black/gray | `[]` / inconclusive (honest) |
| yellow-patch leaf (synthetic) | could route to powdery | `yellowing` (never powdery) |
| avocado / non-leaf | forced to tea/coffee | non-leaf rejected; OOD gate ready (needs calibration) |

\* The route’s healthy-consistency layer displays healthy leaves as healthy with
no disease indicators; real blemishes become an advisory note.

## 8. Remaining limitations

- **OOD thresholds are uncalibrated in this environment.** Reliable avocado
  rejection requires running `scripts/calibrate_crop_ood.py` with the real
  weights + validation images, ideally measuring rejection on a representative
  unsupported-species folder (`--ood-dir`). Until then the gate uses a
  conservative MSP fallback; **avocado rejection is not validated**.
- **Color thresholds calibrated on only 10 images.** They remove the reported
  false positives and improve clear-lesion sensitivity, but need re-validation
  on a larger held-out set.
- **Dark green-hued disease vs. shadow** is inherently ambiguous in HSV; such
  cases return `inconclusive` rather than a guessed pattern (by design).
- Specular/pigmentation suppression is heuristic; extreme lighting may still
  confuse white/gray evidence.

## 9. Issues requiring additional data / retraining

- **Robust unsupported-crop rejection** ultimately benefits from either
  (a) representative OOD images to calibrate/validate the feature-kNN gate, or
  (b) retraining with an explicit “unknown” class / open-set objective.
  **No retraining was performed.** Proposed plan if approved: collect avocado +
  other species + non-leaf images, run the calibration script, measure ID
  retention vs OOD rejection, and only consider retraining if the gate cannot
  meet target rejection at acceptable ID retention.
- **Fine-grained disease patterns** (e.g. distinguishing sooty mold from other
  black coatings) would need lesion-level labels; current evidence supports
  coarse visual patterns only.

## 10. Unrelated functionality preserved

Verified by regression tests + smoke test: authentication (signup/login/logout,
duplicate/mismatch/wrong-password), dashboard rendering, image upload &
validation, crop/health classification flow, disease identification, severity
estimation, color analysis, knowledge base, treatment recommendations (graceful
fallback without GROQ key), diagnosis history logging, database operations,
existing routes and API response keys, frontend templates/styling. New response
fields are additive; existing clients are unaffected.

## Security remediation

- Removed the hardcoded Groq API key from **both** notebooks (cell 63 source
  only; switched to `os.environ.get('GROQ_API_KEY', '')`). Re-scan confirms
  **0** hardcoded credential literals remain in source or notebooks.
- Application secrets load exclusively via environment variables
  (`config.py`: `os.getenv('GROQ_API_KEY', '')`). No secret values were printed.
- The exposed key should be **revoked/rotated** by the owner since it was
  previously committed to Git history (history was not rewritten, per rules).
