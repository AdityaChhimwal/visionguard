# Engineering Log

## Initial audit — 2026-10-24

### What was present

- A Git repository on `main` with two setup/EDA commits.
- A complete local MVTec AD directory containing all 15 categories.
- Exploratory notebooks for EDA and VGG16 feature extraction.
- A starter validator and a minimal environment test.
- TensorFlow 2.21, FastAPI, scikit-learn, Pillow, and pytest in the local
  virtual environment.

### What needed engineering work

- The validator was notebook-style code with repeated machine paths and broad
  exception handling rather than a reusable module.
- VGG16 embeddings, nearest-neighbor scoring, leave-one-out thresholding, and
  evaluation existed only as notebook cells.
- There were no saved inference artifacts, API, test suite, Dockerfile,
  reproducible configuration, or usable README.
- The dependency file was an environment dump rather than portable project
  dependencies.

### Decisions

- Preserve existing notebooks, including the user-edited Day 2 notebook.
- Turn the valid VGG16 nearest-neighbor idea into a small image-level baseline
  rather than prematurely implementing PatchCore.
- Use VGG16 block-5 global-average features (512 values) instead of flattening
  25,088 spatial activations.
- Calibrate with a 95th-percentile leave-one-out normal score, retaining the
  full normal reference bank for inference.
- Keep test labels and masks exclusively in evaluation code.

## Verification record

### 2026-10-24

- `python -m training.validate_dataset --dataset-root mvtec_anomaly_detection
  --category bottle` passed: 209 normal training images, 83 test images, no
  unreadable images.
- `python -m training.train --config configs/default.json` completed:
  - artifact: `artifacts/bottle_v1` (209 x 512 reference bank)
  - threshold: `0.121975` from 209 leave-one-out normal scores
  - held-out image-level AUROC: `0.984921`; AP: `0.994984`
  - threshold confusion matrix: TN=18, FP=2, FN=1, TP=62
  - detailed error analysis: `outputs/bottle_image_level_evaluation.json`
- A fresh Python process loaded the saved artifact successfully.
- A real Uvicorn process loaded the artifact; `/health` returned 200/healthy, a
  real MVTec normal-image upload returned 200, and a text upload returned 422.
- `python -m pytest --basetemp=D:\good_projects\visionguard\pytest_temp`:
  20 passed. The sandbox required an alternate test temp directory; one
  upstream FastAPI TestClient deprecation warning remained before its final
  status-constant cleanup.
- Docker client is installed but Docker Desktop's Linux daemon was unavailable;
  image build/container execution was not attempted.
