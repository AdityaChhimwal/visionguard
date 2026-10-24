# VisionGuard

**Industrial visual anomaly detection — reproducible normal-only baseline.**

VisionGuard learns from defect-free product images and flags new images that
deviate from normal visual features. It is a research/portfolio prototype, not
a certified inspection product or a production-ready quality-control system.

## Why anomaly detection?

In manufacturing, examples of every future defect are seldom available. A
supervised defect classifier consequently has the wrong training assumption for
this baseline. VisionGuard fits only `train/good` images and assigns a distance
score to new images. The score is **not a defect probability**.

## Methodology

```text
normal MVTec training images
  -> RGB resize (224 x 224) + VGG16 ImageNet preprocessing
  -> frozen VGG16 block-5 features + global average pooling (512 values)
  -> L2-normalized normal reference bank

new image -> same preprocessing/features -> nearest-reference Euclidean distance
          -> score >= normal-only threshold ? anomaly : normal
```

The decision threshold is the 95th percentile of **leave-one-out**
nearest-neighbor scores from normal training images. Each calibration image is
excluded from its own reference comparison, avoiding a zero-distance leak. The
full normal bank remains available at inference. Test images, labels, and masks
are never used for fitting or threshold selection.

This is an image-level baseline. It does not create spatial anomaly maps or
claim pixel-level localization.

## Dataset and licensing

This project expects [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
an industrial anomaly-detection benchmark with 15 categories, normal training
images, normal/defective test images, and pixel masks for anomalous samples.
MVTec releases it under **CC BY-NC-SA 4.0**; in particular, it may not be used
commercially. The dataset and its annotations are ignored by Git and excluded
from the Docker image.

```text
mvtec_anomaly_detection/
  bottle/
    train/good/*.png                 # model fitting only
    test/good/*.png                  # evaluation only
    test/<defect_type>/*.png         # evaluation only
    ground_truth/<defect_type>/*_mask.png  # never an inference input
```

## Setup and validation

Use a TensorFlow-supported Python version (the Docker image uses Python 3.11).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m training.validate_dataset --dataset-root mvtec_anomaly_detection --category bottle
```

Validation checks category/split structure, image presence and readability, and
defect ground-truth directories. It fails loudly on malformed data.

The first training run may download public VGG16 ImageNet weights into
TensorFlow's cache. Inference reconstructs that documented network from cached
weights; it does not retrain or download it per request.

## Train and evaluate

The tracked configuration uses the `bottle` category. Copy/edit it to change
paths or category; application source never hard-codes a personal path.

```powershell
$env:PYTHONPATH = "src"
python -m training.train --config configs/default.json
```

The command writes ignored local state:

- `artifacts/bottle_v1/reference_embeddings.npy` — normal reference bank
- `artifacts/bottle_v1/metadata.json` — threshold, feature/preprocessing,
  threshold policy, configuration, runtime metadata
- `outputs/bottle_image_level_evaluation.json` — AUROC, average precision,
  threshold metrics, group summaries, false positives and false negatives

The evaluation report uses held-out MVTec test labels only after training. It is
an image-level portfolio evaluation, not an official leaderboard claim. Add
`--skip-evaluation` to produce an artifact only.

### Observed `bottle` baseline run (2026-10-24)

The configured full run validated 209 normal training images and 83 held-out
test images, then persisted a 209 x 512 reference bank. Its normal-only
threshold was **0.121975**. Held-out image-level metrics were **AUROC 0.984921**
and **average precision 0.994984**. At that preselected threshold, the confusion
matrix was TN=18, FP=2, FN=1, TP=62 (precision 0.968750, recall 0.984127,
F1 0.976378). The error analysis names the two high-scoring normal images and
the missed contamination image in `outputs/bottle_image_level_evaluation.json`.
These figures are one `bottle` baseline run, not a multi-category or official
benchmark claim; no test result was used to select the threshold.

## API

The API loads its artifact once on startup. `GET /health` reports `healthy` or
`unavailable`; `POST /predict` accepts a multipart upload, validates bytes with
Pillow rather than trusting filename/MIME type, and applies training-equivalent
preprocessing.

```powershell
$env:PYTHONPATH = "src"
$env:VISIONGUARD_ARTIFACT_DIR = "artifacts/bottle_v1"
uvicorn api.main:app --host 127.0.0.1 --port 8000
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict -Form @{ file = Get-Item "mvtec_anomaly_detection/bottle/test/good/000.png" }
```

OpenAPI documentation: `http://127.0.0.1:8000/docs`

```json
{
  "prediction": "normal",
  "anomaly_score": 0.0,
  "threshold": 0.0,
  "model_version": "visionguard-v0.1.0",
  "category": "bottle"
}
```

`prediction` is `anomaly` exactly when score is at least threshold. Bad image
bytes receive 422; a >10 MiB upload receives 413; unavailable artifacts return
503. The service never produces a fake fallback prediction.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests use synthetic images, temporary MVTec-like folders, controlled embeddings,
artifact round-trips, and a test-double predictor. They need no full dataset,
GPU, or ImageNet download.

## Docker

An image is a packaged runtime; a container is a running instance; Uvicorn in
the container exposes port 8000. Build after training, then mount artifacts:

```powershell
docker build -t visionguard:local .
docker run --rm -p 8000:8000 -v "${PWD}\artifacts:/models:ro" visionguard:local
Invoke-RestMethod http://127.0.0.1:8000/health
```

The image deliberately excludes the complete dataset and artifact. An omitted
artifact mount yields an unhealthy service and 503 inference responses. Docker
downloads/caches VGG16 weights during the image build, not at request time.
Docker Desktop was not running during the 2026-10-24 verification, so the build
and container commands above remain unverified in this checkout.

## Layout

```text
src/visionguard/   reusable data, feature, detector, artifact, evaluation, inference
training/          validation and training CLIs
api/               FastAPI service
configs/           portable JSON configuration
tests/             lightweight automated tests
artifacts/         ignored local model state
outputs/           ignored evaluation/error-analysis output
```

## Limitations

- This is a frozen ImageNet-feature nearest-neighbor baseline, not PatchCore or
  a state-of-the-art claim.
- A percentile threshold controls a normal-training tail; it is not calibrated
  defect probability and must be revisited for a real inspection line.
- VGG16 features can miss subtle domain-specific defects.
- No localization, cloud deployment, GPU latency claim, or certification is made.
- MVTec AD's non-commercial license restricts appropriate use.

## Interview-ready decisions

1. Preserve the normal-only benchmark split; do not turn anomaly detection into
   disguised supervised classification.
2. Global pooling reduces 25,088 VGG16 spatial activations to a manageable
   512-D embedding without changing the pretrained backbone.
3. Leave-one-out calibration prevents self-neighbor leakage.
4. Explicit persisted artifacts mean inference never retrains.
5. Byte validation, single model load, intentional HTTP failures, and behavior
   tests make the demo an engineered system rather than a notebook endpoint.
