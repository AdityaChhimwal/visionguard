VisionGuard
Industrial Visual Anomaly Detection

Industrial visual anomaly detection — reproducible normal-only baseline.

VisionGuard learns from defect-free product images and flags new images that deviate from normal visual features. It is a research/portfolio prototype, not a certified inspection product or a production-ready quality-control system.

Why anomaly detection?

In manufacturing, examples of every future defect are seldom available. A supervised defect classifier consequently has the wrong training assumption for this baseline. VisionGuard fits only train/good images and assigns a distance score to new images. The score is not a defect probability.

Methodology
text
normal MVTec training images
  -> RGB resize (224 x 224) + VGG16 ImageNet preprocessing
  -> frozen VGG16 block-5 features + global average pooling (512 values)
  -> L2-normalized normal reference bank

new image -> same preprocessing/features -> nearest-reference Euclidean distance
          -> score >= normal-only threshold ? anomaly : normal


The decision threshold is the 95th percentile of leave-one-out nearest-neighbor scores from normal training images. Each calibration image is excluded from its own reference comparison, avoiding a zero-distance leak. The full normal bank remains available at inference. Test images, labels, and masks are never used for fitting or threshold selection.

This is an image-level baseline. It does not create spatial anomaly maps or claim pixel-level localization.

Dataset and licensing

This project expects MVTec AD, an industrial anomaly-detection benchmark with 15 categories, normal training images, normal/defective test images, and pixel masks for anomalous samples.

The dataset and its annotations must be used according to the official MVTec AD license terms. The dataset is ignored by Git and excluded from the Docker image.

text
mvtec_anomaly_detection/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*_mask.png


Training uses defect-free training images only. Test images and ground-truth masks are reserved for evaluation and are never model-fitting inputs.

Setup and validation

Use a TensorFlow-supported Python version. The Docker image uses Python 3.11.

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m training.validate_dataset --dataset-root mvtec_anomaly_detection --category bottle


Validation checks category/split structure, image presence and readability, and defect ground-truth directories. It fails loudly on malformed data.

The first training run may download public VGG16 ImageNet weights into TensorFlow's cache. Inference reconstructs that documented network from cached weights; it does not retrain or download it per request.

Train and evaluate

The tracked configuration uses the bottle category. Copy or edit it to change paths or category; application source does not hard-code a personal path.

powershell
$env:PYTHONPATH = "src"
python -m training.train --config configs/default.json


The command writes ignored local state:

artifacts/bottle_v1/reference_embeddings.npy — normal reference bank
artifacts/bottle_v1/metadata.json — threshold, feature/preprocessing, threshold policy, configuration, and runtime metadata
outputs/bottle_image_level_evaluation.json — AUROC, average precision, threshold metrics, group summaries, false positives, and false negatives

The evaluation report uses held-out MVTec test labels only after training. It is an image-level portfolio evaluation, not an official leaderboard claim. Add --skip-evaluation to produce an artifact only.

Observed bottle baseline run (2026-10-24)

The configured full run validated 209 normal training images and 83 held-out test images, then persisted a 209 × 512 reference bank. Its normal-only threshold was 0.121975.

Held-out image-level metrics:

AUROC: 0.984921
Average precision: 0.994984
True negatives: 18
False positives: 2
False negatives: 1
True positives: 62
Precision: 0.968750
Recall: 0.984127
F1 score: 0.976378

The error analysis identifies the two high-scoring normal images and the missed contamination image in outputs/bottle_image_level_evaluation.json.

These figures describe one bottle baseline run, not a multi-category or official benchmark result. No test result was used to select the threshold.

API

The API loads its artifact once on startup. GET /health reports healthy or unavailable; POST /predict accepts a multipart upload, validates image bytes with Pillow rather than trusting the filename or MIME type, and applies training-equivalent preprocessing.

powershell
$env:PYTHONPATH = "src"
$env:VISIONGUARD_ARTIFACT_DIR = "artifacts/bottle_v1"
uvicorn api.main:app --host 127.0.0.1 --port 8000
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict -Form @{ file = Get-Item "mvtec_anomaly_detection/bottle/test/good/000.png" }


OpenAPI documentation: http://127.0.0.1:8000/docs

Example response schema:

json
{
  "prediction": "normal",
  "anomaly_score": 0.0,
  "threshold": 0.0,
  "model_version": "visionguard-v0.1.0",
  "category": "bottle"
}


The values are illustrative, not a measured prediction. prediction is anomaly exactly when the score is at least the threshold. Bad image bytes receive HTTP 422; uploads larger than 10 MiB receive HTTP 413; unavailable artifacts return HTTP 503. The service never produces a fake fallback prediction.

Tests
powershell
.\.venv\Scripts\python.exe -m pytest


Tests use synthetic images, temporary MVTec-like folders, controlled embeddings, artifact round-trips, and a test-double predictor. They need no full dataset, GPU, or ImageNet download.

Docker

An image is a packaged runtime; a container is a running instance; Uvicorn in the container exposes port 8000. Build after training, then mount artifacts:

powershell
docker build -t visionguard:local .
docker run --rm -p 8000:8000 -v "${PWD}\artifacts:/models:ro" visionguard:local
Invoke-RestMethod http://127.0.0.1:8000/health


The image deliberately excludes the complete dataset and artifact. An omitted artifact mount yields an unhealthy service and HTTP 503 inference responses. Docker downloads/caches VGG16 weights during the image build, not at request time.

Docker Desktop was not running during the 2026-10-24 verification, so the build and container commands remain unverified in that checkout.

Layout
text
src/visionguard/   reusable data, feature, detector, artifact, evaluation, inference
training/          validation and training CLIs
api/               FastAPI service
configs/           portable JSON configuration
tests/             lightweight automated tests
artifacts/         ignored local model state
outputs/            ignored evaluation/error-analysis output

Limitations
This is a frozen ImageNet-feature nearest-neighbor baseline, not PatchCore or a state-of-the-art claim.
A percentile threshold controls a normal-training tail; it is not a calibrated defect probability and must be revisited for a real inspection line.
VGG16 features can miss subtle domain-specific defects.
No localization, cloud deployment, GPU latency claim, or certification is made.
MVTec AD's license terms restrict appropriate use.
Interview-ready decisions
Preserve the normal-only benchmark split; do not turn anomaly detection into disguised supervised classification.
Global pooling reduces 25,088 VGG16 spatial activations to a manageable 512-dimensional embedding without changing the pretrained backbone.
Leave-one-out calibration prevents self-neighbor leakage.
Explicit persisted artifacts mean inference never retrains.
Byte validation, single model load, intentional HTTP failures, and behavior tests make the demo an engineered system rather than a notebook endpoint.