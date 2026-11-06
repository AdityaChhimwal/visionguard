# Bottle release verification

The [successful real-model workflow](https://github.com/AdityaChhimwal/visionguard/actions/runs/42185022406) ran on a GitHub-hosted Linux x86_64 runner with 4 logical CPUs. Fit and evaluation used two CPU threads. Input data was the pinned MVTec AD Bottle mirror documented in [NOTICE](../NOTICE.md).

## Split, model and metrics

- Seed 42: 167 fitting normals, 42 held-out calibration normals, and all 83 test images (20 normal, 63 defective across the category's defect folders).
- Anomalib 2.7.0, PyTorch 2.10.0 CPU, ImageNet-pretrained timm ResNet-18 A1, layers 2 and 3, full-frame 160 × 160 input, one neighbor. The fit produced 66,800 patches of 384 features; the 1% coreset kept 668.
- Higher empirical 95th percentile threshold: **8.511868**. 2/42 calibration normals exceeded it (4.76%); 4/20 test normals exceeded it (20%). The test set never sets the threshold. The model range for display is 0 to 7.776685; it is frozen separately.
- All 42 calibration images passed Torch/ONNX comparison at absolute tolerance 0.001 and relative tolerance 0.0001. Largest observed absolute difference was 0.0000773 for scores and 0.000147 for map pixels.
- Graph SHA-256: `f1e49d1b9ae9aac29ac12e6c3f1417302541f2009aec5fda42225e30de7aa3c1`.

| Metric | Bottle result |
| --- | ---: |
| Image AUROC | 0.9984 |
| Precision at frozen threshold | 0.9403 |
| Recall at frozen threshold | 1.0000 |
| F1 at frozen threshold | 0.9692 |
| Pixel AUROC (67,230,000 pooled original-resolution pixels) | 0.9718 |

| Actual class | Predicted NORMAL | Predicted ANOMALY |
| --- | ---: | ---: |
| Normal | 16 | 4 |
| Defective | 0 | 63 |

These numbers describe one category from one dataset. High ranking quality and perfect observed defect recall do not eliminate false alarms or prove transfer to new production conditions. Pixel AUROC measures ranking on aligned pixels; it does not measure boundary accuracy.

```text
uv run --no-sync python -m scripts.prepare --config configs/bottle.toml --dataset data/mvtec/bottle --weights data/resnet18.a1_in1k.safetensors --output artifacts/bottle-pc-v1
uv run --no-sync python -m scripts.evaluate --dataset data/mvtec/bottle --artifact artifacts/bottle-pc-v1 --category bottle --version bottle-pc-v1 --output outputs/bottle-evaluation.json
uv run --no-sync python -m scripts.benchmark --dataset data/mvtec/bottle
uv run --no-sync python -m scripts.verify_browser --dataset data/mvtec/bottle
```

## Real HTTP, browser and container checks

- Bundle: 13,233,088 bytes (12.62 MiB). Fit/calibration took 26.8 s; evaluation including pixel AUROC took 23.8 s.
- One Uvicorn worker, two ONNX threads, 16 sequential warm HTTP requests on the runner: **220.30 ms median**, **225.85 ms p95**. These include request reading, decode, inference, render, and JSON on loopback. The process was ready after **0.712 s** and reached **187.57 MiB** sampled RSS (10 ms cadence). Two 900 × 900 samples were used, so this is not a worst-case bound. Public host latency and cold starts remain unmeasured.
- Fixed examples: `test/good/000.png` scored **8.162982**, NORMAL. `test/broken_large/000.png` scored **21.014507**, ANOMALY. Both predictions and overlays came from the actual model.
- Desktop Chromium uploaded the normal PNG and mobile Chromium loaded the defective example. Checked result images, layout overflow, processing status, reset/focus, and a malformed drag-and-drop upload on desktop. The browser uses the same HTTP path for examples and uploads.
- The Linux container built successfully, started with `--network none`, passed readiness, and ran both examples. This verifies the artifact is sufficient for serving without weight downloads.
- Synthetic PatchCore integration: 1 passed (15 unrelated tests deselected in that run). Ordinary CI runs lightweight tests without the preparation group. The Windows workstation's full dependency downloads timed out; a local subset passed using already available Python packages. GitHub runners provide fresh-install evidence.

See the workflow for the [exact run steps, evidence and artifacts](https://github.com/AdityaChhimwal/visionguard/actions/runs/42185022406).

The [final rebuild checks](https://github.com/AdityaChhimwal/visionguard/actions/runs/42188042833) passed on Windows and Linux (17 passed, 1 PatchCore integration test deselected in each), and the [real-model rerun](https://github.com/AdityaChhimwal/visionguard/actions/runs/42188043042) passed with the locked dependencies. The [bundle publish job](https://github.com/AdityaChhimwal/visionguard/actions/runs/42188497384) verified the public ZIP at image build time; the image started and ran both examples with no network and no mounted assets.
