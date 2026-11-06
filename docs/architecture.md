# Architecture and model decisions

VisionGuard is a category-aware inspection service for the ten object and five texture categories in MVTec AD. Each category has its own PatchCore graph, normal reference patches, decision threshold and display scale. The browser gets the available categories from `GET /api/categories` and sends the selected category with the image to `POST /api/predict`. Category selection is part of the inspection setup. The service cannot infer which category an uploaded image belongs to. [MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

## Request and deployment paths

A semaphore limits admitted uploads, including slow reads. The request body is capped while streaming, and a worker thread handles decoding, ONNX inference and preview encoding. The image decoder determines the file type from its bytes. The service processes uploaded images in memory and returns the score, cutoff and aligned previews. The browser requests JPEG previews to reduce the response size. API clients get PNG by default and can opt into JPEG with `preview=jpeg`. Encoding happens after scoring and leaves the score and cutoff intact.

At startup, the registry reads the category allowlist and verifies each artifact in turn. It checks the model and split against their stored digests, checks the model's input/output contract and runs a warm-up pass for finite outputs. It releases each checked session and retains the first available category as the warm default. At request time it keeps a bounded least recently used cache of sessions. `VG_MODEL_CACHE_SIZE=1` and `VG_INFERENCE_SLOTS=1` bound the resident cache and admitted inference work on the Free plan. Switching to a category outside the cache takes a model load. Sample paths are checked at startup and images are served from disk on demand.

`/health` reports that the process is running. `/ready` lists the available category models and returns HTTP 503 if none are available. With `VG_REQUIRE_ALL_MODELS=1`, any configured model failing verification also makes the deployment unready. This lets local development run with a subset while the release requires all 15 models.

Training and evaluation run separately from serving. The release groups model graphs, split manifests, calibration metadata, two samples per category and notices in one checksum-pinned archive. Provisioning validates the whole archive, checks its member list and model digests, then publishes the verified files from a staging directory. The Docker build installs the bundle into the image, so the server starts with local artifacts. Use a trusted source for the release and checksum. [Deployment instructions](deployment.md)

## What PatchCore does

Anomalib uses timm to extract ResNet-18 layer2 (128 channels) and layer3 (256 channels). At 160 × 160 their grids are 20 × 20 and 10 × 10. Anomalib averages each 3 × 3 neighborhood, interpolates the coarser grid and concatenates the features into 400 vectors of length 384 per image. Its projected greedy k-center sampler selects a coreset from fitting patches. At inference it uses Euclidean distance to the nearest coreset row for each patch. With one neighbor, the image score is the maximum of those distances. The patch distances are enlarged and smoothed for the model map.

The shared CPU profile uses ResNet-18, a 160-pixel full-frame resize, a 1% coreset and one nearest neighbor. This is a compact configuration of PatchCore. The same EXIF transpose, RGB conversion, bilinear resize and ImageNet normalization run during fit, calibration, evaluation and serving. The model map is resized back to the EXIF-oriented frame.

## Preventing test leakage

The project calls the PatchCore model directly and defines its own split, export checks, calibration and evaluation. This avoids the Anomalib dataset module's default validation behavior, which uses the test split. Anomalib supplies the patch features, coreset sampling, nearest-neighbor distances and anomaly maps. [Versioned Anomalib dataset code](https://github.com/open-edge-platform/anomalib/blob/9fb1337c0ac866797f4eedcaa0972d0f24a84f19/src/anomalib/data/datamodules/image/mvtecad.py)

Within each category, a seeded split reserves 20% of `train/good` for calibration. Fit uses the remaining good images. Exact duplicate hashes are checked within training, across training and test, and within test. Calibration uses the exported graph on the held-out normal images and freezes the higher empirical 95th percentile as the image cutoff. The decision is ANOMALY when the score strictly exceeds that category's threshold.

Evaluation scores the entire original test split with the frozen artifact. Image AUROC measures how the raw distances rank images in one category. Pixel AUROC compares the original-resolution masks with aligned raw anomaly maps. The release matrix uses 65,536 fixed bins of the monotonic transform `score / (score + heatmap_high)` to approximate pixel AUROC with bounded memory. This pooling weights large images and abundant background pixels more heavily. The two demo samples are chosen after evaluation from correctly classified normal and defective examples. Use the full test report to describe performance. [Verification report](verification.md)

The display cutoff is the higher empirical 95th percentile of map peaks on the calibration normals. The display range stays fixed across images in the same category, with values above the cutoff clipped. Warmer regions indicate larger patch distances under that category's model. Lighting, backgrounds, orientation and fine details lost during resizing can all affect the result.
