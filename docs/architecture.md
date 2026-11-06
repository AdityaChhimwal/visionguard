# Architecture and model decisions

The service receives a binary image body with a configured category ID. A per-process semaphore limits admitted uploads, including slow reads. The request is capped while streaming and decoded in a worker thread, so other health and discovery calls remain responsive. The file type comes from the decoder; the service does not need the filename. The server never writes uploads to disk. Inference and PNG encoding run within the same admission slot.

The startup registry reads category configuration, then loads the local ONNX graph and split only if metadata and digests match. A warm-up pass verifies the input/output contract and finite score/map before the category is advertised. The manifest is a corruption check, not a signature; provision the bundle from a trusted, versioned source. Do not let visitors replace any model or metadata file.

## What PatchCore does

Anomalib uses timm to extract ResNet-18 layer2 (128 channels) and layer3 (256 channels). At 160 × 160 their grids are 20 × 20 and 10 × 10. Anomalib averages each 3 × 3 neighborhood, interpolates the coarser grid and concatenates the features into 400 vectors of length 384 per image. Its projected greedy k-center sampler selects a coreset from fitting patches. At inference it uses Euclidean distance to the nearest coreset row for each patch. With one neighbor, the image score is the maximum of those distances. The patch distances are enlarged and smoothed for the model map.

The paper's benchmark configuration is not reproduced. This release uses ResNet-18, 160-pixel full-frame resize, one neighbor without the weighted score, and a 1% coreset to keep CPU inference small. The calibration threshold is an application policy. The same EXIF transpose, RGB conversion, bilinear square resize and ImageNet normalization run during fit, calibration, evaluation, and serving. No crop discards original regions. The model map is resized back to the EXIF-oriented frame.

## Preventing test leakage

The project bypasses Anomalib's Lightning trainer, data module, and postprocessor. The [versioned Anomalib dataset code](https://github.com/open-edge-platform/anomalib/blob/9fb1337c0ac866797f4eedcaa0972d0f24a84f19/src/anomalib/data/datamodules/image/mvtecad.py) defaults to validation from the test split; VisionGuard never invokes it. Anomalib provides embeddings, sampling, nearest-neighbor scores and maps. VisionGuard defines the split, verifies weights, exports and checks parity, calibrates the cutoff, packages the artifact, evaluates test images, and serves results.

All training files are hashed. Exact duplicate content is rejected before splitting and again across training/test and within test. Fit uses only `train/good` and the frozen seed. Calibration uses the held-out normal files and the exported graph. Evaluation has no threshold-fitting code. The test set is used once to measure the frozen release, with label 1 for any defect and image AUROC on raw scores. Pixel AUROC pools original-size binary masks and raw aligned maps. This gives large images and many background pixels more influence.

The map color cutoff comes from the 95th percentile of the **map peak per calibration image**. One shared zero-to-cutoff range applies across all images, with clipping above the cutoff. The overlay is separate from the classifier. A warm region is relatively distant under the model and can include background, lighting or orientation changes. It does not prove a defect.
