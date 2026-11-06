# Defend VisionGuard in the interview

## Two-minute explanation

“I built VisionGuard as a portfolio demo for visual inspection when defect examples are rare. It uses PatchCore from Anomalib. I fit a small reference memory on good Bottle images from MVTec AD, then set the decision threshold on different good images. At inspection time, the system scores unfamiliar regions and shows them as an overlay. A score is a distance; NORMAL just means the cutoff was not exceeded.

“The model ran on the full Bottle test category. It caught 63 of 63 defects and incorrectly flagged four of 20 good bottles at the cutoff I froze before evaluation. Image AUROC was 0.9984; pixel AUROC was 0.9718. That's evidence for the benchmark conditions, not proof for a production line. I used PyTorch and Anomalib for preparation, exported the graph for CPU serving, and built a FastAPI service with one web page. The local runner tests included real browser uploads and Docker inference without a network. I used Codex to help rebuild and test the project; the algorithm and encoder are credited third-party work. I can trace and explain the system and the decisions it makes.”

If public hosting is still pending, say that explicitly. Do not imply the benchmark is your original research or say you wrote code you cannot explain.

## Five-minute walkthrough

| Minute | What to explain |
| --- | --- |
| 0–1 | Show the user task: choose Bottle, select a real image, compare original and overlay. Explain that capture conditions matter. |
| 1–2 | Follow `scripts/dataset.py` and `scripts/prepare.py`: 167 normal fit images, 42 separate calibration normals, test files untouched until evaluation. |
| 2–3 | Follow `visionguard/ml/patchcore.py`: pretrained ResNet-18, two scales, pooled patches, greedy coreset, nearest-neighbor distances. Show the metadata counts. |
| 3–4 | Trace `visionguard/web/app.js` → `POST /api/predict` → streaming limits → EXIF/RGB/resize → reused ONNX graph → cutoff and overlay. |
| 4–5 | Discuss the confusion matrix, four false positives, fixed display scale, 220 ms warm runner median and roughly 188 MiB sampled peak. Distinguish these from public-host performance. |

### Draw the algorithm from memory

1. Pixels become an oriented RGB image. Resize the whole frame to 160 × 160, then normalize channels using the ImageNet mean and standard deviation.
2. A fixed, pretrained ResNet-18 produces feature grids from layers 2 and 3. A 3 × 3 average pools neighboring features. Resize the coarse grid and concatenate it with the fine grid; each of the 400 spatial locations has 384 feature values.
3. Fit collects patches only from normal images. It uses a projected greedy k-center coreset to keep 668 representative vectors out of 66,800 instead of remembering every patch.
4. Inference measures each patch's nearest Euclidean distance to the memory bank. The maximum distance becomes the image score for this one-neighbor variant. The spatial grid of distances is enlarged and smoothed; the application aligns it to the input preview.
5. Compare the raw image score with 8.511868. Overlay color uses a different range derived from calibration map peaks. It is possible to see warm pixels on a NORMAL image.

## Twenty likely questions

1. **Why PatchCore?** It is a useful fit for inspection with many good examples and few known defects. It gives both a product score and a localization signal.
2. **Why fit only good bottles?** The bank is a reference for acceptable appearance. Including defects would make them easier to match, possibly hiding them.
3. **Why a pretrained CNN?** ImageNet features already encode edges, shapes and textures. We can reuse intermediate features without training a network on 167 images.
4. **Why intermediate layers?** The higher-resolution grid preserves local detail; the coarser one adds context. Neither is the classifier's final class output.
5. **What is the memory bank?** A matrix of representative normal patch vectors. A row is a 384-number description of one region, not a stored input image.
6. **Why a coreset?** It favors coverage of the feature space while reducing storage and comparisons. Too small a coreset can miss legitimate variation and increase false positives.
7. **Where is the anomaly score computed?** Anomalib's one-neighbor PatchCore takes the maximum nearest-normal-patch distance. The exported graph still computes that score.
8. **How did you choose the threshold?** I used the higher empirical 95th percentile of the 42 held-out normal scores, 8.511868, and applied strict greater-than. This produced two calibration exceedances.
9. **How do you prevent leakage?** No Anomalib data module, trainer or postprocessor runs. Fit and calibration use separate seeded subsets of train/good. Evaluation loads the frozen graph and cutoff; it checks exact duplicate hashes.
10. **Why isn't 21.01 a confidence score?** It is an uncalibrated feature distance. A probability would require a separate, validated mapping from distance to likelihood.
11. **Why was test FPR 20% if the target was 5%?** Two of 42 calibration exceedances were an empirical operating choice. The test normal set is small and can differ. Four of 20 normals crossed the cutoff.
12. **What is AUROC versus precision and recall?** AUROC summarizes how well raw scores rank defects above normal images across cutoffs. Precision and recall describe decisions at one cutoff and precision also depends on defect prevalence.
13. **What does pixel AUROC mean?** It ranks mask pixels against non-mask pixels using the raw aligned map. At 0.9718 the ranking was good on this category; the figure does not certify segmentation boundaries.
14. **What changes did you make from the paper?** Smaller ResNet-18, 160-pixel resize, a 1% bank, and a one-neighbor maximum score without reweighting. I did not attempt to reproduce the paper's benchmark.
15. **How do you know the serving graph matches training?** All calibration samples passed raw score/map parity with PyTorch within 0.001 absolute and 0.0001 relative tolerance. A separate integration test reloads and compares the export.
16. **Why FastAPI and plain JavaScript?** The use case has one upload form and one result. Serving it and the API from one process keeps deployment and debugging simple.
17. **How do you keep inference resource use predictable?** One warmed registry per worker, one admitted request by default, streaming and pixel caps, and bounded PNG previews. The service rejects another request with 429 while the slot is occupied.
18. **Where can it fail?** Glare, lighting drift, different backgrounds, unfamiliar bottles, and tiny defects after resizing. A hot area may reflect the setup rather than a defect.
19. **Which parts are yours to explain?** Data split and metadata, preprocessing, calibration, evaluation, export/parity, serving, upload handling, overlay and deployment. Anomalib provides PatchCore; timm provides the pretrained CNN.
20. **What would you improve next?** Collect independent normal and defect images under target conditions; measure score drift, false-alarm cost and size/latency tradeoffs. Add a category only after its own end-to-end evaluation.

## Five-minute demo with a local fallback

1. Before the interview, provision the release or prepare it using the [README](../README.md). Verify `/ready` and run both fixed files through the local backend. The exact local commands also work when external hosting sleeps.
2. In the browser, use Bottle and load the normal example. Click Analyze. Describe the distance of about 8.163 versus the threshold of 8.512.
3. Reset. Load the defective example. Explain its score of about 21.015 and point to the hot area without calling the overlay a segmentation mask.
4. Open the [verification report](verification.md). Explain why the two demos do not represent perfect accuracy: the 83-image test set includes four false alarms.
5. If the connection fails, use `http://127.0.0.1:8000` or the offline Docker command. If no model is loaded, show readiness and the recorded results and say the live fallback is unavailable.

## Three resume bullets

- Rebuilt an industrial inspection demo around Anomalib PatchCore and a CPU FastAPI/ONNX service that returns aligned defect overlays.
- Implemented held-out normal calibration and evaluated all 83 MVTec AD Bottle test images: image AUROC 0.9984, pixel AUROC 0.9718, F1 0.9692 at the frozen threshold.
- Verified desktop/mobile inference and Linux Docker startup offline; measured 220 ms median warm loopback latency and about 188 MiB sampled process RSS on a GitHub Linux runner.

## Seven days of practice

| Day | Do this yourself |
| --- | --- |
| 1 | Start the server and run both sample inferences in the browser and with curl. Explain score versus threshold without notes. |
| 2 | Trace one upload from the Analyze click through `read_upload`, `infer`, and `Graph.run`. Trigger a malformed PNG and explain the 422. |
| 3 | Read `metadata.json` and `split.json`. Open the ONNX model with the preparation environment using `onnx.load` and inspect initializer shapes. Locate the bank or its transposed matrix with 668 and 384 dimensions. |
| 4 | Explain a heatmap in terms of local patch distances and the fixed display cutoff. Compare a sample with an unusually lit bottle. |
| 5 | Copy the config, change the coreset ratio to 0.02 and use a new version and output path. Refit and rerun evaluation without adjusting parameters based on the test results. Compare size and speed. |
| 6 | Use `outputs/bottle-evaluation.json` to locate one false positive. Inspect it and propose two hypotheses; distinguish hypotheses from proven causes. |
| 7 | Rehearse the two-minute explanation, five-minute architecture, and live demo. Practice saying what Codex helped with and which results you verified. |
