# Defend VisionGuard in the interview

## Two-minute explanation

“VisionGuard is a visual inspection demo for the ten object and five texture categories in MVTec AD. I used Anomalib's PatchCore implementation and a pretrained ResNet-18. The preparation pipeline makes a reference memory from good images within each category. It uses a separate set of good images to set each category's cutoff. The user selects the category, uploads an image, and gets a distance score, a decision and an overlay of regions with larger patch distances.

“I export the graphs from PyTorch and check that ONNX Runtime reproduces the scores and maps on the calibration inputs. One FastAPI service hosts the browser UI and CPU inference. It keeps a small cache of models so switching categories does not require all 15 sessions to be in memory. I used Codex to help implement and verify the system. The algorithm, encoder and data are credited third-party work.”

Across the 1,725 test images, the macro image AUROC was 0.9344 and the macro F1 was 0.8819 at the frozen thresholds. Grid recall was 0.4035, so there is a concrete example of the trade-off between calibration on normal images and missed defects. [Full benchmark](mvtec-ad-results.md)

## Five-minute walkthrough

| Minute | What to explain |
| --- | --- |
| 0–1 | Show the Objects and Textures groups. Choose one, select a real image, and compare the original with the overlay. Explain why the category and capture conditions matter. |
| 1–2 | Follow `scripts/build_mvtec.py`, `scripts/dataset.py` and `scripts/prepare.py`. Show the independent fitting, calibration and test splits for one category. |
| 2–3 | Follow `visionguard/ml/patchcore.py` from a pretrained ResNet-18 to pooled patches, the coreset and nearest-neighbor distances. Show the saved metadata. |
| 3–4 | Trace `visionguard/web/app.js` → `POST /api/predict` → upload limits → EXIF/RGB/resize → category session → cutoff and overlay. Explain what happens on a cache miss. |
| 4–5 | Read a per-category confusion matrix. Explain false alarms, fixed display scale, and the difference between model latency, response transfer and a sleeping host's wakeup. |

### Draw the algorithm from memory

1. Pixels become an oriented RGB image. Resize the whole frame to 160 × 160, then normalize channels using the ImageNet mean and standard deviation.
2. A fixed, pretrained ResNet-18 produces feature grids from layers 2 and 3. A 3 × 3 average pools neighboring features. Resize the coarse grid and concatenate it with the fine grid. Each of the 400 spatial locations has 384 feature values.
3. Fit collects patches from normal images in one category. Projected greedy k-center sampling keeps a 1% reference bank. The original Bottle fit kept 668 representative vectors out of 66,800.
4. For each input patch, compute its Euclidean distance to the nearest vector in the selected category's memory bank. The maximum distance becomes the image score. The application aligns the resized, smoothed distance map to the input preview.
5. Compare the image score against the selected model's cutoff. The overlay colors use a separate range derived from calibration map peaks. A NORMAL image can still have warm regions.

## Twenty likely questions

1. **Why PatchCore?** It is a useful fit for inspection with many good examples and few known defects. It gives both a product score and a localization signal.
2. **Why one bank per category?** A good metal nut and a good carpet have different visual patterns. Each bank describes acceptable appearance within its category. The user chooses the appropriate model before inspecting.
3. **Why a pretrained CNN?** ImageNet features already encode edges, shapes and textures. Reusing intermediate features avoids fitting a deep network from scratch on a small normal-only dataset.
4. **Why intermediate layers?** The higher-resolution grid preserves local detail and the coarser one adds context. Both retain spatial positions that can be mapped back to the image.
5. **What is the memory bank?** A matrix of representative normal patch vectors. Each row holds 384 feature values describing one region.
6. **Why a coreset?** It favors coverage of the feature space while reducing storage and comparisons. Too small a coreset can miss legitimate variation and increase false positives.
7. **Where is the anomaly score computed?** Anomalib's one-neighbor PatchCore takes the maximum nearest-normal-patch distance. The exported graph still computes that score.
8. **How did you choose the thresholds?** For each category I took the higher empirical 95th percentile of its held-out normal scores. Strict greater-than determines ANOMALY. The original Bottle run used 42 calibration normals and got a cutoff of 8.511868.
9. **How do you prevent leakage?** Fit and calibration use separate seeded subsets of `train/good` under our own split logic. Evaluation uses the frozen graph and cutoff and checks duplicate hashes. The demo examples are selected after all test predictions have been recorded.
10. **What does an anomaly score mean?** It is a feature distance. Its scale depends on the selected model and its normal reference bank. A validated conversion would be required before reporting a probability.
11. **Why was Bottle's test FPR 20% if the target was 5%?** The threshold had two exceedances among 42 calibration normals in the original run. Four of 20 separate test normals crossed it. That difference illustrates the uncertainty of a cutoff estimated from a small sample.
12. **How do AUROC, precision and recall differ?** Image AUROC measures how well raw scores rank defects above normals across cutoffs within a category. Precision and recall describe decisions at the chosen cutoff. Precision also depends on the prevalence of defects.
13. **What does pixel AUROC mean?** It ranks defect mask pixels against other pixels using the aligned anomaly map. For the full dataset release, the evaluator pools pixels within each category using 65,536 fixed score bins. That is a bounded-memory approximation of ranking quality.
14. **What changes did you make from the paper?** The demo uses a smaller ResNet-18, 160-pixel full-frame resize, a 1% coreset and a one-neighbor maximum score. These choices keep CPU inference and export small.
15. **How do you know the serving graph matches training?** Preparation compares every calibration image's raw score and map from PyTorch with its exported ONNX output. It checks absolute tolerance of 0.001 and relative tolerance of 0.0001. A separate integration test reloads and compares an export.
16. **Why FastAPI and plain JavaScript?** The use case has one upload form and one result. Serving it and the API from one process keeps deployment and debugging simple.
17. **How do you keep inference resource use predictable?** Startup verifies models one at a time. The default cache keeps one active ONNX session, and the default admission limit is one request. Streaming and pixel limits bound upload work. Browser JPEG previews reduce transfer size, and busy requests receive 429.
18. **Where can it fail?** An incorrect category, glare, lighting drift, a different background or an unfamiliar product can increase distances. Resizing can hide tiny defects. The overlay highlights regions for a human to review.
19. **Which parts are yours to explain?** Data splitting and metadata, preprocessing, calibration, evaluation, export parity, model caching, upload handling, overlays and deployment. Anomalib supplies PatchCore and timm supplies the pretrained encoder.
20. **What would you improve next?** Collect independent normal and defective images under target inspection conditions. Measure score drift and false-alarm costs and evaluate whether larger inputs recover small defects.

## Five-minute demo with a local fallback

1. Before the interview, provision the release and start the local server. Verify `/ready` lists all 15 expected categories. Run the normal and defective samples from at least two categories before relying on the live demo. [Local setup](../README.md)
2. In the browser, choose an object category and load its normal example. Click **Analyze Image**. Read the displayed score and cutoff aloud and explain why the decision follows from them.
3. Reset and run its defective example. Point out a hot region on the overlay. Switch to a texture and show how the category changes the reference bank and cutoff.
4. Show the full category test report and discuss a false alarm or miss. Explain that the UI examples were chosen from correct predictions and the benchmark includes every test image. [Full benchmark](mvtec-ad-results.md)
5. If the hosted app is waking up or the connection fails, run the same demo on `http://127.0.0.1:8000` with the prepared bundle. [Container instructions](deployment.md)

## Three resume bullets

- Built a CPU inspection demo for all 15 MVTec AD categories using Anomalib PatchCore, per-category thresholds and aligned anomaly heatmaps.
- Packaged 15 ONNX models into a checksum-verified bundle and tested 30 real predictions in an offline 512 MiB container with a bounded session cache.
- Evaluated 1,725 held-out images with 0.9344 macro image AUROC and 0.8819 macro F1, publishing per-category thresholds, confusion matrices and sample provenance.

## Seven days of practice

| Day | Do this yourself |
| --- | --- |
| 1 | Start the server and run a normal and defective sample for one object and one texture. Explain score versus cutoff without notes. |
| 2 | Trace one upload from the Analyze click through `read_upload`, `infer`, and `Graph.run`. Trigger a malformed PNG and explain the 422. |
| 3 | Read two models' `metadata.json` and `split.json`. Open each model with `onnx.load` and compare the bank or transposed matrix shapes. |
| 4 | Explain a heatmap in terms of local patch distances and the category's display cutoff. Compare a sample with an unusually lit image from the same category. |
| 5 | Copy the config, change the coreset ratio to 0.02 and use a new version and output path. Refit and rerun evaluation without adjusting parameters based on the test results. Compare size and speed. |
| 6 | Use a per-category evaluation JSON file to locate a false positive or a miss. Inspect the image and propose two hypotheses you could test. |
| 7 | Rehearse the two-minute explanation, five-minute architecture, and live demo. Practice saying what Codex helped with and which results you verified. |
