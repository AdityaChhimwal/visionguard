# MVTec AD evaluation

This benchmark covers all 15 categories in the `mvtec-ad-pc-v1` bundle. The models were trained and scored on GitHub-hosted Linux CPU runners with two threads per job at commit `bf2e3a8`. Each category uses Anomalib PatchCore with the frozen ImageNet-pretrained ResNet-18 A1 encoder, layer2 and layer3 features, 160 × 160 inputs and a 1% coreset. The same seed of 42 is applied to the normal split, Torch and NumPy. The benchmark scores the exported ONNX graphs used for inference. [Training run](https://github.com/AdityaChhimwal/visionguard/actions/runs/42259953023) · [Pinned MVTec AD mirror](https://huggingface.co/datasets/foersben/mvtec-ad/tree/c75b39616f84db43677bcc8228caaafaf5096d7f)

The first release's measurements describe its separate Bottle model. The Bottle row below uses the graph trained for the full-category bundle. [Historical Bottle verification](verification.md)

| Category | Fit / calibration normals | Test normal / defective | Image AUROC | Pixel AUROC (binned) | Precision | Recall | F1 | TN / FP / FN / TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle | 167 / 42 | 20 / 63 | 0.9992 | 0.9719 | 0.9545 | 1.0000 | 0.9767 | 17 / 3 / 0 / 63 |
| Cable | 179 / 45 | 58 / 92 | 0.9818 | 0.9773 | 0.9545 | 0.9130 | 0.9333 | 54 / 4 / 8 / 84 |
| Capsule | 175 / 44 | 23 / 109 | 0.8787 | 0.9586 | 0.9577 | 0.6239 | 0.7556 | 20 / 3 / 41 / 68 |
| Carpet | 224 / 56 | 28 / 89 | 0.9390 | 0.9768 | 0.8913 | 0.9213 | 0.9061 | 18 / 10 / 7 / 82 |
| Grid | 211 / 53 | 21 / 57 | 0.7277 | 0.8459 | 0.9200 | 0.4035 | 0.5610 | 19 / 2 / 34 / 23 |
| Hazelnut | 312 / 79 | 40 / 70 | 0.9929 | 0.9836 | 1.0000 | 0.9571 | 0.9781 | 40 / 0 / 3 / 67 |
| Leather | 196 / 49 | 32 / 92 | 1.0000 | 0.9813 | 0.8214 | 1.0000 | 0.9020 | 12 / 20 / 0 / 92 |
| Metal Nut | 176 / 44 | 22 / 93 | 0.9746 | 0.9809 | 1.0000 | 0.9140 | 0.9551 | 22 / 0 / 8 / 85 |
| Pill | 213 / 54 | 26 / 141 | 0.8322 | 0.9686 | 0.9596 | 0.6738 | 0.7917 | 22 / 4 / 46 / 95 |
| Screw | 256 / 64 | 41 / 119 | 0.8795 | 0.9301 | 0.9615 | 0.6303 | 0.7614 | 38 / 3 / 44 / 75 |
| Tile | 184 / 46 | 33 / 84 | 0.9942 | 0.9178 | 0.9877 | 0.9524 | 0.9697 | 32 / 1 / 4 / 80 |
| Toothbrush | 48 / 12 | 12 / 30 | 0.9139 | 0.9767 | 0.9032 | 0.9333 | 0.9180 | 9 / 3 / 2 / 28 |
| Transistor | 170 / 43 | 60 / 40 | 0.9788 | 0.9804 | 0.9250 | 0.9250 | 0.9250 | 57 / 3 / 3 / 37 |
| Wood | 197 / 50 | 19 / 60 | 0.9737 | 0.9000 | 0.8939 | 0.9833 | 0.9365 | 12 / 7 / 1 / 59 |
| Zipper | 192 / 48 | 32 / 119 | 0.9504 | 0.9525 | 0.9504 | 0.9664 | 0.9583 | 26 / 6 / 4 / 115 |
| **Macro mean / totals** | 2,900 / 729 | 467 / 1,258 | 0.9344 | 0.9535 | 0.9387 | 0.8532 | 0.8819 | 398 / 69 / 205 / 1,053 |

The reports cover all 1,725 test images and 1,609,983,136 original-resolution pixels. Macro means weight each category equally. The last column sums decisions at the individual category thresholds.

Each model uses disjoint normal images for fitting and calibration. The frozen threshold targets a 5% false-positive rate on its held-out normal split. This rate can differ on the test split. Precision, recall and F1 treat defects as positive.

Pixel AUROC compares original-resolution masks with bilinearly aligned anomaly maps. It approximates score ranking with 65,536 fixed bins, using one image at a time. Scores in the same bin are treated as ties. These results apply to the MVTec AD benchmark and its imaging conditions.

Grid recall was 0.4035 and Leather had 20 false positives on 32 normal test images. The app's examples are selected after evaluation as the first correctly classified normal and defective images in each category. The table includes every test image and captures the errors those examples leave out.

Two separate Toothbrush training runs on the same Windows CPU produced byte-identical ONNX graphs and split files. All 42 test scores and labels, the threshold and the reported metrics also matched exactly. The GitHub reports include the graph and split SHA-256 values, model settings, command, runtime information, example hashes, and per-image scores for each category. [Download the full evaluation reports](https://github.com/AdityaChhimwal/visionguard/releases/download/mvtec-ad-pc-v1/mvtec-ad-pc-v1-evaluation.zip)

To regenerate this table, extract the evaluation ZIP to `outputs/mvtec-reports` and run:

```bash
python -m scripts.report_mvtec --reports outputs/mvtec-reports
```
