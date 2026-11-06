# Rebuild progress

- Baseline: GitHub main at `f5ab175d71783f783f799d5225e8ad731366b986`, 7 commits. The local folder initially had no checkout, remote, or uncommitted user work. Fetched and inspected main, then created `codex/rebuild` and retired the tracked VGG16 prototype. Original history is preserved.
- Design: one FastAPI container and static web page. Offline Anomalib PatchCore with pretrained ResNet-18; a verified versioned ONNX graph for a smaller serving process.
- Data and model: pinned MVTec AD Bottle mirror (publisher's old archive URLs returned 404), 167 fitting normals, 42 calibration normals, 83 untouched test images, 668-row coreset. Model and dataset bytes are excluded from the Git tree.
- Evidence: [real-model workflow](https://github.com/AdityaChhimwal/visionguard/actions/runs/42185022406) passed export parity, frozen evaluation, real desktop/mobile Chromium, and a network-disabled Linux container. See [verification](verification.md) for measured metrics and latency.
- Local hardware: Ryzen 5 3550H, 8 logical CPUs, GTX 1650 4 GiB; the local Docker engine was unavailable. Large Windows dependency downloads timed out, so fresh installs and full verification used GitHub runners.
- Hosting: the verified process is within Render Free's published memory budget on the sample workload. Public trial needs an authenticated free workspace; no paid service started. Next step is [deploy and run anonymous verification](deployment.md) once hosting access is available.
- Publication: [rebuild PR](https://github.com/AdityaChhimwal/visionguard/pull/1) merged without squashing. The [Bottle release](https://github.com/AdityaChhimwal/visionguard/releases/tag/bottle-pc-v1) is public; the exact bundled container was verified offline with no volume.
- Windows serving fallback: locked runtime installed locally, verified artifact and optional samples staged, both real uploads and invalid-input handling passed.
