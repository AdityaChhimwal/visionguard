# Rebuild progress

- Baseline: GitHub main at `f5ab175d71783f783f799d5225e8ad731366b986`, 7 commits. The local folder initially had no checkout, remote, or uncommitted user work. Fetched main, inspected tracked files and instructions, and branched to `codex/rebuild`.
- Design: one FastAPI container and static web page. Offline Anomalib PatchCore with a pretrained ResNet-18; versioned ONNX graph for a smaller serving process. Export must match PyTorch before release.
- Milestones: verify dataset and pinned dependencies; fit and calibrate on separate normal images; evaluate untouched test split; validate API, browser and container; review and publish Git history; verify anonymous HTTPS.
- Hardware: Ryzen 5 3550H, 8 logical CPUs; GTX 1650 (4 GiB) available. Start with CPU for parity with deployment.
- Dataset: historical publisher URLs return 404. Downloading pinned MVTec AD Bottle mirror under non-commercial terms; provenance and counts will be recorded.
- Hosting: Docker Desktop installed, engine initially unavailable. No known authenticated hosting account. Render Free is a candidate only if measured footprint fits.
