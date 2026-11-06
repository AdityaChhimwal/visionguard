# Run and deploy the container

Use the versioned archive with 15 model graphs, their metadata and split manifests, 30 attributed examples, and license notices. The release includes an SHA-256 checksum and a separate evaluation ZIP. [Model release](https://github.com/AdityaChhimwal/visionguard/releases/tag/mvtec-ad-pc-v1) · [Attribution](../NOTICE.md) · [Reproduction instructions](../README.md#prepare-and-evaluate-from-scratch)

Download the two assets into `data/`, verify and provision them:

```powershell
$Digest = (Get-Content data/mvtec-ad-pc-v1.sha256 -Raw).Trim()
uv run --no-sync python -m scripts.provision_bundle --file data/mvtec-ad-pc-v1.tar.xz --sha256 $Digest
```

On Git Bash or Linux use `--sha256 "$(cat data/mvtec-ad-pc-v1.sha256)"`. The installer verifies the archive and its expected files before moving the staged directory into place. For upgrades, move the previous `artifacts/` directory to a backup first. The service verifies each graph and keeps a bounded cache of sessions.

## Local Linux container

Run this from the repo root:

```text
docker build -t visionguard:local .
docker run --rm -p 8000:8000 -e VG_REQUIRE_ALL_MODELS=1 -v "${PWD}/artifacts:/models:ro" visionguard:local
```

Check `http://localhost:8000/ready`. `/health` only means the process is alive. The Docker health check probes readiness. The container runs as an unprivileged user, uses one Uvicorn worker, and needs no model or weight downloads at startup or during requests. Make mounted files readable inside the container. The server reads environment variables from the process; it does not implicitly load `.env`. See [environment settings](../.env.example).

## Render Free web service

The Blueprint pins the full-category release. Check the public readiness response to see which models are available on the running service. [Live demo](https://visionguard-ljwy.onrender.com) · [Hosted readiness](https://visionguard-ljwy.onrender.com/ready)

An offline Linux container limited to 512 MiB and 0.1 CPU reached readiness in 91 seconds, peaked at 201.64 MiB in the container memory controller, and completed all 30 example predictions. Median HTTP prediction time was 1.70 seconds in that test. Render can sleep after inactivity. The Docker build verifies the pinned archive and includes the models in the image. [Release evidence](https://github.com/AdityaChhimwal/visionguard/releases/download/mvtec-ad-pc-v1/mvtec-ad-pc-v1-evaluation.zip) · [Render Free](https://render.com/docs/free)

For the existing service, sync the Blueprint after its bundle URL and checksum change. With Blueprint Auto Sync enabled, Render applies changes to affected resources from the linked branch. With Auto Sync disabled, use **Manual Sync** on that Blueprint. [Blueprint configuration](../render.yaml) · [Render Blueprint guide](https://render.com/docs/infrastructure-as-code)

For manual setup:

1. Sign in to Render, create a **Web Service** from the public Git repository `https://github.com/AdityaChhimwal/visionguard`, and select the verified branch (normally `main` after merge). Select **Docker**, root `.`, Dockerfile `./Dockerfile`, and **Free**. Do not select paid compute.
2. Under Advanced set **Health Check Path** to `/ready`. Use these settings:

| Key | Value |
| --- | --- |
| `VG_BUNDLE_URL` | `https://github.com/AdityaChhimwal/visionguard/releases/download/mvtec-ad-pc-v1/mvtec-ad-pc-v1.tar.xz` |
| `VG_BUNDLE_SHA256` | Exact value from `mvtec-ad-pc-v1.sha256` |
| `PORT` | `8000` |
| `VG_THREADS` | `1` |
| `VG_INFERENCE_SLOTS` | `1` |
| `VG_MODEL_CACHE_SIZE` | `1` |
| `VG_REQUIRE_ALL_MODELS` | `1` |
| `VG_MAX_PIXELS` | `4000000` |
| `VG_UPLOAD_BYTES` | `6291456` |

3. After deploying, check all 15 model IDs in `/ready`. Run `python -m scripts.verify_public --url https://visionguard-ljwy.onrender.com --all-categories` to check the current public service over HTTPS, then try an object and a texture in the browser.
