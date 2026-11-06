# Run and deploy the container

Use the [Bottle PatchCore v1 GitHub release](https://github.com/AdityaChhimwal/visionguard/releases/tag/bottle-pc-v1) or reproduce the model following the [README](../README.md). The versioned ZIP contains the ONNX graph, metadata, split manifest, two attributed examples, and notices. Check [attribution and terms](../NOTICE.md). The expected SHA-256 is in the separate release checksum file.

Download the two assets into `data/`, verify and provision them:

```powershell
$BottleDigest = (Get-Content data/bottle-pc-v1.sha256 -Raw).Trim()
uv run --no-sync python -m scripts.provision_bundle --file data/bottle-pc-v1.zip --sha256 $BottleDigest
```

On Linux use `--sha256 "$(cat data/bottle-pc-v1.sha256)"`. The installer rejects an altered archive, unexpected files, or overwrites. The service independently checks the graph and manifest hashes and warms up the model. Download and install at build time or before service start.

## Local Linux container

Run this from the repo root:

```text
docker build -t visionguard:local .
docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/models:ro" visionguard:local
```

Check `http://localhost:8000/ready`. `/health` only means the process is alive. The Docker health check probes readiness. The container runs as an unprivileged user, uses one Uvicorn worker, and needs no model or weight downloads at startup or during requests. Make mounted files readable inside the container. The server reads environment variables from the process; it does not implicitly load `.env`. See [environment settings](../.env.example).

## Render Free web service

The public service serves the Bottle model and both example images. A fresh Chrome session loaded the page and ran both examples after a Render wakeup. [Live demo](https://visionguard-ljwy.onrender.com) · [Verification results](verification.md#public-render-verification)

The Linux test runner measured 187.57 MiB peak process memory while serving 900 × 900 samples. The Render configuration uses a 4-megapixel decoded image limit and one ONNX thread. Render can sleep after inactivity. The Docker runtime passes the public bundle settings below as build arguments, so the build checks the model archive and includes it in the image. [Render Free](https://render.com/docs/free) · [Docker builds](https://render.com/docs/docker)

To create a separate service, sign in and apply the Blueprint with the **Free** plan selected. The Blueprint disables automatic deploys. Use **Manual Deploy** in its dashboard when you need to publish an application change. [Deploy on Render Free](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FAdityaChhimwal%2Fvisionguard) · [Blueprint](../render.yaml)

For manual setup:

1. Sign in to Render, create a **Web Service** from the public Git repository `https://github.com/AdityaChhimwal/visionguard`, and select the verified branch (normally `main` after merge). Select **Docker**, root `.`, Dockerfile `./Dockerfile`, and **Free**. Do not select paid compute.
2. Under Advanced set **Health Check Path** to `/ready`. Use these settings:

| Key | Value |
| --- | --- |
| `VG_BUNDLE_URL` | `https://github.com/AdityaChhimwal/visionguard/releases/download/bottle-pc-v1/bottle-pc-v1.zip` |
| `VG_BUNDLE_SHA256` | Exact value from `bottle-pc-v1.sha256` |
| `PORT` | `8000` |
| `VG_THREADS` | `1` |
| `VG_INFERENCE_SLOTS` | `1` |
| `VG_MAX_PIXELS` | `4000000` |
| `VG_UPLOAD_BYTES` | `6291456` |

3. After deploying, open the assigned hostname in a signed-out browser and analyze both fixed examples. Check the label and heatmap. Run `python -m scripts.verify_public --url https://visionguard-ljwy.onrender.com` to check the current public service over HTTPS.
