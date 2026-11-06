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

The [measured process](verification.md) peaked at 187.57 MiB while serving 900 × 900 samples. [Render Free](https://render.com/docs/free) provides [512 MB and 0.1 CPU](https://render.com/docs/compute-plans). Use a 4-megapixel decoded limit and one ONNX thread for the first public trial; real host latency and peak usage still need checking. Render sleeps after inactivity and the first request can take longer. The [Docker runtime](https://render.com/docs/docker) passes the non-secret settings below through as build arguments, so the asset is checked and baked into the image.

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

3. Deploy, then open the assigned HTTPS hostname in a signed-out browser and analyze both fixed examples. Check that the overlay renders and that an invalid file reports an error. Run `python -m scripts.verify_public --url https://YOUR-SERVICE.onrender.com` for anonymous readiness, two real predictions and a malformed upload check. Record the first page latency after an observed sleep, along with warm results and any wrong classifications.
4. Add the HTTPS link to the README only after the anonymous workflow passes.

Vercel can host FastAPI, but its [function payload limits](https://vercel.com/docs/functions/limitations) require a separate API size budget. A new [Hugging Face Docker Space](https://huggingface.co/docs/hub/spaces-overview) requires an eligible plan. Render is the preferred path for the tested Docker build. Deployment is pending access to a signed-in free hosting workspace. No paid service has been started.
