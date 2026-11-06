FROM ghcr.io/astral-sh/uv:0.10.12 AS uv
FROM python:3.13-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 \
    VG_ARTIFACTS=/models VG_SAMPLES=/models/samples PATH=/app/.venv/bin:$PATH
COPY --from=uv /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-python-downloads --no-cache
COPY configs/ configs/
COPY visionguard/ visionguard/
COPY scripts/provision_bundle.py /app/scripts/provision_bundle.py
ARG VG_BUNDLE_URL=""
ARG VG_BUNDLE_SHA256=""
RUN if [ -n "$VG_BUNDLE_URL" ] || [ -n "$VG_BUNDLE_SHA256" ]; then \
      python /app/scripts/provision_bundle.py --url "$VG_BUNDLE_URL" --sha256 "$VG_BUNDLE_SHA256" --target /models; \
    fi
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/ready',timeout=2)"
CMD ["sh", "-c", "exec uvicorn visionguard.api.main:app --host 0.0.0.0 --port \"$PORT\" --workers 1 --no-access-log"]
