FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    VISIONGUARD_ARTIFACT_DIR=/models/bottle_v1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system visionguard \
    && useradd --system --gid visionguard --create-home visionguard

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache ImageNet VGG16 weights in the image. The API reconstructs the documented
# architecture once at startup and never downloads weights per request.
RUN python -c "from tensorflow.keras.applications import VGG16; VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))"

COPY src ./src
COPY api ./api

RUN chown -R visionguard:visionguard /app
USER visionguard

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
