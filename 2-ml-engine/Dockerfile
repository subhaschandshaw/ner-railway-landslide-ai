FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8001

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py serve.py ./

RUN mkdir -p /app/models /app/outputs

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Auto-train on start if model is not pre-mounted, then serve
CMD python -c "import os, train; os.path.exists('models/landslide_model.joblib') or train.train_pipeline()" && \
    uvicorn serve:app --host 0.0.0.0 --port 8001