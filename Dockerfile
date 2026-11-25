FROM python:3.12-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KT_KULLANICI_DB=/var/lib/kt/kullanicilar.json \
    KT_GECMIS_DIZIN=/var/lib/kt/gecmis \
    KT_MODEL_ONBELLEK=/var/lib/kt/models/ecapa

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin kt \
    && mkdir -p /var/lib/kt/gecmis /var/lib/kt/models/ecapa \
    && chown -R kt:kt /var/lib/kt /home/kt

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Keep the image limited to application code. Dataset, notebooks, accounts,
# user history and model weights belong outside the image.
COPY app.py ./app.py
COPY src ./src

USER kt
EXPOSE 7860
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
