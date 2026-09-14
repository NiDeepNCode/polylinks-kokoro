FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    flask \
    kokoro \
    soundfile \
    numpy

COPY app.py .

EXPOSE 8080

CMD ["python", "app.py"]
