FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (opencv / moviepy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-full.txt ./requirements-full.txt
RUN pip install --no-cache-dir -r requirements-full.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

# Use a shell so ${PORT} is expanded (Render injects PORT at runtime).
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 4 --timeout 180 -b 0.0.0.0:${PORT} app:app"]
