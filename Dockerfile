# ── Base image ──
FROM python:3.10-slim

# ── System dependencies (needed for OpenCV + psycopg2) ──
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ──
WORKDIR /app

# ── Install Python dependencies first (cached layer) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy entire project ──
COPY . .

# ── Expose Flask port ──
EXPOSE 5000

# ── Start the app ──
CMD ["python", "app.py"]