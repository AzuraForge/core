# ========== GÜNCELLEME: core/Dockerfile ==========
# Stage 1: Builder
FROM python:3.10-slim-bullseye AS builder

RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

# KRİTİK DÜZELTME: Çalışma dizinini doğrudan 'src' klasörünün içine ayarla
WORKDIR /app/src 

# src klasörünün içeriğini (yani azuraforge_core klasörünü) mevcut WORKDIR'e kopyala
COPY src ./src

# pyproject.toml ve setup.py'ı bir üst dizine (/app) kopyala, 
# çünkü pip install oradan çalışacak.
COPY pyproject.toml /app/
COPY setup.py /app/

# pip install komutunu ana paketin kök dizininden (/app) çalıştır.
# Bu, setup.py'ın package_dir={"": "src"} ayarını doğru algılamasını sağlar.
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir /app

# Stage 2: Runtime
FROM python:3.10-slim-bullseye AS runtime

# Runtime'da da aynı çalışma dizinini koru
WORKDIR /app/src

# Builder aşamasından kurulu paketi kopyala
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# src klasörünün içeriğini kopyala
COPY --from=builder /app/src ./src

CMD ["python", "-c", "print('AzuraForge Core library image built successfully!')"]