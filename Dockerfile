FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY talkingtaco ./talkingtaco
COPY scripts ./scripts

# Bot runs as a worker: no ports, no root.
RUN useradd --create-home --shell /bin/bash taco
USER taco

CMD ["python", "-m", "talkingtaco"]
