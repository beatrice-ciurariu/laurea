FROM python:3.10

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U crawl4ai

COPY backend/requirements-advanced-evaluation.lock.txt ./
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-advanced-evaluation.lock.txt
RUN crawl4ai-setup

COPY backend/src/ .
COPY domains.json ./domains.json
COPY gs_data/ ./gs_data

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]