FROM python:3.10

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install -r requirements.txt

COPY frontend/src/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]