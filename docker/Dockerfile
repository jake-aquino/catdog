FROM tensorflow/tensorflow:2.15.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

# Run FastAPI app (expects app/server.py)
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
