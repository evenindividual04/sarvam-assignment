FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# V3.8: pre-warm the bge-small-en-v1.5 ONNX weights into an image layer so the
# *first* user query after container start doesn't pay the 2-4s cold-start.
# Runs ONLY when requirements.txt changes (sits above the source COPY); on HF
# Spaces the model downloads from HF's own CDN, typically sub-5s in their
# build environment.
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
