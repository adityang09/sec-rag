FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf
COPY requirements.txt .
# CPU-only torch keeps the image ~2GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
# Bake the reranker into the image so cold starts do not download it
RUN python -c "__import__('sentence_transformers').CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
COPY . .
EXPOSE 8000
CMD ["uvicorn", "sec_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
