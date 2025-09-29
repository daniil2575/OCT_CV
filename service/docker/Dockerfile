FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .[api]
ENV PYTHONUNBUFFERED=1
EXPOSE ***
CMD ["uvicorn", "service.app:app", "--host", "***", "--port", "***"]
