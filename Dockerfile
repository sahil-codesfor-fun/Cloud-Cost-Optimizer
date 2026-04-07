FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
