FROM python:3.12.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "app.main"]
