FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py /app/models.py
COPY emails.py /app/emails.py
COPY server/__init__.py /app/server/__init__.py
COPY server/environment.py /app/server/environment.py
COPY server/app.py /app/server/app.py

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

# Run
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
