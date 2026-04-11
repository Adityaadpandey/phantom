# Stage 1: build deps
FROM python:3.12-slim-bookworm AS builder
WORKDIR /app
# Copy only the manifest first so dep installation is cached independently of source changes
COPY pyproject.toml .
# Install all declared dependencies (cached unless pyproject.toml changes)
RUN pip install --no-cache-dir \
    "pydantic>=2.0" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "pyyaml>=6.0" \
    "httpx>=0.27" \
    "python-dotenv>=1.0" \
    "openenv-core>=0.2.0" \
    "openai>=1.0"
# Now copy source and install the package itself (no dep re-download)
COPY phantom/ phantom/
RUN pip install --no-cache-dir --no-deps .

# Stage 2: production image
FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY phantom/ phantom/
COPY server/ server/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

ENV PHANTOM_ENV=production
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "phantom.api:app", "--host", "0.0.0.0", "--port", "7860"]
