# Two stages: install dependencies once into a venv, then copy just that venv
# into a clean runtime image. Keeps uv and the build cache out of the shipped
# layer, and means editing app code doesn't reinstall dependencies.
FROM python:3.12-slim AS builder

# Pinned, not :latest -- this image is meant to build identically on someone
# else's machine months from now.
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Dependency layer: cached unless pyproject.toml changes.
COPY pyproject.toml ./
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python -r pyproject.toml


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Run unprivileged.
RUN useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app app/ ./app/
COPY --chown=app:app data/ ./data/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app pyproject.toml ./

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
