# syntax=docker/dockerfile:1.7
#
# Two targets from one file:
#   runtime (default)  minimal image that runs the experiments: python -m airquality
#   dev                runtime + pytest, matplotlib, tests, validation and scripts
#   api                runtime + trained model + HTTP API on port 8000
#
#   docker build -t airquality .                    # runtime
#   docker build --target dev -t airquality:dev .   # dev (tests, validation, plot)
#   docker build --target api -t airquality:api .   # API

ARG PYTHON_VERSION=3.12

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

# Non-root user. UID/GID are build args so files written to a bind-mounted
# ./results are owned by you on the host, not by root.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} app && useradd --uid ${UID} --gid ${GID} --create-home app

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# The package is run from /app/src (PYTHONPATH) and not pip-installed, because
# data.py locates data/ relative to the source tree.
COPY --chown=app:app src ./src
COPY --chown=app:app data ./data
RUN mkdir results && chown app:app results .

USER app

LABEL org.opencontainers.image.title="air-quality-ml" \
      org.opencontainers.image.description="Data-leakage experiments on world air quality data" \
      org.opencontainers.image.source="https://github.com/carmencanobermell/air-quality-ml"

# Results are written to /app/results (mount a volume to keep them)
CMD ["python", "-m", "airquality", "--out", "/app/results"]

# ---------- dev ----------
FROM runtime AS dev

USER root
COPY requirements-dev.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-dev.txt
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app tests ./tests
COPY --chown=app:app validation ./validation
COPY --chown=app:app scripts ./scripts
USER app

CMD ["pytest", "-q"]

# ---------- api ----------
FROM runtime AS api

USER root
COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-api.txt
USER app

# The model is trained while the image is built (about 10 seconds), so the image is
# self-contained and the model always matches the installed scikit-learn.
RUN python -m airquality.model --out /app/models/model.joblib
ENV MODEL_PATH=/app/models/model.joblib

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "airquality.api:app", "--host", "0.0.0.0", "--port", "8000"]
