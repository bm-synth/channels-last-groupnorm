# Copyright (c) 2025 Synthesia Limited - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.

# Python version
ARG PYTHON_VERSION="311"

FROM 812634467557.dkr.ecr.eu-west-1.amazonaws.com/synthesia-cudnn-python:12.2.2-cudnn8-runtime-ubuntu22.04-py${PYTHON_VERSION}-latest AS base

##################
# Common Image. Used to pre-install dependencies and leverage cache
##################
FROM base as image-common

# Disable uv cache
ENV UV_NO_CACHE=true

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /root

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

# hadolint ignore=SC1091
RUN --mount=type=secret,id=uv_creds,dst=/root/.config/uv/uv.toml  \
    touch README.md && \
    uv venv && \
    . /root/.venv/bin/activate && \
    uv sync --frozen --no-install-project --no-dev --inexact --config-file /root/.config/uv/uv.toml
ENV PATH="/root/.venv/bin:$PATH"

##################
# Local Image. Copy code from local. This image is used when the `--local` flag is specified to `build.py`
##################
FROM image-common as image-local
# Name of the package, as it appears in the source folder e.g. {package}
ARG PYTHON_PACKAGE_NAME
ARG PIP_PACKAGE_NAME

WORKDIR /root

ENV PIP_PACKAGE_NAME=${PIP_PACKAGE_NAME}

RUN --mount=type=secret,id=uv_creds,dst=/root/.config/uv/uv.toml  \
    uv sync --frozen --all-extras --no-install-project --inexact --config-file /root/.config/uv/uv.toml

COPY repo.tar.gz repo/repo.tar.gz

WORKDIR /root/repo

# Remove --no-dependencies flag if requirements.txt is not installed in the common image
# hadolint ignore=SC1091
RUN --mount=type=secret,id=uv_creds,dst=/root/.config/uv/uv.toml  \
    tar -xf repo.tar.gz && \
    uv sync --frozen --all-extras --inexact --config-file /root/.config/uv/uv.toml

WORKDIR /app

ENV PIP_PACKAGE_NAME=${PIP_PACKAGE_NAME}
ENV PYTHON_PACKAGE_NAME=${PYTHON_PACKAGE_NAME}

# hadolint ignore=DL3025
CMD python -m "$PYTHON_PACKAGE_NAME.main" --salutation "Hello"

##################
# Production Image. Install code from CodeArtifact
##################
FROM image-common as image-prod

# Name of the package, as it appears in CodeArtifacts e.g. synthesia-{package}
ARG PIP_PACKAGE_NAME
# Version of the package. Defaults to latest.
ARG PIP_PACKAGE_VERSION=
# Name of the package, as it appears in the source folder e.g. {package}
ARG PYTHON_PACKAGE_NAME

# Create package name depending on whether version is specified or not
# Saving it to a file is necessary in order to make it available during the `uv pip install` phase
RUN if [ -z "${PIP_PACKAGE_VERSION}" ] ; then \
    echo "${PIP_PACKAGE_NAME}" > .package_env ; \
    else \
    echo "${PIP_PACKAGE_NAME}==${PIP_PACKAGE_VERSION}" > .package_env ;  \
    fi


# Temporarily mount credentials in order to log into AWS CodeArtifact & install package
# Remove --no-dependencies flag if requirements.txt is not installed in the common image
# hadolint ignore=DL3013,DL3042,SC2046, SC1091
RUN --mount=type=secret,id=uv_creds,dst=/root/.config/uv/uv.toml  \
    export PIP_PACKAGE=$(cat .package_env) && \
    uv pip install "${PIP_PACKAGE}" --no-deps --config-file /root/.config/uv/uv.toml

WORKDIR /app

ENV PIP_PACKAGE_NAME=${PIP_PACKAGE_NAME}
ENV PYTHON_PACKAGE_NAME=${PYTHON_PACKAGE_NAME}

# hadolint ignore=DL3025
CMD python -m "$PYTHON_PACKAGE_NAME.main" --salutation "Hello"
