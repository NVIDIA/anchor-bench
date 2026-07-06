# syntax=docker/dockerfile:1.6

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS kitless

ARG NEWTON_PACKAGE=newton
ARG NEWTON_VERSION=1.2.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANCHOR_BENCH_REPO=/opt/anchor-bench

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/anchor-bench

COPY pyproject.toml README.md check_environment.py ./
COPY anchor_bench ./anchor_bench
COPY assets ./assets
COPY configs ./configs
COPY examples ./examples
COPY scripts ./scripts
COPY tests ./tests
COPY versions ./versions

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[dev]" \
    && python -m pip install "${NEWTON_PACKAGE}==${NEWTON_VERSION}" cmaes matplotlib optuna pandas pyarrow tqdm

COPY docker/entrypoint.sh /usr/local/bin/anchor-bench-entrypoint
RUN chmod +x /usr/local/bin/anchor-bench-entrypoint \
    && groupadd -g 1000 anchor \
    && useradd -m -u 1000 -g 1000 -s /bin/bash anchor \
    && mkdir -p \
      /workspace/anchor-bench \
      /opt/anchor-bench/outputs \
      /home/anchor/.cache/ov \
      /home/anchor/.cache/pip \
      /home/anchor/.cache/nvidia/GLCache \
      /home/anchor/.nv/ComputeCache \
      /home/anchor/.nvidia-omniverse/logs \
      /home/anchor/.local/share/ov/data \
      /home/anchor/Documents \
    && chown -R anchor:anchor /opt/anchor-bench /workspace/anchor-bench /home/anchor

WORKDIR /opt/anchor-bench
ENV HOME=/home/anchor \
    XDG_CACHE_HOME=/home/anchor/.cache
USER anchor
ENTRYPOINT ["/usr/local/bin/anchor-bench-entrypoint"]
CMD ["bash"]

# In the `full` stage we briefly switch back to root for system-level pip installs
# (IsaacSim + IsaacLab), then drop back to the non-root `anchor` user at the end.

FROM kitless AS full

ARG NEWTON_PACKAGE=newton
ARG NEWTON_VERSION=1.2.0
ARG ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git
ARG ISAACLAB_BRANCH=develop
ARG ISAACLAB_REF=482d21c936b863e323f0a135c8e8e8e8a9c166b2
ARG ISAACLAB_DIR=/opt/IsaacLab
ARG ISAACSIM_VERSION=6.0.0.0

USER root
ENV HOME=/root \
    XDG_CACHE_HOME=/tmp/anchor-bench-root-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libx11-6 \
      libxext6 \
      libxrender1 \
      libxt6 \
      libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
      "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
      cmaes \
      optuna \
      scipy \
      pandas \
      pyarrow \
      tqdm \
      matplotlib \
    && python -m pip install --force-reinstall "${NEWTON_PACKAGE}==${NEWTON_VERSION}" \
    && git clone --branch "${ISAACLAB_BRANCH}" --single-branch "${ISAACLAB_REPO}" "${ISAACLAB_DIR}" \
    && cd "${ISAACLAB_DIR}" \
    && git fetch origin "${ISAACLAB_REF}" \
    && git checkout "${ISAACLAB_REF}" \
    && python -m pip install \
         --extra-index-url https://download.pytorch.org/whl/cu128 \
         --extra-index-url https://pypi.nvidia.com \
         --editable source/isaaclab \
    && for package in \
         source/isaaclab_assets \
         source/isaaclab_tasks \
         source/isaaclab_newton \
         source/isaaclab_visualizers; do \
         if [ -d "$package" ]; then python -m pip install --no-deps --editable "$package"; fi; \
       done \
    && python -m pip install --force-reinstall "${NEWTON_PACKAGE}[sim]==${NEWTON_VERSION}" \
    && python -m pip install --editable /opt/anchor-bench

RUN chown -R anchor:anchor "${ISAACLAB_DIR}" /opt/anchor-bench

ENV ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    HOME=/home/anchor \
    XDG_CACHE_HOME=/home/anchor/.cache \
    ANCHOR_BENCH_ISAACLAB_DIR=${ISAACLAB_DIR} \
    PYTHONPATH=${ISAACLAB_DIR}/source/isaaclab:${ISAACLAB_DIR}/source/isaaclab_assets:${ISAACLAB_DIR}/source/isaaclab_tasks:${ISAACLAB_DIR}/source/isaaclab_newton:${ISAACLAB_DIR}/source/isaaclab_visualizers

USER anchor
