# =====================================================================
# Dockerfile for EMA-MRI integration project
#   Base: PyTorch公式イメージ(-develタグ。pyworld等のネイティブビルドに必要)
# =====================================================================
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ---------------------------------------------------------------------
# OS依存パッケージ(shuushiのDockerfileと同趣旨)
# ---------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        ca-certificates \
        libsndfile1 \
        ffmpeg \
        sox \
        flac \
        libgl1 \
        libglib2.0-0 \
        swig \
        libpulse-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# Node.js 20 + Claude Code CLI(コンテナ内でclaudeコマンドを使う用)
# ---------------------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

WORKDIR /workspace

RUN python -m pip install --upgrade pip "setuptools<81" wheel cython

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# プロジェクト本体は実行時にマウントする運用(docker-compose.yml参照)
CMD ["/bin/bash"]
