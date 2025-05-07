FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11.9
# 使用CUDA基础镜像（已包含CUDA 11.8和cuDNN 8）
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置时区（优化版）
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    cmake \
    build-essential \
    g++ \
    git \
    libopenblas-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 设置Python3.11为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# 先安装PyTorch（确保CUDA兼容性）
RUN pip install --no-cache-dir \
    torch==2.4.1+cu118 \
    torchvision==0.19.1+cu118 \
    torchaudio==2.4.1+cu118 \
    --extra-index-url https://mirrors.aliyun.com/pytorch-wheels/cu118


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


COPY . .

EXPOSE 7005

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7005"]