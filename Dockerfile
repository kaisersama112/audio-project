# 第一阶段：构建阶段（使用CUDA开发镜像）
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

# 设置时区和系统环境
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

# 配置系统基础环境
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    cmake \
    ninja-build \
    build-essential \
    libopenblas-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 升级pip和工具链
RUN python3 -m pip install --upgrade pip setuptools wheel -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装PyTorch
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --extra-index-url https://mirrors.aliyun.com/pytorch-wheels/cu118

# 安装项目依赖
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 第二阶段：生产镜像（使用CUDA运行时镜像）
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置运行时环境
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

# 安装运行时依赖
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 从构建阶段复制安装内容
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# 复制应用代码
WORKDIR /app
COPY . .

# 优化运行参数
ENV OMP_NUM_THREADS=1 \
    TF_ENABLE_ONEDNN_OPTS=0

EXPOSE 7005
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7005", "--workers", "2"]