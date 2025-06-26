# 第一阶段：构建阶段（使用CUDA开发镜像）
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.2.2-base-ubuntu22.04 AS builder
# 设置时区和系统环境
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

# 配置系统基础环境
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    cmake \
    ninja-build \
    build-essential \
    libopenblas-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 升级pip和工具链
RUN python3 -m pip install --upgrade pip setuptools wheel -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 安装PyTorch
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --index-url https://pypi.tuna.tsinghua.edu.cn/project/pytorch
# 安装项目依赖
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY . .

# 第二阶段：生产镜像（使用CUDA运行时镜像）
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.2.2-base-ubuntu22.04

# 设置运行时环境
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

# 安装运行时依赖
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 从构建阶段复制 CUDA 库和 Python 依赖
COPY --from=builder /usr/local/cuda /usr/local/cuda
#COPY --from=builder /usr/lib/x86_64-linux-gnu/libcudnn* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# 设置 CUDA 环境变量
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda/bin:$PATH

# 复制应用代码
WORKDIR /app
COPY --from=builder /app /app

# 优化运行参数
ENV OMP_NUM_THREADS=1 \
    TF_ENABLE_ONEDNN_OPTS=0

EXPOSE 7005
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7005", "--workers", "4"]
#CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:7005", "--workers", "2", "main:app"]
