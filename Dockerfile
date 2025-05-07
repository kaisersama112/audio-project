
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata


FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


ENV TZ=Asia/Shanghai
RUN apt-get update && \
    apt-get install -y tzdata && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip install --no-cache-dir \
    torch==2.4.1+cu118 \
    torchvision==0.19.1+cu118 \
    torchaudio==2.4.1+cu118 \
     -f https://mirrors.aliyun.com/pytorch-wheels/cu118

COPY . .


EXPOSE 7005

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7005"]