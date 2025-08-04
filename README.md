# v3 版本单gpu 运行方式

```cmd
docker build -t audio-split-server-v3 . 
docker run -d --gpus all -p 7005:7005 --name audio-split-server-v3 audio-split-server-v3
docker run  -v /home/ubuntu/audio-project/temp_audio_files:/app/temp_audio_files -d --gpus all -p 7005:7005 --name audio-split-server-v3 audio-split-server-v3 
```

# v4 多服务器运行方式

```cmd
docker build -t audio-split-server-v4 .
docker-compose up --build --force-recreate --scale app_gpu0=1 --scale app_gpu1=1
```

```cmd
-v /etc/localtime:/etc/localtime
# 服务1
docker run  -v /etc/localtime:/etc/localtime -v /home/ubuntu/audio_temp_audio_files:/app/temp_audio_files -d --gpus '"device=GPU-389d06c9-0857-80d3-3d33-9311e1954a5f"' -p 7005:7005 --name audio-split-server-v1 audio-split-server
# 服务2
docker run  -v /etc/localtime:/etc/localtime -v /home/ubuntu/audio_temp_audio_files:/app/temp_audio_files -d --gpus '"device=GPU-389d06c9-0857-80d3-3d33-9311e1954a5f"' -p 7005:7005 --name audio-split-server-v2 audio-split-server

```

# 最新

```cmd
 docker run  -v /home/ubuntu/audio_temp_audio_files:/app/temp_audio_files -d --gpus all -p 7005:7005 --name audio-split-server-v1 audio-split-server
```

# 容器操作记录

```bash
source activate
conda create -n audio-project python==3.10
conda activate audio-project 

sudo apt update
sudo apt install ffmpeg -y
python main.py
```