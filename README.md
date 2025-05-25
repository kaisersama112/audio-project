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
# 服务1
docker run  -v /home/ubuntu/audio_temp_audio_files:/app/temp_audio_files -d --gpus '"device=GPU-118b1917-4dce-31b6-2cd8-5b4cc7449d94"' -p 7005:7005 --name audio-split-server-v3 audio-split-server-v3
# 服务2
docker run -d --gpus '"device=GPU-472005c6-3cbb-22a4-8eaf-70beed892fc2"' -v /home/ubuntu/audio_temp_audio_files:/app/temp_audio_files -p 7006:7006 --name audio-split-server-v4 audio-split-server-v4

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