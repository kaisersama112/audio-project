# v3 版本单gpu 运行方式

```cmd
docker build -t audio-split-server-v3 . 
docker run -d --gpus all -p 7005:7005 --name audio-split-server-v3 audio-split-server-v3
docker run  -v /home/ubuntu/audio-project/temp_audio_files:/app/temp_audio_files -d --gpus all -p 7005:7005 --name audio-split-server-v3 audio-split-server-v3 
```

# v4 多服务器运行方式

```cmd
docker build -t audio-split-server-v4.
docker-compose up --build --force-recreate --scale app_gpu0=1 --scale app_gpu1=1
```