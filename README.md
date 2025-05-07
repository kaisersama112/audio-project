
docker build -t audio-split-server-v1 . 
docker run -d --gpus all -p 7005:7005 --name audio-split-server-v1 audio-split-server:v1.0