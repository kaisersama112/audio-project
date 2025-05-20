
docker build -t audio-split-server-v3 . 
docker run -d --gpus all -p 7005:7005 --name audio-split-server-v3 audio-split-server-v3
docker run -d --gpus '"device=0,1"' -p 7005:7005 --name audio-split-server-v3 audio-split-server-v3