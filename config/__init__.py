"""
@Project ：pythonProject 
@File    ：__init__.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:50 
"""
import os


class BaseConfig:
    model_path = "pre_model/faster_whisper/whisper-large-v3-turbo-ct2"
    TEMP_DIR = "temp_audio_files"
    HOST = "0.0.0.0"
    PORT = 7005
    num_workers = os.cpu_count()
    reload = False
    quantize = True
    use_fp16 = True
    cache_dir = "./cache"


OSS_CONFIG = {
    "access_key_id": "LTAI5tGE9kR3DQYcFWJfKkk8",
    "access_key_secret": "bngm8wNfbMw9oFcjiNJaIgRJ93Czvc",
    "bucket_name": "ailive2025",
    "endpoint": "oss-accelerate.aliyuncs.com",
    "cdn_domain": "ailive2025.oss-cn-chengdu.aliyuncs.com",
    "ssl": True,
    "is_cname": False
}

settings = BaseConfig()

# 需要检测的敏感词
hotword_list = [
    "运费险",
    "七天无理由",
    "退款",
    "百分百",
    "纯羊毛",
    "号链接",
    "价格"
]
