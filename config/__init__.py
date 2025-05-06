"""
@Project ：pythonProject 
@File    ：__init__.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:50 
"""


class BaseConfig:
    model_path = "pre_model/faster_whisper/whisper-large-v3-turbo-ct2"
    TEMP_DIR = "temp_audio_files"
    HOST = "0.0.0.0"
    PORT = 7005


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
