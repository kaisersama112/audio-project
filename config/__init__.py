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

TEMP_DIR = "temp_audio_files"

DOWNLOAD_DIR = os.path.join(TEMP_DIR, "downloads")
# 正式站
base_url = "https://audio.cqhuancheng.cn/"


# 测试站
# base_url = "https://audiotest.cqhckj.cn/"


class MysqlConfig:
    """
    mysql 配置
    """
    # ---------正式站--------------
    host = "1.14.127.39"
    user = "broadcast_ai"
    password = "2Afsp2cGCdk7dRf8"
    database = "broadcast_ai"
    port = 3388
    charset = "utf8mb4"
    # ---------测试站--------------
    # host = "123.57.150.136"
    # user = "broadcast_ai"
    # password = "eMRtryH6LcpidGRR"
    # database = "broadcast_ai"
    # port = 3306
    # charset = "utf8mb4"


class UCloudConfig:
    """
    oss 配置
    """
    UCloud_file_public_key = "4eZCa18pxZ7GGHmWzV4PL2IiA1HHnMc2H"
    UCloud_file_private_key = "FAFUxPmdnJsD6vKpBe3SiGEcoXEzAuwiSLpoLR18L2FX"
    UCloud_file_bucket = "aipublic"
    neiwang = ".internal-cn-wlcb.ufileos.com"
    waiwang = ".cn-wlcb.ufileos.com"
    UCloud_file_upload_suffix = neiwang
    UCloud_file_download_suffix = neiwang
