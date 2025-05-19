"""
@Project ：audio-split-src 
@File    ：oss_service.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：10/5/2025 下午5:13 
"""
import os

import oss2
from config import settings, OSS_CONFIG
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential


class OSSService:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cfg = OSS_CONFIG
            auth = oss2.Auth(cfg['access_key_id'], cfg['access_key_secret'])
            cls._instance = super().__new__(cls)

            if cfg['is_cname']:
                cls._instance.bucket = oss2.Bucket(
                    auth, cfg['cdn_domain'], cfg['bucket_name'],
                    connect_timeout=30
                )
            else:
                endpoint = f"{'https' if cfg['ssl'] else 'http'}://{cfg['endpoint']}"
                cls._instance.bucket = oss2.Bucket(
                    auth, endpoint, cfg['bucket_name'],
                    connect_timeout=30
                )

        return cls._instance

    def generate_file_path(self, task_id: str, filename: str) -> str:
        """生成OSS存储路径"""
        today = datetime.now().strftime("%Y%m%d")
        return f"audio_segments/{today}/{task_id}/{filename}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def upload_file(self, local_path: str, task_id: str) -> str:
        """上传文件到OSS"""
        try:
            filename = os.path.basename(local_path)
            oss_path = self.generate_file_path(task_id, filename)

            ext = os.path.splitext(filename)[1].lower()
            content_type = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav'
            }.get(ext, 'application/octet-stream')

            headers = {
                'Content-Type': content_type,
                'x-oss-forbid-overwrite': 'false'
            }

            result = self.bucket.put_object_from_file(
                oss_path, local_path, headers=headers
            )

            if result.status == 200:
                res_data=result.resp.response
                return res_data.url
            raise Exception("Upload failed")

        except oss2.exceptions.OssError as e:
            raise Exception(f"OSS Error: {e}")
        except Exception as e:
            raise Exception(f"Upload error: {e}")


oss_service = OSSService()

# if __name__ == '__main__':
#
#     data=oss_service.upload_file(
#         r"F:\python_project\audio-project\pythonProject\audio-split-src\temp\audio\20250510115240.wav",
#                             "111"
#     )
#     print(data)