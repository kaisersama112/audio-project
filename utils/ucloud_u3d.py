import base64
import hmac
from hashlib import sha1
from typing import Dict, Any

from fastapi import HTTPException
from ufile import config, filemanager

from config import UCloudConfig

PUBLIC_KEY = UCloudConfig.UCloud_file_public_key
PRIVATE_KEY = UCloudConfig.UCloud_file_private_key


class Authroizator:
    def __init__(self, json_data: Dict[str, Any]):
        self.json_data = json_data

    def calculateAuthSignature(self) -> str:
        """

        """
        # 参数校验
        if not self.json_data.get("method") or not self.json_data.get("bucket"):
            raise HTTPException(status_code=400, detail="'method' and 'bucket' are required!")

        method = self.json_data.get("method")
        bucket = self.json_data.get("bucket")
        content_type = self.json_data.get("content_type", "")
        content_md5 = self.json_data.get("content_md5", "")
        date = self.json_data.get("date", "")
        key = self.json_data.get("key", "")

        content = f"{method}\n{content_md5}\n{content_type}\n{date}\n"
        content += f"/{bucket}/{key}"

        signature = self.__signature(content)
        return f"UCloud {PUBLIC_KEY}:{signature}"

    def calculatePrivateUrlAuthroization(self) -> str:
        """

        """
        # 参数校验
        required_fields = ["method", "bucket", "key", "expires"]
        for field in required_fields:
            if not self.json_data.get(field):
                raise HTTPException(status_code=400, detail=f"'{field}' is required!")

        method = self.json_data.get("method")
        bucket = self.json_data.get("bucket")
        key = self.json_data.get("key")
        expires = str(self.json_data.get("expires"))

        content = f"{method}\n\n\n{expires}\n"
        content += f"/{bucket}/{key}"

        return self.__signature(content)

    def __signature(self, content: str) -> str:
        hmac_res = hmac.new(PRIVATE_KEY.encode(), content.encode(), sha1).digest()
        return base64.standard_b64encode(hmac_res).decode("utf-8")


class UCloudFileDownloader:
    def __init__(self):
        """
        初始化UCloud文件下载器

        参数:
            public_key: UCloud账户公钥
            private_key: UCloud账户私钥
            bucket: 存储空间名称
            upload_suffix: 上传host后缀
            download_suffix: 下载host后缀
        """
        self.public_key = UCloudConfig.UCloud_file_public_key
        self.private_key = UCloudConfig.UCloud_file_private_key
        self.bucket = UCloudConfig.UCloud_file_bucket

        # 设置UCloud配置
        config.set_default(uploadsuffix=UCloudConfig.UCloud_file_upload_suffix)
        config.set_default(downloadsuffix=UCloudConfig.UCloud_file_download_suffix)

        # 初始化文件管理器
        self.file_manager = filemanager.FileManager(
            UCloudConfig.UCloud_file_public_key,
            UCloudConfig.UCloud_file_private_key
        )

    def download_file(self, put_key, save_file):
        """
        从UCloud下载文件

        参数:
            put_key: 存储在UCloud中的文件名称
            save_file: 本地保存的文件名称

        返回:
            下载成功返回True，失败返回False
        """
        try:
            # 下载文件
            _, resp = self.file_manager.download_file(
                self.bucket,
                put_key,
                save_file
            )
            # 检查响应状态码
            if resp.status_code == 200:
                print(f"文件下载成功: {put_key} -> {save_file}")
                return True
            else:
                print(f"文件下载失败，状态码: {resp.status_code}")
                return False
        except Exception as e:
            print(f"下载文件时发生错误: {str(e)}")
            return False

    def upload_file(self, put_key, file_path):
        """
        上传文件到UCloud
        """
        try:
            # 上传文件
            _, resp = self.file_manager.putfile(
                self.bucket,
                put_key,
                file_path
            )
            if resp.status_code == 200:
                print(f"文件上传成功: {put_key} -> {file_path}->{resp.etag}")
                print(resp.etag)
                return resp.etag
            else:
                print(f"文件上传失败，状态码: {resp.status_code}")
                return False
        except Exception as e:
            print(f"上传文件时发生错误: {str(e)}")
            return False

#
# if __name__ == "__main__":
#     # 创建文件下载器
#     downloader = UCloudFileDownloader()
#     # 文件信息
#     put_key = "7/merged_segments/merged_7_00000.mp3"  # 存储在UCloud中的文件名称
#     save_file = "merged_7_00000.mp3"  # 本地保存的文件名称
#
#     # 执行下载
#     download_result = downloader.upload_file(put_key, save_file)
