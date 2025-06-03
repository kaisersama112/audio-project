from ufile import config, filemanager

from config import UCloudConfig


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

# if __name__ == "__main__":
#     # 创建文件下载器
#     downloader = UCloudFileDownloader()
#     # 文件信息
#     put_key = "5月21日.MP3"  # 存储在UCloud中的文件名称
#     save_file = "5月21日.MP3"  # 本地保存的文件名称
#
#     # 执行下载
#     download_result = downloader.download_file(put_key, save_file)
