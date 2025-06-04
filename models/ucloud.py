from pydantic import BaseModel, Field
from typing import Optional, List


class AuthRequest(BaseModel):
    method: str = Field(..., description="请求方法")
    bucket: str = Field(..., description="存储桶名称")
    content_type: str = Field("", description="内容类型")
    content_md5: str = Field("", description="内容的 MD5 值")
    date: str = Field("", description="日期")
    key: str = Field("", description="对象的键")


class PrivateUrlAuthRequest(BaseModel):
    method: str = Field(..., description="请求方法")
    bucket: str = Field(..., description="存储桶名称")
    key: str = Field(..., description="对象的键")
    expires: int = Field(..., description="过期时间戳")
