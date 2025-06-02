"""
@Project ：src 
@File    ：schemas.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:55 
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class Segment(BaseModel):
    index: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    url: Optional[str] = None
    text: Optional[str] = None
    speaker: Optional[str] = None
    suffix: Optional[str] = None


class PaginatedSegments(BaseModel):
    items: List[Segment]
    total: int
    page: int
    per_page: int
    total_pages: int
    search_hits: int = Field(..., description="包含关键字的记录总数")


# 修改状态响应模型
class TaskStatusResponse(BaseModel):
    task_id: Optional[str]
    status: Optional[str]
    message: Optional[str]
    progress: Optional[int]
    start_time: Optional[str]
    complete_time: Optional[str]
    duration: Optional[float]
    error: Optional[str]
    is_upload: Optional[int] = 0
    # 详细信息
    data: Optional[List[Segment]] = None




# 分片上传响应模型
class ChunkUploadResponse(BaseModel):
    task_id: str
    chunk_number: int
    uploaded_chunks: int
    total_chunks: int
    status: str  # partial/complete
