"""
@Project ：pythonProject 
@File    ：transcribe.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51 
"""

from datetime import datetime
from typing import Optional
from urllib.parse import quote
from uuid import uuid4
import re
import os
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Form

import shutil

from sqlalchemy import select
from starlette.responses import FileResponse

from config import TEMP_DIR
from curd.async_crud import get_db_async, get_task_async, create_task_async, get_task_results_async, \
    get_segments_by_indices_async, get_all_segments_async, delete_segments_by_indices_async, \
    delete_segments_by_keywords_async
# from curd.crud import get_db, create_task, get_task, get_task_results, get_segments_by_indices, get_all_segments, \
#     delete_segments_by_keywords, delete_segments_by_indices
from curd.models import AITaskResult, AIDownloadTask
from models.schemas import TaskStatusResponse, PaginatedSegments, Segment
from services.audio_service import format_task_merged_filename, load_segments_if_completed, process_audio_task, \
    process_download_task
from utils.ucloud_u3d import UCloudFileDownloader

router = APIRouter(tags=["音频切块"])


# 启动音频处理任务接口
@router.post("/start_audio_processing", summary="启动音频处理任务")
async def start_audio_processing(
        task_id: str = Form(...),
        file_url: str = Form(...),
        min_chunk_duration: float = Form(3.0),  # 最小分片时长，单位秒
        separate: bool = Form(False),  # 人声背景分离
        background_tasks: BackgroundTasks = None
):
    print("task_id:", task_id)
    downloader = UCloudFileDownloader()
    task_dir = os.path.join(TEMP_DIR, task_id)
    async with get_db_async() as db:
        existing_task = await get_task_async(db, task_id)  # 使用之前定义的 get_task 函数
        if existing_task:
            shutil.rmtree(task_dir, ignore_errors=True)
            raise HTTPException(status_code=409, detail="任务已存在")
        # 创建任务目录
        try:
            os.makedirs(task_dir, exist_ok=True)
        except Exception as e:
            raise HTTPException(500, f"创建任务目录失败: {str(e)}")
        # 下载音频文件
        # 下载音频文件（使用UCloud下载器）
        try:
            # 确定下载文件的保存路径
            # 假设音频是 wav 或 mp3 格式 - 根据实际需求调整
            audio_filename = os.path.basename(file_url)
            audio_filename_lower = audio_filename.lower()
            if not audio_filename_lower.endswith(('.wav', '.mp3')):
                raise HTTPException(400, "不支持的音频格式，仅支持 .wav 或 .mp3")

            download_path = os.path.join(task_dir, audio_filename)
            print(f"下载路径: {download_path}")
            # 使用UCloud下载器下载文件
            download_success = downloader.download_file(file_url, download_path)

            if not download_success:
                shutil.rmtree(task_dir, ignore_errors=True)
                raise HTTPException(500, "从UCloud下载音频文件失败")

            print(f"音频文件已下载到: {download_path}")
        except Exception as e:
            shutil.rmtree(task_dir, ignore_errors=True)
            raise HTTPException(500, f"下载音频文件失败: {str(e)}")
        try:
            # 创建任务记录
            await create_task_async(db, {
                "task_id": task_id,
                "status": "pending",
                "message": "音频文件已下载，等待处理",
                "progress": 20,
                "original_path": download_path,
                "created_at": datetime.now().isoformat(),
                "start_time": None
            })
            db.commit()
            # 添加后台任务进行音频处理
            background_tasks.add_task(process_audio_task, task_id, download_path, audio_filename, min_chunk_duration,
                                      separate)

            return {
                "task_id": task_id,
                "status": "pending",
                "message": "任务已开始处理"
            }

        except Exception as e:
            shutil.rmtree(task_dir, ignore_errors=True)
            raise HTTPException(500, f"任务启动失败: {str(e)}")


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse, summary="获取任务状态")
async def get_task_status(task_id: str):
    async with get_db_async() as conn:
        task = await get_task_async(conn, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        segments = None
        if task.status == "completed":  # 使用属性访问
            segments = await load_segments_if_completed(conn, task_id)

        print(segments)
        return TaskStatusResponse(
            task_id=task.task_id,  # 使用属性访问
            status=task.status,  # 使用属性访问
            message=task.message,  # 使用属性访问
            progress=task.progress,  # 使用属性访问
            start_time=task.start_time,
            complete_time=str(task.complete_time),
            duration=task.duration,
            error=task.error,
            is_upload=task.is_upload,
            data=segments,
        )


@router.get("/tasks/{task_id}/segments",
            response_model=PaginatedSegments,
            summary="获取任务结果",
            responses={
                200: {"description": "成功返回语音分段结果"},
                404: {"description": "任务不存在"},
                425: {"description": "任务未完成"},
                500: {"description": "结果文件读取失败"}
            })
async def get_task_segments(
        task_id: str,
        keyword: Optional[str] = Query(None,
                                       min_length=1,
                                       description="模糊搜索关键字（匹配文本内容、类型等字段）"),
        speaker: Optional[str] = Query(None,
                                       description="筛选特定说话人（支持模糊匹配）"),
        page: int = Query(1, ge=1),
        per_page: int = Query(10, ge=1, le=100)
):
    """获取任务分段数据（增强版）"""

    # 1. 验证任务状态
    async with get_db_async() as conn:
        task = await get_task_async(conn, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.status != "completed":
            raise HTTPException(status_code=425, detail="任务尚未完成")
        # 查询结果数据
        results = await get_task_results_async(conn, task_id, keyword, speaker, page, per_page)
        # 将 AITaskResult 对象转换为 Segment 对象
    segment_items = []
    for result in results["items"]:
        segment = Segment(
            index=result.index,
            start=result.start,
            end=result.end,
            url=result.url,
            text=result.text,
            speaker=result.speaker,
            suffix=result.suffix if hasattr(result, "suffix") else None
        )
        segment_items.append(segment)

    # 构建响应
    return PaginatedSegments(
        items=segment_items,
        total=results["total"],
        page=results["page"],
        per_page=results["per_page"],
        total_pages=results["total_pages"],
        search_hits=results["total"]
    )


@router.get("/download/single/{task_id}/{segment_index}", responses={
    200: {"content": {"audio/mpeg": {}}, "description": "返回MP3音频片段"}},
            summary="单音频下载")
async def download_single_segment(task_id: str, segment_index: int):
    async with get_db_async() as conn_task:
        task = await get_task_async(conn_task, task_id)
        if not task:
            raise HTTPException(404, "任务不存在")
    # 本地文件下载逻辑
    base_dir = os.path.join(TEMP_DIR, task_id)
    segments_dir = os.path.join(base_dir, "merged_segments")
    if not os.path.exists(segments_dir):
        raise HTTPException(404, "音频文件夹不存在")
    filename = format_task_merged_filename(task_id, segment_index)
    file_path = os.path.join(segments_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "音频文件不存在")
    # 从数据库获取当前片段的文本、开始时间和结束时间信息

    async with get_db_async() as conn_status:
        stmt = select(AITaskResult).where(
            AITaskResult.task_id == task_id,
            AITaskResult.index == segment_index
        ).limit(1)

        result = (await conn_status.execute(stmt)).scalars().first()

        if not result:
            raise HTTPException(404, "片段信息不存在")
        text = result.text
    filename_base = f"{text[:50]}.mp3"
    safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename_base)
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=safe_filename,
        headers={
            "Content-Disposition": f"attachment; filename*=utf-8''{quote(safe_filename)}",
        }
    )


@router.get("/download/bulk/{task_id}", responses={
    200: {"content": {"application/zip": {}}, "description": "返回ZIP压缩包"}},
            summary="多音频下载")
async def download_bulk_segments(
        task_id: str,
        indices: str = Query(..., description="逗号分隔的原始索引列表"),
        background_tasks: BackgroundTasks = None
):
    async with get_db_async() as session:
        async with session.begin():  # 开启事务
            task = await get_task_async(session, task_id)
            if not task:
                raise HTTPException(404, "任务不存在")

            try:
                # index_list = [int(i.strip()) for i in indices.split(",") if i.strip()]
                index_list = re.split(r'[;,；，、]', indices)

            except ValueError:
                raise HTTPException(400, "索引格式错误")

            segments = await get_segments_by_indices_async(session, task_id, index_list)
            if not segments:
                raise HTTPException(404, "指定的音频片段不存在")

            download_task_id = str(uuid4())

            db_download_task = AIDownloadTask(
                task_id=download_task_id,
                original_task_id=task_id,
                status='processing',
                progress=0
            )
            session.add(db_download_task)

        # 背景任务不能放在事务中执行，所以这里离开事务后再添加
        background_tasks.add_task(process_download_task, download_task_id, task_id, index_list, "bulk")

    return {
        "download_task_id": download_task_id,
        "status": "processing"
    }


@router.get("/download/all/{task_id}", responses={
    200: {"content": {"application/zip": {}}, "description": "返回全部音频"}},
            summary="全部音频下载")
async def download_all_segments(
        task_id: str,
        background_tasks: BackgroundTasks = None
):
    async with get_db_async() as session:
        async with session.begin():
            task = await get_task_async(session, task_id)
            if not task:
                raise HTTPException(404, "任务不存在")

            all_segments = await get_all_segments_async(session, task_id)
            if not all_segments:
                raise HTTPException(404, "没有可下载的音频片段")

            download_task_id = str(uuid4())

            db_download_task = AIDownloadTask(
                task_id=download_task_id,
                original_task_id=task_id,
                status='processing',
                progress=0
            )
            session.add(db_download_task)

        background_tasks.add_task(process_download_task, download_task_id, task_id, None, "all")

    return {
        "download_task_id": download_task_id,
        "status": "processing"
    }


@router.get("/download/status/{download_task_id}", summary="查询下载任务状态")
async def get_download_task_status(download_task_id: str):
    async with get_db_async() as db:
        stmt = select(AIDownloadTask).where(AIDownloadTask.task_id == download_task_id)
        result = await db.execute(stmt)
        download_task = result.scalars().first()

        if not download_task:
            return {"status": "not_found"}

        return {
            "download_task_id": download_task.task_id,
            "status": download_task.status,
            "progress": download_task.progress,
            "file_url": download_task.file_url
        }


@router.post("/segments/{task_id}", summary="批量删除指定分段")
async def delete_segments(
        task_id: str,
        indices: str = Query(..., description="逗号分隔的分段索引列表")
):
    """删除指定任务的多个分段数据"""
    try:
        segment_indices = re.split(r'[;,；，、]', indices)

        # 将传入的字符串转换为整数列表
        # segment_indices = [int(idx) for idx in indices.split(',') if idx.strip()]
    except ValueError:
        raise HTTPException(400, detail="索引格式错误，请使用逗号分隔的整数")

    base_dir = os.path.join(TEMP_DIR, task_id)
    segments_dir = os.path.join(base_dir, "merged_segments")

    if not os.path.exists(segments_dir):
        raise HTTPException(404, detail="音频文件夹不存在")

    async with get_db_async() as conn:
        # 调用通用删除函数
        deleted_indices = await delete_segments_by_indices_async(conn, task_id, segment_indices)
        if not deleted_indices:
            raise HTTPException(404, detail=f"任务 {task_id} 下指定的分段不存在")

        # 删除本地文件
        for index in deleted_indices:
            file_name = format_task_merged_filename(task_id, index)
            file_path = os.path.join(segments_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"文件不存在: {file_path}")

        return {"message": f"任务 {task_id} 下索引为 {indices} 的分段已删除"}


@router.post("/segments_keyword/{task_id}", summary="删除包含任意关键词的分段")
async def delete_segments_keyword(
        task_id: str,
        keyword: str = Query(..., description="关键词，可用逗号、分号分隔多个")
):
    """删除指定任务中包含任意关键词的分段数据"""
    async with get_db_async() as conn:
        # 检查任务是否存在及状态
        task = await get_task_async(conn, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.status != "completed":  # 使用属性访问
            raise HTTPException(status_code=425, detail="任务尚未完成")

        # 解析关键词
        keywords = re.split(r'[;,；，、]', keyword)
        keywords = [f"%{k.strip()}%" for k in keywords if k.strip()]
        if not keywords:
            raise HTTPException(status_code=400, detail="关键词不能为空")

        # 调用数据库操作函数，获取被删除的分段索引
        deleted_indices = await delete_segments_by_keywords_async(conn, task_id, keywords)
        if not deleted_indices:
            raise HTTPException(status_code=404, detail="未找到匹配的分段")

        # 删除本地文件
        base_dir = os.path.join(TEMP_DIR, task_id)
        segments_dir = os.path.join(base_dir, "merged_segments")
        if os.path.exists(segments_dir):
            for index in deleted_indices:
                file_path = os.path.join(
                    segments_dir,
                    format_task_merged_filename(task_id, index)
                )
                if os.path.exists(file_path):
                    os.remove(file_path)
                else:
                    print(f"文件不存在: {file_path}")
        else:
            print("音频文件夹不存在")

        return {
            "code": 200,
            "message": f"任务 {task_id} 中包含任意关键词的分段已删除",
            "deleted_count": len(deleted_indices)
        }


@router.get("/speakers/{task_id}", summary="获取发音人列表")
async def get_speakers(
        task_id: str
):
    """获取指定任务的所有发音人列表"""
    try:
        async with get_db_async() as conn:
            # 使用新的异步查询方式
            stmt = select(AITaskResult.speaker).where(
                AITaskResult.task_id == task_id
            ).distinct()

            result = await conn.execute(stmt)
            speakers = result.scalars().all()

            return list(speakers)

    except Exception as e:
        raise HTTPException(500, detail=f"获取发音人列表时出错: {str(e)}")
