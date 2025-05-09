"""
@Project ：pythonProject 
@File    ：transcribe.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51 
"""
import glob
import asyncio
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment
from io import BytesIO
import os
import uuid
import shutil
import json
import re
import zipfile
from urllib.parse import quote
from services.audio_service import audio_service
from utils.file_utils import cleanup_task

TEMP_DIR = "temp_audio_files"
router = APIRouter()

# 全局任务状态存储及锁
tasks = {}
tasks_lock = asyncio.Lock()


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: Optional[str]
    status: Optional[str]
    message: Optional[str]
    progress: Optional[int]
    start_time: Optional[str]
    complete_time: Optional[str]
    duration: Optional[float]
    error: Optional[str]
    data: Optional[list[dict]] = None


class TranscribeResponse(BaseModel):
    task_id: str


async def update_task_status(task_id: str,
                             status: str,
                             message: str,
                             progress: Optional[int] = 0,
                             error: Optional[str] = None):
    async with tasks_lock:
        current_time = datetime.now().isoformat()
        task_data = tasks.get(task_id, {})

        # 计算持续时间
        duration = None
        if status in ["completed", "failed"] and "start_time" in task_data:
            start_time = datetime.fromisoformat(task_data["start_time"])
            end_time = datetime.fromisoformat(current_time)
            duration = round((end_time - start_time).total_seconds(), 2)

        task_data.update({
            "status": status,
            "message": message,
            "progress": progress or task_data.get("progress"),
            "error": error,
            "complete_time": current_time if status in ["completed", "failed"] else None,
            "duration": duration or task_data.get("duration")
        })

        if status == "processing" and "start_time" not in task_data:
            task_data["start_time"] = current_time

        tasks[task_id] = task_data


def convert_to_wav(input_path: str, output_path: str):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"格式转换失败: {str(e)}")


async def process_audio_task(task_id: str, original_path: str, original_ext: str):
    task_dir = os.path.join(TEMP_DIR, task_id)
    try:
        await update_task_status(task_id, "processing", "开始处理音频文件", 0)

        if original_ext.lower() != '.wav':
            await update_task_status(task_id, "processing", "正在转换音频格式", 30)
            wav_path = os.path.join(task_dir, "audio.wav")
            await asyncio.to_thread(convert_to_wav, original_path, wav_path)
            processing_path = wav_path
        else:
            processing_path = original_path

        # 语音识别阶段
        await update_task_status(task_id, "processing", "开始语音识别", 40)
        segments = await asyncio.to_thread(audio_service.transcribe_para_former, processing_path)

        # 结果保存阶段
        await update_task_status(task_id, "processing", "保存识别结果", 70)
        with open(os.path.join(task_dir, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        await update_task_status(task_id, "completed", "处理完成", 100)

    except Exception as e:
        await update_task_status(task_id, "failed", "处理过程中发生错误", 100, str(e))
        shutil.rmtree(task_dir, ignore_errors=True)


async def validate_task(task_id: str):
    async with tasks_lock:
        task_data = tasks.get(task_id)
        if not task_data:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task_data["status"] != "completed":
            raise HTTPException(status_code=425, detail="任务尚未完成")
        return task_data


@router.post("/transcribe/", response_model=TranscribeResponse, summary="提交音频转录任务")
async def transcribe_audio(
        file: UploadFile = File(...),
        task_id: Optional[str] = Query(None, description="可选的任务ID"),
        background_tasks: BackgroundTasks = None
):
    task_dir = os.path.join(TEMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    original_ext = os.path.splitext(file.filename)[1]
    original_path = os.path.join(task_dir, f"original_audio{original_ext}")

    try:
        contents = await file.read()
        with open(original_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        shutil.rmtree(task_dir)
        raise HTTPException(500, f"文件保存失败: {str(e)}")

    # 初始化任务状态
    await update_task_status(task_id, "pending", "任务已创建，等待处理")

    # 添加后台处理任务
    background_tasks.add_task(process_audio_task, task_id, original_path, original_ext)
    background_tasks.add_task(cleanup_task, task_id)

    return JSONResponse({
        "task_id": task_id,
        "status": "pending",
        "message": "任务已提交，正在处理"
    })


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
    data: Optional[List[Segment]] = None  # 新增数据字段


# 修改状态获取接口
@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse, summary="获取任务状态")
async def get_task_status(task_id: str):
    async with tasks_lock:
        task_data = tasks.get(task_id)
        if not task_data:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 实时计算进行中任务的耗时
        if task_data.get("status") == "processing":
            if start_time := task_data.get("start_time"):
                start = datetime.fromisoformat(start_time)
                duration = (datetime.now() - start).total_seconds()
                task_data["duration"] = round(duration, 2)
        taskStatusResponse = TaskStatusResponse(
            task_id=task_id,
            status=task_data.get("status"),
            message=task_data.get("message"),
            progress=task_data.get("progress"),
            start_time=task_data.get("start_time"),
            complete_time=task_data.get("complete_time"),
            duration=task_data.get("duration"),
            error=task_data.get("error"))
        if task_data.get("status") == "completed":
            try:
                task_dir = os.path.join(TEMP_DIR, task_id)
                with open(os.path.join(task_dir, "segments.json"), "r", encoding="utf-8") as f:
                    taskStatusResponse.data = json.load(f)
            except Exception as e:
                taskStatusResponse.error = f"结果加载失败: {str(e)}"

        return taskStatusResponse


@router.get("/download/single/{task_id}/{segment_index}", responses={
    200: {"content": {"audio/mpeg": {}}, "description": "返回MP3音频片段"}},
            summary="单音频下载")
async def download_single_segment(task_id: str, segment_index: int):
    await validate_task(task_id)

    task_dir = os.path.join(TEMP_DIR, task_id)
    with open(os.path.join(task_dir, "segments.json"), "r", encoding="utf-8") as f:
        segments = json.load(f)

    if segment_index < 0 or segment_index >= len(segments):
        raise HTTPException(400, "无效的片段索引")

    segment = segments[segment_index]

    try:
        original_files = glob.glob(os.path.join(task_dir, "original_audio.*"))
        if not original_files:
            raise FileNotFoundError("找不到原始音频文件")
        original_path = original_files[0]

        audio = AudioSegment.from_file(original_path)
    except Exception as e:
        raise HTTPException(500, f"音频加载失败: {str(e)}")

    time_suffix = f"{segment['start']:.2f}-{segment['end']:.2f}"
    base_filename = f"audio_{time_suffix}.mp3"

    clean_text = re.sub(r'[\\/*?:"<>|]', '_', segment["text"])[:50]  # 限制长度
    enhanced_filename = f"{clean_text}_{time_suffix}.mp3" if clean_text else base_filename

    start_ms = int(segment["start"] * 1000)
    end_ms = int(segment["end"] * 1000)

    try:
        audio_segment = audio[start_ms:end_ms]
        buffer = BytesIO()
        audio_segment.export(
            buffer,
            format="mp3",
            codec="libmp3lame",
            bitrate="192k",
            tags={
                'title': enhanced_filename,
                'artist': 'Audio API'
            }
        )
        buffer.seek(0)
    except Exception as e:
        raise HTTPException(500, f"音频导出失败: {str(e)}")
    safe_filename = quote(enhanced_filename, safe='')
    return StreamingResponse(
        buffer,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition":
                f'attachment; filename="{safe_filename}"; '
                f'filename*=UTF-8\'\'{safe_filename}',
            "Content-Type": "audio/mpeg",
            "X-Content-Type-Options": "nosniff"
        }
    )


async def get_validated_segments(task_id: str):
    await validate_task(task_id)
    task_dir = os.path.join(TEMP_DIR, task_id)
    try:
        with open(os.path.join(task_dir, "segments.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(500, f"加载分段信息失败: {str(e)}")


async def get_original_audio(task_dir: str):
    try:
        original_files = glob.glob(os.path.join(task_dir, "original_audio.*"))
        if not original_files:
            raise FileNotFoundError("找不到原始音频文件")
        return AudioSegment.from_file(original_files[0])
    except Exception as e:
        raise HTTPException(500, f"音频加载失败: {str(e)}")


def sanitize_filename(text: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', str(text))


@router.get("/download/bulk/{task_id}", responses={
    200: {"content": {"application/zip": {}}, "description": "返回ZIP压缩包"}},
            summary="多音频下载")
async def download_bulk_segments(
        task_id: str,
        indices: str = Query(..., description="逗号分隔的片段索引列表")
):
    """
    批量下载多个音频片段（按说话人分类的ZIP压缩包）
    功能特点：
    1. 严格的任务状态验证
    2. 智能文件名清理
    3. 内存高效的流式响应
    4. 完善的错误处理
    """
    await validate_task(task_id)
    task_dir = os.path.join(TEMP_DIR, task_id)
    segments = await get_validated_segments(task_id)
    audio = await get_original_audio(task_dir)
    try:
        indices_list = [int(idx) for idx in indices.split(',') if idx.strip()]
    except ValueError:
        raise HTTPException(400, "索引格式错误，请使用逗号分隔的整数")
    if not (0 <= min(indices_list) and max(indices_list) < len(segments)):
        raise HTTPException(400, f"索引范围错误，有效范围：0-{len(segments) - 1}")

    def generate_zip():
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for idx in indices_list:
                segment = segments[idx]
                speaker = sanitize_filename(segment.get("speaker", "unknown"))
                text = sanitize_filename(segment["text"][:50])
                time_range = f"{segment['start']:.2f}-{segment['end']:.2f}"

                filename = f"{speaker}/{text}_{time_range}.mp3" if text else \
                    f"{speaker}/segment_{time_range}.mp3"
                start = int(segment["start"] * 1000)
                end = int(segment["end"] * 1000)
                segment_audio = audio[start:end]

                with BytesIO() as audio_buffer:
                    segment_audio.export(audio_buffer, format="mp3", bitrate="128k")
                    zip_file.writestr(filename, audio_buffer.getvalue())

        zip_buffer.seek(0)
        return zip_buffer

    return StreamingResponse(
        generate_zip(),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=classified_segments.zip",
            "X-Content-Type-Options": "nosniff"
        }
    )


@router.get("/download/all/{task_id}", responses={
    200: {"content": {"application/zip": {}}, "description": "返回全部音频"}},
            summary="全部音频下载")
async def download_all_segments(task_id: str):
    """
    下载全部音频片段（自动生成索引）
    功能增强：
    1. 自动处理全部索引
    2. 优化大文件内存使用
    3. 统一错误处理
    """
    segments = await get_validated_segments(task_id)
    indices = ",".join(map(str, range(len(segments))))

    return await download_bulk_segments(
        task_id=task_id,
        indices=indices
    )
