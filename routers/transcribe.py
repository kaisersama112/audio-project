"""
@Project ：pythonProject 
@File    ：transcribe.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51 
"""
import glob
import asyncio
import subprocess
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Query, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydub import AudioSegment
from io import BytesIO
import os
import uuid
import shutil
import json
import re
import zipfile
from urllib.parse import quote

from models.schemas import TranscribeResponse, Segment, TaskStatusResponse, PaginatedSegments, ChunkUploadResponse
from services.audio_service import audio_service
from utils.file_utils import cleanup_task
from utils.mysql_db import init_db, get_db_connection, get_task, create_task, get_task_results, update_task_status

TEMP_DIR = "temp_audio_files"
router = APIRouter(tags=["音频切块"])

# 全局任务状态存储及锁
tasks = {}
tasks_lock = asyncio.Lock()


def convert_to_wav(input_path: str, output_path: str):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"格式转换失败: {str(e)}")


async def process_audio_task(task_id: str, original_path: str, original_ext: str):
    task_dir = os.path.join(TEMP_DIR, task_id)
    try:
        update_task_status({
            "task_id": task_id,
            "status": "processing",
            "message": "开始处理音频文件",
            "progress": 0
        })

        if original_ext.lower() != '.wav':
            update_task_status({
                "task_id": task_id,
                "status": "processing",
                "message": "正在转换音频格式",
                "progress": 30
            })

            wav_path = os.path.join(task_dir, "audio.wav")
            await asyncio.to_thread(convert_to_wav, original_path, wav_path)
            processing_path = wav_path
        else:
            processing_path = original_path

        # 语音识别阶段

        update_task_status({
            "task_id": task_id,
            "status": "processing",
            "message": "开始语音识别",
            "progress": 40
        })
        segments = await asyncio.to_thread(audio_service.transcribe_para_former, processing_path, task_id)

        # 结果保存阶段
        update_task_status({
            "task_id": task_id,
            "status": "processing",
            "message": "保存识别结果",
            "progress": 70
        })

        # 将结果保存到数据库
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                for segment in segments:
                    cursor.execute('''
                        INSERT INTO ai_task_results (task_id, `index`, start, `end`, text, speaker,`url`)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        task_id,
                        segment.get("index"),
                        segment.get("start"),
                        segment.get("end"),
                        segment.get("text"),
                        segment.get("speaker"),
                        segment.get("url")
                    ))
                conn.commit()

        update_task_status({
            "task_id": task_id,
            "status": "completed",
            "message": "处理完成",
            "progress": 100
        })

    except Exception as e:
        update_task_status({
            "task_id": task_id,
            "status": "failed",
            "message": "处理过程中发生错误",
            "progress": 100,
            "error": str(e)
        })

        shutil.rmtree(task_dir, ignore_errors=True)


async def validate_task(task_id: str):
    async with tasks_lock:
        task_data = tasks.get(task_id)
        if not task_data:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task_data["status"] != "completed":
            raise HTTPException(status_code=425, detail="任务尚未完成")
        return task_data


@router.on_event("startup")
async def startup():
    init_db()


# 在merge_chunks接口中修改合并逻辑
def merge_with_ffmpeg(task_dir: str, output_path: str):
    """使用FFmpeg合并分片文件"""
    # 生成分片列表文件
    concat_list = os.path.join(task_dir, "concat_list.txt")
    with open(concat_list, "w") as f:
        for chunk in sorted(glob.glob(os.path.join(task_dir, "chunk_*.part"))):
            f.write(f"file '{os.path.basename(chunk)}'\n")

    # 使用FFmpeg合并
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        "-map_metadata", "0",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg合并失败: {e.stderr.decode()}"
        raise RuntimeError(error_msg)
    finally:
        os.remove(concat_list)


def validate_audio_file(file_path: str):
    """使用FFprobe验证文件有效性"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
        duration = float(result.stdout.decode().strip())
        if duration <= 0:
            raise ValueError("无效的音频时长")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"文件验证失败: {e.stderr.decode()}"
        raise RuntimeError(error_msg)


# 分片上传接口
@router.post("/upload_chunk", response_model=ChunkUploadResponse)
async def upload_chunk(
        file: UploadFile = File(...),
        chunk_number: int = Form(...),
        total_chunks: int = Form(...),
        file_name: str = Form(...),  # 原始文件名
        task_id: Optional[str] = Form(None),
):
    """上传文件分片"""
    # 生成或验证任务ID
    task_id = task_id
    if not task_id:
        raise HTTPException(400, "任务ID不能为空")
    task_dir = os.path.join(TEMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 验证分片序号有效性
    if chunk_number < 0 or total_chunks <= 0 or chunk_number >= total_chunks:
        raise HTTPException(400, "分片参数不合法")

    # 保存元数据（第一个分片时）
    if chunk_number == 0:
        metadata = {
            "file_name": file_name,
            "total_chunks": total_chunks,
            "uploaded_chunks": 0
        }
        with open(os.path.join(task_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    # 保存分片文件
    chunk_path = os.path.join(task_dir, f"chunk_{chunk_number:04d}.part")
    try:
        # 流式写入（每次1MB）
        with open(chunk_path, "wb") as f:
            while content := await file.read(1024 * 1024):
                f.write(content)
    except Exception as e:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(500, f"分片保存失败: {str(e)}")

    # 更新已上传分片计数
    uploaded = len([f for f in os.listdir(task_dir) if f.startswith("chunk_")])
    return {
        "task_id": task_id,
        "chunk_number": chunk_number,
        "uploaded_chunks": uploaded,
        "total_chunks": total_chunks,
        "status": "partial" if uploaded < total_chunks else "complete"
    }


# 合并分片接口
@router.post("/merge_chunks", response_model=TranscribeResponse)
async def merge_chunks(
        task_id: str = Form(...),
        background_tasks: BackgroundTasks = None
):
    task_dir = os.path.join(TEMP_DIR, task_id)
    if not os.path.exists(task_dir):
        raise HTTPException(404, "任务不存在")

    try:
        # 加载元数据
        with open(os.path.join(task_dir, "metadata.json")) as f:
            metadata = json.load(f)

        # 验证分片完整性
        chunk_files = glob.glob(os.path.join(task_dir, "chunk_*.part"))
        if len(chunk_files) != metadata["total_chunks"]:
            raise HTTPException(400, "分片数量不匹配")

        # 合并文件（使用FFmpeg）
        original_ext = os.path.splitext(metadata["file_name"])[1]
        original_path = os.path.join(task_dir, f"merged{original_ext or '.mp4'}")
        merge_with_ffmpeg(task_dir, original_path)

        # 格式验证
        if not validate_audio_file(original_path):
            raise HTTPException(400, "合并文件格式异常")

        # 创建数据库记录
        with get_db_connection() as conn:
            create_task(conn, {
                "task_id": task_id,
                "status": "pending",
                "message": "文件合并完成，等待处理",
                "progress": 20,
                "original_path": original_path,
                "created_at": datetime.now().isoformat(),
                "start_time": None
            })
            conn.commit()

        # 添加后台处理任务
        background_tasks.add_task(process_audio_task, task_id, original_path, original_ext)
        background_tasks.add_task(cleanup_task, task_id)

        return JSONResponse({
            "task_id": task_id,
            "status": "pending",
            "message": "任务已开始处理"
        })

    except Exception as e:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(500, f"文件处理失败: {str(e)}")


def load_segments_if_completed(conn, task_id):
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM ai_task_results WHERE task_id = %s ORDER BY `index`", (task_id,))
            results = cursor.fetchall()
        return [Segment(**val) for val in results]
    except Exception as e:
        return f"Error loading segments: {str(e)}"


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse, summary="获取任务状态")
async def get_task_status(task_id: str):
    with get_db_connection() as conn:
        task = get_task(conn, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        segments = load_segments_if_completed(conn, task_id) if task['status'] == "completed" else None
        print(segments)
        return TaskStatusResponse(
            task_id=task['task_id'],
            status=task['status'],
            message=task['message'],
            progress=task['progress'],
            start_time=task['start_time'],
            complete_time=str(task['complete_time']),
            duration=task['duration'],
            error=task['error'],
            data=segments
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
    with get_db_connection() as conn:
        task = get_task(conn, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task['status'] != "completed":
            raise HTTPException(status_code=425, detail="任务尚未完成")
        # 查询结果数据
        results = get_task_results(conn, task_id, keyword, speaker, page, per_page)
    # 构建响应
    return PaginatedSegments(
        items=results["items"],
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
    with get_db_connection() as conn:
        task = get_task(conn, task_id)
        if not task:
            raise HTTPException(404, "任务不存在")

        # 获取原始文件路径
        original_path = task["original_path"]  # 假设 original_path 是任务表中的第5个字段

        # 获取分段信息
        segments = get_task_results(conn, task_id)
        if segment_index < 0 or segment_index >= len(segments["items"]):
            raise HTTPException(400, "无效的片段索引")

        segment = segments["items"][segment_index]

    try:
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
    with get_db_connection() as conn:
        segments = get_task_results(conn, task_id)
    return segments


async def get_original_audio(task_id: str):
    with get_db_connection() as conn:
        task = get_task(conn, task_id)
        if not task:
            raise HTTPException(404, "任务不存在")
        original_path = task["original_path"]  # 假设 original_path 是任务表中的第5个字段
    try:
        return AudioSegment.from_file(original_path)
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
    with get_db_connection() as conn:
        segments = get_task_results(conn, task_id)["items"]
        task = get_task(conn, task_id)
        original_path = task["original_path"]  # 假设 original_path 是任务表中的第5个字段
    audio = await get_original_audio(task_id)
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


@router.delete("/segments/{task_id}", summary="批量删除指定分段")
async def delete_segments(
    task_id: str,
    indices: str = Query(..., description="逗号分隔的分段索引列表")
):
    """删除指定任务的多个分段数据"""
    try:
        # 将传入的字符串转换为整数列表
        segment_indices = [int(idx) for idx in indices.split(',') if idx.strip()]
    except ValueError:
        raise HTTPException(400, detail="索引格式错误，请使用逗号分隔的整数")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # 构造IN子句
                in_clause = ', '.join(['%s'] * len(segment_indices))
                # 删除对应分段的数据
                cursor.execute(
                    f"DELETE FROM ai_task_results WHERE task_id = %s AND `index` IN ({in_clause})",
                    (task_id,) + tuple(segment_indices)
                )
                conn.commit()

                # 检查是否有行被删除
                if cursor.rowcount == 0:
                    raise HTTPException(404, detail=f"任务 {task_id} 下指定的分段不存在")

                return {"message": f"任务 {task_id} 下索引为 {indices} 的分段已删除"}
    except Exception as e:
        raise HTTPException(500, detail=f"批量删除分段时出错: {str(e)}")

@router.get("/speakers/{task_id}", summary="获取发音人列表")
async def get_speakers(
    task_id: str
):
    """获取指定任务的所有发音人列表"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT speaker FROM ai_task_results WHERE task_id = %s", (task_id,))
                speakers = cursor.fetchall()
                return [speaker['speaker'] for speaker in speakers]
    except Exception as e:
        raise HTTPException(500, detail=f"获取发音人列表时出错: {str(e)}")