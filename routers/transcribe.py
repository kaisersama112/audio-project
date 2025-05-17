"""
@Project ：pythonProject 
@File    ：transcribe.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51 
"""
import glob
import asyncio
import io
import subprocess
from datetime import datetime
from typing import Optional, List

import httpx
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
import concurrent.futures
from models.schemas import TranscribeResponse, Segment, TaskStatusResponse, PaginatedSegments, ChunkUploadResponse
from services.audio_service import audio_service
from utils.file_utils import cleanup_task
from utils.mysql_db import  get_db_connection, get_task, create_task, get_task_results, update_task_status
import threading
TEMP_DIR = "temp_audio_files"
# 互斥锁
text_recognition_lock =  asyncio.Lock()
router = APIRouter(tags=["音频切块"])

def convert_to_wav(input_path: str, output_path: str):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"格式转换失败: {str(e)}")

async def process_audio_task(task_id: str, original_path: str, original_ext: str,min_chunk_duration:float):

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
        # 使用互斥锁确保文本识别阶段串行执行
        async with text_recognition_lock:
            # 文本识别
            update_task_status({
                "task_id": task_id,
                "status": "processing",
                "message": "正在进行文本识别",
                "progress": 50
            })
            result =await asyncio.to_thread(audio_service.transcribe_para_former, processing_path)
        # 发音人合并
        update_task_status({
            "task_id": task_id,
            "status": "processing",
            "message": "合并发音人信息",
            "progress": 60
        })
        raw_segments = result[0]["sentence_info"]
        merged_segments=await asyncio.to_thread(audio_service.merge_segments, raw_segments,min_chunk_duration)

        # 格式化结果并上传
        update_task_status({
            "task_id": task_id,
            "status": "processing",
            "message": "格式化识别结果",
            "progress": 70
        })
        segments =await asyncio.to_thread(audio_service.formatted_results_upload,merged_segments, processing_path, task_id)
        update_task_status({
            "task_id": task_id,
            "status": "processing",
            "message": "保存识别结果到数据库",
            "progress": 80
        })
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
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)
def merge_with_ffmpeg(task_dir: str, output_path: str):
    """使用FFmpeg合并分片文件"""
    # 生成分片列表文件

    concat_list = os.path.join(task_dir, "concat_list.txt")
    if os.path.exists(concat_list):
        os.remove(concat_list)
        os.remove(os.path.join(task_dir, "merged.wav"))
    with open(concat_list, "w") as f:
        # 修改为查找 .wav 文件
        for chunk in sorted(glob.glob(os.path.join(task_dir, "chunk_*.wav"))):
            # 写入相对路径
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
    """上传文件分片，支持覆盖现有任务"""
    # 生成或验证任务ID
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

    # 保存并转换分片文件为.wav
    chunk_path = os.path.join(task_dir, f"chunk_{chunk_number:08d}.part")
    wav_chunk_path = os.path.join(task_dir, f"chunk_{chunk_number:08d}.wav")
    try:
        # 流式写入（每次1MB）
        with open(chunk_path, "wb") as f:
            while content := await file.read(1024 * 1024):
                f.write(content)

        # 转换为.wav
        await asyncio.to_thread(convert_to_wav, chunk_path, wav_chunk_path)

        # 删除原始.part文件
        os.remove(chunk_path)

    except Exception as e:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(500, f"分片保存或转换失败: {str(e)}")

    # 更新已上传分片计数
    uploaded = len([f for f in os.listdir(task_dir) if f.startswith("chunk_") and f.endswith(".wav")])
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
        min_chunk_duration: float = Form(1.0),  # 最小分片时长，单位秒
        background_tasks: BackgroundTasks = None
):
    print("task_id:",task_id)
    task_dir = os.path.join(TEMP_DIR, task_id)
    if not os.path.exists(task_dir):
        raise HTTPException(404, "任务不存在")

    try:
        with open(os.path.join(task_dir, "metadata.json")) as f:
            metadata = json.load(f)
        chunk_files = glob.glob(os.path.join(task_dir, "chunk_*.wav"))
        if len(chunk_files) != metadata["total_chunks"]:
            raise HTTPException(400, "分片数量不匹配")
        original_path = os.path.join(task_dir, "merged.wav")
        merge_with_ffmpeg(task_dir, original_path)
        if not validate_audio_file(original_path):
            raise HTTPException(400, "合并文件格式异常")
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
        background_tasks.add_task(process_audio_task, task_id, original_path, ".wav",min_chunk_duration)
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
        segments = get_task_results(conn, task_id)
        if segment_index < 0 or segment_index >= len(segments["items"]):
            raise HTTPException(400, "无效的片段索引")
        segment = segments["items"][segment_index]
        audio_url = segment.get("url")
        if not audio_url:
            raise HTTPException(404, "音频URL不存在")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(audio_url)
            if response.status_code != 200:
                raise HTTPException(500, f"无法下载音频片段: {response.status_code}")
            audio_content = response.content
        buffer = BytesIO(audio_content)
    except Exception as e:
        raise HTTPException(500, f"音频加载失败: {str(e)}")
    safe_filename = quote(segment["text"][:50] + f"_{segment['start']:.2f}-{segment['end']:.2f}.mp3", safe='')
    return StreamingResponse(
        buffer,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
        }
    )


@router.get("/download/bulk/{task_id}", responses={
    200: {"content": {"application/zip": {}}, "description": "返回ZIP压缩包"}},
    summary="多音频下载")
async def download_bulk_segments(task_id: str, indices: str = Query(..., description="逗号分隔的片段索引列表")):
    with get_db_connection() as conn:
        segments = get_task_results(conn, task_id)["items"]
        if not segments:
            raise HTTPException(404, "任务结果不存在")

    try:
        indices_list = [int(idx) for idx in indices.split(',') if idx.strip()]
    except ValueError:
        raise HTTPException(400, "索引格式错误，请使用逗号分隔的整数")

    if not all(0 <= idx < len(segments) for idx in indices_list):
        raise HTTPException(400, f"索引范围错误，有效范围：0-{len(segments) - 1}")

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        # 强制使用 UTF-8 编码写入文件名
        for idx in indices_list:
            segment = segments[idx]
            audio_url = segment.get("url")
            speaker = segment.get("speaker", "unknown")  # 获取说话人信息
            if not audio_url:
                continue

            async with httpx.AsyncClient() as client:
                response = await client.get(audio_url)
                if response.status_code != 200:
                    continue
                content = response.content
                filename_base = f"{segment['text'][:50]}_{segment['start']:.2f}-{segment['end']:.2f}.mp3"
                safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename_base)
                folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', speaker)  # 创建安全的文件夹名
                final_path = f"{folder_name}/{safe_filename}"
                zip_file.writestr(final_path, content)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{task_id}_segments_bulk_{datetime.now().strftime("%Y%m%d%H%M%S")}.zip"'
        }
    )

@router.get("/download/all/{task_id}", responses={
    200: {"content": {"application/zip": {}}, "description": "返回全部音频"}},
    summary="全部音频下载")
async def download_all_segments(task_id: str):
    with get_db_connection() as conn:
        segments = get_task_results(conn, task_id)["items"]
        if not segments:
            raise HTTPException(404, "任务结果不存在")

    indices = ",".join(str(i) for i in range(len(segments)))

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