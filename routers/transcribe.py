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
from typing import Optional
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Query, Form
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
import os
import shutil
import json
import re
import zipfile

from starlette.responses import FileResponse

from config import TEMP_DIR
from models.schemas import TranscribeResponse, TaskStatusResponse, PaginatedSegments, ChunkUploadResponse
from services.audio_service import format_task_merged_filename, extract_index_from_filename, \
    load_segments_if_completed, convert_to_wav, merge_with_ffmpeg, validate_audio_file, process_audio_task
from utils.file_utils import cleanup_task
from utils.mysql_db import get_db_connection, get_task, create_task, get_task_results, update_task_status, \
    get_all_task_results

router = APIRouter(tags=["音频切块"])


@router.post("/upload_chunk", response_model=ChunkUploadResponse, summary="上传文件分片")
async def upload_chunk(
        file: UploadFile = File(...),
        chunk_number: int = Form(...),
        total_chunks: int = Form(...),
        file_name: str = Form(...),  # 原始文件名
        task_id: Optional[str] = Form(None),
):
    """上传文件分片，支持覆盖现有任务"""
    if not task_id:
        raise HTTPException(400, "任务ID不能为空")

    task_dir = os.path.join(TEMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    if chunk_number < 0 or total_chunks <= 0 or chunk_number >= total_chunks:
        raise HTTPException(400, "分片参数不合法")
    if chunk_number == 0:
        metadata = {
            "file_name": file_name,
            "total_chunks": total_chunks,
            "uploaded_chunks": 0
        }
        with open(os.path.join(task_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    chunk_path = os.path.join(task_dir, f"chunk_{chunk_number:08d}.part")
    wav_chunk_path = os.path.join(task_dir, f"chunk_{chunk_number:08d}.wav")
    try:
        with open(chunk_path, "wb") as f:
            while content := await file.read(1024 * 1024 * 5):
                f.write(content)
        await asyncio.to_thread(convert_to_wav, chunk_path, wav_chunk_path)
        os.remove(chunk_path)
    except Exception as e:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(500, f"分片保存或转换失败: {str(e)}")
    uploaded = len([f for f in os.listdir(task_dir) if f.startswith("chunk_") and f.endswith(".wav")])
    return {
        "task_id": task_id,
        "chunk_number": chunk_number,
        "uploaded_chunks": uploaded,
        "total_chunks": total_chunks,
        "status": "partial" if uploaded < total_chunks else "complete"
    }


@router.post("/merge_chunks", response_model=TranscribeResponse, summary="合并分片并开始处理")
async def merge_chunks(
        task_id: str = Form(...),
        min_chunk_duration: float = Form(3.0),  # 最小分片时长，单位秒
        background_tasks: BackgroundTasks = None
):
    if not min_chunk_duration or min_chunk_duration == "":
        min_chunk_duration = 3.0
    print("task_id:", task_id)
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
        background_tasks.add_task(process_audio_task, task_id, original_path, ".wav", min_chunk_duration)
        background_tasks.add_task(cleanup_task, task_id)

        return JSONResponse({
            "task_id": task_id,
            "status": "pending",
            "message": "任务已开始处理"
        })

    except Exception as e:
        # shutil.rmtree(task_dir, ignore_errors=True)
        # 删除记录
        with get_db_connection() as conn:
            conn.execute("DELETE FROM ai_tasks WHERE task_id = %s", (task_id,))
            conn.commit()
        raise HTTPException(500, f"文件处理失败: {str(e)}")


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
            is_upload=task['is_upload'],
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
    with get_db_connection() as conn_task:
        task = get_task(conn_task, task_id)
        if not task:
            raise HTTPException(404, "任务不存在")
    if task['is_upload']:
        # 上传文件下载逻辑
        with get_db_connection() as conn_status:
            segments = get_task_results(conn_status, task_id)
            if segment_index < 0 or segment_index >= len(segments["items"]): raise HTTPException(400, "无效的片段索引")
            segment = segments["items"][segment_index]
            audio_url = segment.get("url")
            if not audio_url: raise HTTPException(404, "音频URL不存在")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(audio_url)
                    if response.status_code != 200:
                        raise HTTPException(500, f"无法下载音频片段: {response.status_code}")
                    audio_content = response.content
                buffer = BytesIO(audio_content)
            except Exception as e:
                raise HTTPException(500, f"音频加载失败: {str(e)}")
            filename_base = f"{segment['text'][:50]}_{segment['start']:.2f}-{segment['end']:.2f}.mp3"
            safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename_base)
            return StreamingResponse(
                buffer,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"attachment; filename*=utf-8''{quote(safe_filename)}",
                }
            )
    else:
        # 本地文件下载逻辑
        base_dir = os.path.join("temp_audio_files", task_id)
        segments_dir = os.path.join(base_dir, "merged_segments")
        if not os.path.exists(segments_dir):
            raise HTTPException(404, "音频文件夹不存在")
        filename = format_task_merged_filename(task_id, segment_index)
        file_path = os.path.join(segments_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(404, "音频文件不存在")
        # 从数据库获取当前片段的文本、开始时间和结束时间信息
        with get_db_connection() as conn_status:
            cursor = conn_status.cursor()
            cursor.execute(
                "SELECT text, `start`, end FROM ai_task_results WHERE task_id = %s AND `index` = %s",
                (task_id, segment_index)
            )
            result = cursor.fetchone()
            if not result:
                raise HTTPException(404, "片段信息不存在")
            text = result.get("text", "")
            start = result.get("start", 0.0)
            end = result.get("end", 0.0)
        # filename_base = f"{text[:50]}_{start:.2f}-{end:.2f}.mp3"
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
async def download_bulk_segments(task_id: str,
                                 indices: str = Query(...,
                                                                    description="逗号分隔的原始索引列表")):
    with get_db_connection() as conn_task:
        task = get_task(conn_task, task_id)
        if not task:
            raise HTTPException(404, "任务不存在")
    if task['is_upload']:
        # 上传文件下载逻辑
        with get_db_connection() as conn_status:
            segments = get_all_task_results(conn_status, task_id)
            if not segments:
                raise HTTPException(404, "任务结果不存在")

        try:
            target_indices = [int(idx) for idx in indices.split(',') if idx.strip()]
        except ValueError:
            raise HTTPException(400, "索引格式错误，请使用逗号分隔的整数")
        # 从 segments 中筛选出 index 匹配的项
        matched_segments = []
        for seg in segments:
            if int(seg.get("index")) in target_indices:
                matched_segments.append(seg)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            # 创建外层文件夹
            outer_folder = f"{task_id}_segments_bulk_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            for segment in matched_segments:
                audio_url = segment.get("url")
                speaker = segment.get("speaker", "unknown")
                if not audio_url:
                    continue

                async with httpx.AsyncClient() as client:
                    response = await client.get(audio_url)
                    if response.status_code != 200:
                        continue
                    content = response.content
                    filename_base = f"{segment['text'][:50]}_{segment['start']:.2f}-{segment['end']:.2f}.mp3"
                    safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename_base)
                    folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', speaker)
                    final_path = f"{outer_folder}/{folder_name}/{safe_filename}"
                    zip_file.writestr(final_path, content)

        zip_buffer.seek(0)
        content_length = str(zip_buffer.getbuffer().nbytes)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{outer_folder}.zip"',
                "Content-Length": content_length
            }
        )
    else:
        # 本地文件下载逻辑
        base_dir = os.path.join("temp_audio_files", task_id)
        segments_dir = os.path.join(base_dir, "merged_segments")
        if not os.path.exists(segments_dir):
            raise HTTPException(404, "音频文件夹不存在")
        try:
            target_indices = [int(idx) for idx in indices.split(',') if idx.strip()]
        except ValueError:
            raise HTTPException(400, "索引格式错误，请使用逗号分隔的整数")
        zip_buffer = BytesIO()
        with get_db_connection() as conn_task:
            with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
                # 创建外层文件夹
                outer_folder = f"{task_id}_segments_bulk_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                for index in target_indices:
                    filename = format_task_merged_filename(task_id, index)
                    file_path = os.path.join(segments_dir, filename)
                    if os.path.exists(file_path):
                        # 从数据库获取当前片段的文本、开始时间、结束时间和发音人信息
                        cursor = conn_task.cursor()
                        cursor.execute(
                            "SELECT text, `start`, `end`, speaker FROM ai_task_results WHERE task_id = %s AND `index` = %s",
                            (task_id, index)
                        )
                        result = cursor.fetchone()
                        if not result:
                            raise HTTPException(404, "片段信息不存在")
                        text = result.get("text", "")
                        start = result.get("start", 0.0)
                        end = result.get("end", 0.0)
                        speaker = result.get("speaker", "unknown")
                        # filename_base = f"{text[:50]}_{start:.2f}-{end:.2f}.mp3"
                        filename_base = f"{text[:50]}.mp3"
                        safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename_base)
                        folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', speaker)
                        final_path = f"{outer_folder}/{folder_name}/{safe_filename}"
                        zip_file.writestr(final_path, open(file_path, 'rb').read())
                    else:
                        raise HTTPException(400, f"索引 {index} 对应的文件不存在")
        zip_buffer.seek(0)
        content_length = str(zip_buffer.getbuffer().nbytes)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{outer_folder}.zip"',
                "Content-Length": content_length
            }
        )


@router.get("/download/all/{task_id}", responses={
    200: {"content": {"application/zip": {}}, "description": "返回全部音频"}},
            summary="全部音频下载")
async def download_all_segments(task_id: str):
    with get_db_connection() as conn_task:
        task = get_task(conn_task, task_id)
        if not task:
            raise HTTPException(404, "任务不存在")
    if task['is_upload']:
        # 上传文件下载逻辑
        with get_db_connection() as conn_status:
            segments = get_all_task_results(conn_status, task_id)
            if not segments:
                raise HTTPException(404, "任务结果不存在")

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            # 创建外层文件夹
            outer_folder = f"{task_id}_all_segments_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            for segment in segments:
                audio_url = segment.get("url")
                speaker = segment.get("speaker", "unknown")
                if not audio_url:
                    continue

                async with httpx.AsyncClient() as client:
                    response = await client.get(audio_url)
                    if response.status_code != 200:
                        continue
                    content = response.content
                    filename_base = f"{segment['text'][:50]}_{segment['start']:.2f}-{segment['end']:.2f}.mp3"
                    safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename_base)
                    folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', speaker)
                    final_path = f"{outer_folder}/{folder_name}/{safe_filename}"
                    zip_file.writestr(final_path, content)
        zip_buffer.seek(0)
        content_length = str(zip_buffer.getbuffer().nbytes)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{outer_folder}.zip"',
                "Content-Length": content_length
            }
        )
    else:
        # 本地文件下载逻辑
        base_dir = os.path.join("temp_audio_files", task_id)
        segments_dir = os.path.join(base_dir, "merged_segments")
        if not os.path.exists(segments_dir):
            raise HTTPException(404, "音频文件夹不存在")

        zip_buffer = BytesIO()
        with get_db_connection() as conn_status:
            with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
                # 从数据库获取所有片段的信息
                cursor = conn_status.cursor()
                cursor.execute(
                    "SELECT `index`, text, `start`, `end`, speaker FROM ai_task_results WHERE task_id = %s",
                    (task_id,)
                )
                segments = cursor.fetchall()
                if not segments:
                    raise HTTPException(404, "任务结果不存在")

                # 创建外层文件夹
                outer_folder = f"{task_id}_all_segments_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                speaker_dict = {}
                for segment in segments:
                    speaker_dict[int(segment["index"])] = {
                        "text": segment.get("text", ""),
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "speaker": segment.get("speaker", "unknown")
                    }

                for file_name in os.listdir(segments_dir):
                    file_path = os.path.join(segments_dir, file_name)
                    if os.path.isfile(file_path):
                        # 从文件名中提取索引信息
                        index = int(extract_index_from_filename(file_name))
                        segment_info = speaker_dict.get(index, None)
                        if segment_info:
                            text = segment_info["text"]
                            start = segment_info["start"]
                            end = segment_info["end"]
                            speaker = segment_info["speaker"]
                            # filename_base = f"{text[:50]}_{start:.2f}-{end:.2f}.mp3"
                            filename_base = f"{text[:50]}.mp3"
                            safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename_base)
                            folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', speaker)
                            final_path = f"{outer_folder}/{folder_name}/{safe_filename}"
                            zip_file.writestr(final_path, open(file_path, 'rb').read())
                        else:
                            final_path = f"{outer_folder}/{file_name}"
                            zip_file.writestr(final_path, open(file_path, 'rb').read())

        zip_buffer.seek(0)
        content_length = str(zip_buffer.getbuffer().nbytes)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{outer_folder}.zip"',
                "Content-Length": content_length
            }
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

@router.delete("/segments_keyword/{task_id}", summary="删除包含全部关键词的分段")
async def delete_segments_keyword(
    task_id: str,
    keyword: str =Query(...,description="关键词")
):
    """删除指定任务中包含全部关键词的分段数据"""
    try:
        # 验证任务是否存在及状态
        with get_db_connection() as conn:
            task = get_task(conn, task_id)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            if task['status'] != "completed":
                raise HTTPException(status_code=425, detail="任务尚未完成")

            # 构造关键词列表
            keywords = re.split(r'[;,；，]', keyword)
            keywords = [f"%{k.strip()}%" for k in keywords if k.strip()]

            if not keywords:
                raise HTTPException(status_code=400, detail="关键词不能为空")

            # 构造查询条件
            query = '''
                DELETE FROM ai_task_results 
                WHERE task_id = %s
            '''
            params = [task_id]

            # 添加关键词匹配条件
            for i, keyword in enumerate(keywords):
                if i == 0:
                    query += " AND (text LIKE %s OR speaker LIKE %s"
                    params.extend([keyword, keyword])
                else:
                    query += " AND text LIKE %s AND speaker LIKE %s"
                    params.extend([keyword, keyword])
            query += ")"

            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()

                # 检查是否有行被删除
                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="未找到匹配的分段")

            return {"message": f"任务 {task_id} 中包含全部关键词的分段已删除"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除分段时出错: {str(e)}")


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
