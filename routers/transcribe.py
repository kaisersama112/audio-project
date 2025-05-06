"""
@Project ：pythonProject 
@File    ：transcribe.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51 
"""
import glob
from typing import Optional, List

from fastapi import APIRouter
from fastapi import File, UploadFile, HTTPException, BackgroundTasks, Query
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

from pydub import AudioSegment


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class TranscribeResponse(BaseModel):
    task_id: str
    segments: List[Segment]


def convert_to_wav(input_path: str, output_path: str):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"格式转换失败: {str(e)}")


@router.post("/transcribe/", response_model=TranscribeResponse, summary="音频切片")
async def transcribe_audio(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None
):
    task_id = str(uuid.uuid4())
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
    if background_tasks:
        background_tasks.add_task(cleanup_task, task_id)
    # try:
    # 添加格式转换逻辑（新增代码）
    if original_ext.lower() != '.wav':
        wav_path = os.path.join(task_dir, "audio.wav")
        convert_to_wav(original_path, wav_path)
        processing_path = wav_path
    else:
        processing_path = original_path

    segments = audio_service.transcribe_para_former(processing_path)
    with open(os.path.join(task_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    #
    # except Exception as e:
    #     shutil.rmtree(task_dir)
    #     raise HTTPException(500, f"语音识别失败: {str(e)}")

    return JSONResponse({
        "task_id": task_id,
        "segments": segments
    })


@router.get("/download/single/{task_id}/{segment_index}", responses={
    200: {
        "content": {"audio/mpeg": {}},
        "description": "返回MP3音频片段"
    }
}, summary="单音频下载")
async def download_single_segment(task_id: str, segment_index: int):
    task_dir = os.path.join(TEMP_DIR, task_id)
    if not os.path.exists(task_dir):
        raise HTTPException(404, "任务ID不存在或已过期")

    try:
        with open(os.path.join(task_dir, "segments.json"), "r", encoding="utf-8") as f:
            segments = json.load(f)
    except Exception as e:
        raise HTTPException(500, f"加载分段信息失败: {str(e)}")

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


@router.get("/download/bulk/{task_id}", responses={
    200: {
        "content": {"application/zip": {}},
        "description": "返回ZIP压缩包（按说话人分类的音频片段）"
    }
}, summary="多音频下载")
async def download_bulk_segments(
        task_id: str,
        indices: str = Query(..., description="逗号分隔的片段索引列表")
):
    """批量下载多个音频片段（按说话人分类的ZIP压缩包）"""
    task_dir = os.path.join(TEMP_DIR, task_id)
    if not os.path.exists(task_dir):
        raise HTTPException(404, "任务ID不存在或已过期")

    # 索引参数处理（保持原有增强验证逻辑）
    try:
        cleaned_indices = indices.replace(" ", "").strip()
        if not cleaned_indices:
            raise ValueError("空索引列表")
        indices_list = [int(idx) for idx in cleaned_indices.split(',') if idx]
    except ValueError as e:
        error_detail = f"无效的索引格式: {str(e)}. 请使用逗号分隔的整数，例如：0,1,3"
        raise HTTPException(400, error_detail)

    # 加载分段数据
    try:
        with open(os.path.join(task_dir, "segments.json"), "r", encoding="utf-8") as f:
            segments = json.load(f)
    except Exception as e:
        raise HTTPException(500, f"加载分段信息失败: {str(e)}")

    # 索引验证
    max_index = len(segments) - 1
    invalid_indices = [i for i in indices_list if i < 0 or i > max_index]
    if invalid_indices:
        error_detail = f"包含无效索引: {invalid_indices}（有效范围：0-{max_index}）"
        raise HTTPException(400, error_detail)

    # 加载音频文件
    try:
        original_files = glob.glob(os.path.join(task_dir, "original_audio.*"))
        if not original_files:
            raise FileNotFoundError("找不到原始音频文件")
        audio = AudioSegment.from_file(original_files[0])
    except Exception as e:
        raise HTTPException(500, f"音频加载失败: {str(e)}")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx in indices_list:
            segment = segments[idx]

            # 获取并清理说话人信息
            speaker = segment.get("speaker", "unknown")
            speaker_clean = re.sub(r'[\\/*?:"<>|]', '_', speaker)  # 替换非法字符

            # 生成文件名
            time_suffix = f"{segment['start']:.2f}-{segment['end']:.2f}"
            clean_text = re.sub(r'[\\/*?:"<>|]', '_', segment["text"])[:50]  # 限制长度
            filename = os.path.join(
                speaker_clean,
                f"{clean_text}_{time_suffix}.mp3" if clean_text else f"segment_{time_suffix}.mp3"
            )

            # 提取音频片段
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            audio_segment = audio[start_ms:end_ms]

            # 写入压缩包
            audio_buffer = BytesIO()
            audio_segment.export(audio_buffer, format="mp3", bitrate="128k")
            zip_file.writestr(filename, audio_buffer.getvalue())

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=classified_segments.zip",
            "Content-Type": "application/zip"  # 显式设置
        },

    )


@router.get("/download/all/{task_id}",
            responses={
                200: {
                    "content": {"application/zip": {}},
                    "description": "返回包含所有片段的ZIP压缩包"
                }
            }, summary="全部音频下载")
async def download_all_segments(task_id: str):
    """下载全部音频片段（按说话人分类的ZIP压缩包）"""
    task_dir = os.path.join(TEMP_DIR, task_id)
    if not os.path.exists(task_dir):
        raise HTTPException(404, "任务ID不存在或已过期")

    # 加载所有片段的索引
    try:
        with open(os.path.join(task_dir, "segments.json"), "r", encoding="utf-8") as f:
            segments = json.load(f)
        indices = list(range(len(segments)))  # 生成所有索引
    except Exception as e:
        raise HTTPException(500, f"加载分段信息失败: {str(e)}")

    # 复用批量下载逻辑
    return await download_bulk_segments(
        task_id=task_id,
        indices=",".join(map(str, indices))  # 生成逗号分隔的索引字符串
    )
