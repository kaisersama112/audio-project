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


from services.audio_service import audio_service
from utils.database import init_db, update_task, get_db_connection, get_task, create_task
from utils.file_utils import cleanup_task

TEMP_DIR = "temp_audio_files"
router = APIRouter(tags=["音频切块"])

# 全局任务状态存储及锁
tasks = {}
tasks_lock = asyncio.Lock()


class Segment(BaseModel):
    index: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    url: Optional[str] = None
    text: Optional[str] = None
    speaker: Optional[str] = None
    suffix: Optional[str] = None


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


async def update_task_status(task_data: dict):
    """更新任务状态（数据库集成版）"""
    with get_db_connection() as conn:
        # 构造完整任务数据
        full_data = {
            "task_id": task_data["task_id"],
            "status": task_data.get("status", "pending"),
            "message": task_data.get("message", ""),
            "progress": task_data.get("progress", 0),
            "segments_path": task_data.get("segments_path"),
            "start_time": task_data.get("start_time"),
            "complete_time": task_data.get("complete_time"),
            "duration": task_data.get("duration"),
            "error": task_data.get("error")
        }

        # 处理时间计算
        if full_data["status"] in ["completed", "failed"]:
            full_data["complete_time"] = datetime.now().isoformat()
            if full_data.get("start_time"):
                start = datetime.fromisoformat(full_data["start_time"])
                end = datetime.fromisoformat(full_data["complete_time"])
                full_data["duration"] = round((end - start).total_seconds(), 2)

        # 更新数据库
        conn.execute("""
        UPDATE tasks SET
            status = ?,
            message = ?,
            progress = ?,
            segments_path = ?,
            start_time = ?,
            complete_time = ?,
            duration = ?,
            error = ?
        WHERE task_id = ?
        """, (
            full_data["status"],
            full_data["message"],
            full_data["progress"],
            full_data["segments_path"],
            full_data["start_time"],
            full_data["complete_time"],
            full_data["duration"],
            full_data["error"],
            full_data["task_id"]
        ))
        conn.commit()



def convert_to_wav(input_path: str, output_path: str):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"格式转换失败: {str(e)}")


async def process_audio_task(task_id: str, original_path: str, original_ext: str):
    task_dir = os.path.join(TEMP_DIR, task_id)
    try:
        await update_task_status({
            "task_id": task_id,
            "status": "processing",
            "message": "开始处理音频文件",
            "progress": 0
        })

        if original_ext.lower() != '.wav':
            await update_task_status({
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

        await update_task_status({
            "task_id": task_id,
            "status": "processing",
            "message": "开始语音识别",
            "progress": 40
        })
        segments = await asyncio.to_thread(audio_service.transcribe_para_former, processing_path, task_id)

        # 结果保存阶段

        await update_task_status({
            "task_id": task_id,
            "status": "processing",
            "message": "保存识别结果",
            "progress": 70
        })
        print(segments)
        with open(os.path.join(task_dir, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        await update_task_status({
            "task_id": task_id,
            "status": "completed",
            "message": "处理完成",
            "progress": 100
        })

    except Exception as e:
        await update_task_status({
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


# 分片上传响应模型
class ChunkUploadResponse(BaseModel):
    task_id: str
    chunk_number: int
    uploaded_chunks: int
    total_chunks: int
    status: str  # partial/complete
# 分片上传接口
@router.post("/upload_chunk", response_model=ChunkUploadResponse)
async def upload_chunk(
    file: UploadFile = File(...),
    chunk_number: int = Form(...),
    total_chunks: int = Form(...),
    file_name: str = Form(...),        # 原始文件名
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


def load_segments_if_completed(task):
    if task[1] == "completed" and task[5]:
        try:
            with open(task[5], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return f"Error loading segments: {str(e)}"
    return None


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse, summary="获取任务状态")
async def get_task_status(task_id: str):
    with get_db_connection() as conn:
        task = get_task(conn, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 将数据库记录转换为响应模型
        return TaskStatusResponse(
            task_id=task[0],
            status=task[1],
            message=task[2],
            progress=task[3],
            start_time=task[6],
            complete_time=task[7],
            duration=task[8],
            error=task[9],
            data=load_segments_if_completed(task)
        )

class PaginatedSegments(BaseModel):
    items: List[Segment]
    total: int
    page: int
    per_page: int
    total_pages: int
    search_hits: int = Field(..., description="包含关键字的记录总数")
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
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100)
):
    # 验证任务状态
    with get_db_connection() as conn:
        task = get_task(conn, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task[1] != "completed":
            raise HTTPException(status_code=425, detail="任务尚未完成")

    # 构建文件路径
    segments_path = os.path.join(TEMP_DIR, task_id, "segments.json")

    try:
        with open(segments_path, "r", encoding="utf-8") as f:
            segments_data = json.load(f)  # 读取原始数据
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="结果文件不存在")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="结果文件格式错误")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"结果读取失败: {str(e)}")

    # 应用筛选条件
    filtered_data = []
    for seg_dict in segments_data:
        # 关键字模糊匹配逻辑
        if keyword:
            # 定义搜索字段池（根据实际数据结构调整）
            search_fields = {
                'text': str(seg_dict.get('text', '')),  # 语音转文字内容
                'type': str(seg_dict.get('segment_type', '')),  # 分段类型
                'labels': '|'.join(seg_dict.get('labels', []))  # 标签列表
            }
            # 组合搜索文本
            search_text = ' '.join(search_fields.values()).lower()
            if keyword.lower() not in search_text:
                continue

        filtered_data.append(seg_dict)
    search_hits = len(filtered_data) if keyword else None
    # 分页处理
    total = len(filtered_data)
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 1
    start = (page - 1) * per_page
    end = start + per_page

    # 处理超出范围的情况
    if start >= total:
        current_page_data = []
    else:
        current_page_data = filtered_data[start:end]

    # 转换为Segment对象
    items = [Segment(**seg) for seg in current_page_data]

    return PaginatedSegments(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        search_hits=search_hits or total
    )
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
