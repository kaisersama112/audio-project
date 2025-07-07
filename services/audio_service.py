"""
@Project ：pythonProject
@File    ：audio_service.py.py
@IDE     ：PyCharm
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51
"""
import concurrent
import shutil
from functools import wraps
from urllib.parse import urlparse

import aiofiles
import zipfile
from datetime import datetime
import glob
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, Tuple

import torchaudio
from fastapi import HTTPException
from pydub import AudioSegment
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from config import base_url
from config import hotword_list, TEMP_DIR, DOWNLOAD_DIR
import time
import os
import subprocess
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from curd.async_crud import get_db_async, update_task_status_async, get_task_async, create_task_async
from curd.models import AITaskResult, AIDownloadTask
from models.schemas import Segment
from services.oss_service import oss_service

import asyncio

from utils.ucloud_u3d import UCloudFileDownloader

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 定义最大并发数和全局锁

TRANSCRIBE_LOCK = asyncio.Lock()


def convert_to_wav(input_path: str, output_path: str):
    """
    将音频文件转换为WAV格式
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :return: None
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    except Exception as e:
        raise RuntimeError(f"格式转换失败: {str(e)}")


def convert_to_wav_ffmpeg(input_path: str, output_path: str):
    """
    将音频文件转换为WAV格式

    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :return: None
    """
    try:

        command = [
            "ffmpeg",
            "-i", input_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]

        # 执行命令
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        # 捕获命令执行错误
        raise RuntimeError(f"格式转换失败: {e.stderr.decode().strip()}") from e
    except Exception as e:
        raise RuntimeError(f"格式转换失败: {str(e)}") from e


async def update_status(task_id, status: str, message: str, progress: int, complete_time=None):
    """
    更新任务状态
    :param task_id: 任务ID
    :param status: 状态
    :param message: 消息
    :param progress: 进度
    :param complete_time: 创建时间
    :return: None
    """
    async with get_db_async() as db:
        if complete_time:
            return await update_task_status_async(db, {
                "task_id": task_id,
                "status": status,
                "message": message,
                "progress": progress,
                "complete_time": complete_time
            })
        return await update_task_status_async(db, {
            "task_id": task_id,
            "status": status,
            "message": message,
            "progress": progress,
        })


def split_progress_callback(task_id, progress, message):
    """
    分片进度回调函数
    :param task_id: 任务ID
    :param progress: 进度百分比
    :param message: 进度消息
    :return: None
    """
    update_status(task_id=task_id,
                  status="processing",
                  message=f"保存片段: {message}",
                  progress=60 + int(progress * 0.3),
                  complete_time=None
                  )


def merge_progress_callback(task_id, progress, message):
    """
    合并进度回调函数
    :param task_id: 任务ID
    :param progress: 进度百分比
    :param message: 进度消息
    :return: None
    """
    update_status(task_id, "processing", f"合并片段: {message}", 30 + int(progress * 0.3))


def retry(max_retries: int, retry_delay: float, exception_types: Tuple[type] = (Exception,), db_update: bool = True):
    """任务失败重试装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exception_types as e:
                    task_id = args[0]
                    print(f"Task {task_id} - 第 {attempt + 1} 次尝试失败: {str(e)}")
                    if db_update:
                        async with get_db_async() as db:
                            await update_task_status_async(
                                db,
                                {
                                    "task_id": task_id,
                                    "status": "retrying",
                                    "message": f"第 {attempt + 1} 次尝试失败，正在重试",
                                    "progress": 0,
                                    "error": str(e)
                                }
                            )

                await asyncio.sleep(retry_delay)
            task_id = args[0]
            print(f"Task {task_id} - 已达到最大重试次数: {max_retries}")
            if db_update:
                async with get_db_async() as db:
                    await update_task_status_async(
                        db,
                        {
                            "task_id": task_id,
                            "status": "failed",
                            "message": "已达到最大重试次数",
                            "progress": 100,
                            "error": f"已达到最大重试次数: {max_retries}"
                        }
                    )

            raise Exception(f"Task {task_id} - 已达到最大重试次数: {max_retries}")

        return wrapper

    return decorator


def timeout(seconds: int, db_update: bool = True):
    """任务超时控制装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = None
            start_time = time.time()
            task_id = args[0]
            try:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                print(f"Task {task_id} - 超过 {seconds} 秒未完成，已中断")
                if db_update:
                    async with get_db_async() as db:
                        await update_task_status_async(
                            db,
                            {
                                "task_id": task_id,
                                "status": "failed",
                                "message": "处理超时",
                                "progress": 100,
                                "error": f"处理超时，超过 {seconds} 秒未完成",
                            }
                        )
                raise
            finally:
                print(f"Task {task_id} - 任务总耗时: {time.time() - start_time:.2f}秒")
            return result

        return wrapper

    return decorator


@retry(max_retries=3, retry_delay=5)
@timeout(seconds=60 * 60 * 6)  # 设置每个音频任务的最大处理时间为6小时：60分钟*6
async def process_audio_task(
        task_id: str,
        file_url,
        min_chunk_duration: float,
        separate
):
    """
    处理音频任务
    :param task_id: 任务ID
    :param file_url: 音频文件URL
    :param min_chunk_duration: 最小分片时长
    :param separate: 是否分片
    """
    task_dir = os.path.join(TEMP_DIR, task_id)
    start_time = time.time()  # 开始计时

    loop = asyncio.get_running_loop()

    original_path, original_ext = await loop.run_in_executor(
        None,
        lambda: asyncio.run(audio_service.audio_download_progress_callback(task_id, file_url, task_dir))
    )

    print(f"Task {task_id} - 音频下载耗时: {time.time() - start_time:.2f}秒")
    await update_status(task_id, "processing", "开始处理音频文件", 0)
    stage_start_time = time.time()
    if original_ext.lower() != '.wav':
        await update_status(task_id, "processing", "正在转换音频格式", 10)
        wav_path = os.path.join(task_dir, "audio.wav")
        await asyncio.to_thread(convert_to_wav_ffmpeg, original_path, wav_path)
        print(f"Task {task_id} - 音频格式转换耗时: {time.time() - stage_start_time:.2f}秒")
        # 清理原始音频文件
        os.remove(original_path)
        processing_path = wav_path
    else:
        processing_path = original_path
        await update_status(task_id, "processing", "音频格式无需转换", 10)
    try:
        await update_status(task_id, "processing", "开始语音识别", 20)
        stage_start_time = time.time()

        # 使用锁确保 transcribe_para_former 是串行调用
        async with TRANSCRIBE_LOCK:
            result = await asyncio.to_thread(
                audio_service.transcribe_para_former,
                task_id,
                processing_path,
                separate
            )
            if not isinstance(result, list) or len(result) == 0 or not isinstance(result[0], dict):
                await update_status(task_id, "processing", "当前音频并未识别到有效数据,请检查你的原始音频文件！", 100)
                return
        print(f"Task {task_id} - 语音识别耗时: {time.time() - stage_start_time:.2f}秒")

        # 发音人合并
        await update_status(task_id, "processing", "合并发音人信息", 85)
        stage_start_time = time.time()

        raw_segments = result[0]["sentence_info"]
        merged_segments = await asyncio.to_thread(
            audio_service.merge_segments,
            task_id,
            raw_segments,
            min_chunk_duration,
            merge_progress_callback
        )
        print(f"Task {task_id} - 合并发音人信息耗时: {time.time() - stage_start_time:.2f}秒")

        await update_status(task_id, "processing", "格式化识别结果", 90)
        stage_start_time = time.time()

        # 保存所有分片到本地
        segments_paths = await asyncio.to_thread(
            audio_service.split_segments,
            merged_segments,
            processing_path,
            task_id,
            split_progress_callback
        )
        print(f"Task {task_id} - 分割音频片段耗时: {time.time() - stage_start_time:.2f}秒")

        # 结果保存阶段
        await update_status(task_id, "processing", "保存识别结果", 95)
        stage_start_time = time.time()

        async with get_db_async() as db:
            try:
                # 构建要插入的对象列表
                objects_to_insert = []
                for idx, segment_path, merged_seg in segments_paths:
                    url = segment_path.replace("\\", "/").replace("/root/autodl-fs", "")
                    objects_to_insert.append(
                        AITaskResult(
                            task_id=task_id,
                            index=idx,
                            start=merged_seg.get("start"),
                            end=merged_seg.get("end"),
                            text=merged_seg.get("text"),
                            speaker=str(merged_seg.get("spk")),
                            url=base_url + url
                        )
                    )

                # 批量添加
                db.add_all(objects_to_insert)
                await db.commit()

            except SQLAlchemyError as e:
                await db.rollback()

                raise HTTPException(500, detail=f"数据库插入失败: {str(e)}")
        print(f"Task {task_id} - 保存识别结果耗时: {time.time() - stage_start_time:.2f}秒")

        await update_status(task_id, "completed", "处理完成", 100, complete_time=datetime.now().isoformat())
        print(f"Task {task_id} - 总耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        async with get_db_async() as conn:
            await update_task_status_async(conn, {
                "task_id": task_id,
                "status": "failed",
                "message": "处理过程中发生错误",
                "progress": 100,
                "error": str(e),
                "complete_time": datetime.now().isoformat()
            })
            print(f"Task {task_id} - 处理失败，耗时: {time.time() - start_time:.2f}秒")
    finally:
        # 清理原始数据，只保留切片数据
        try:
            # 清理原始音频文件（如果存在）
            if os.path.exists(original_path):
                os.remove(original_path)
                print(f"Task {task_id} - 已清理原始音频文件: {original_path}")
            # 清理处理后的音频文件（如果存在）
            if os.path.exists(processing_path) and processing_path != original_path:
                os.remove(processing_path)
                print(f"Task {task_id} - 已清理处理后的音频文件: {processing_path}")
            # 清理任务目录中的中间文件
            for chunk in glob.glob(os.path.join(task_dir, "chunk_*.wav")):
                os.remove(chunk)
                print(f"Task {task_id} - 已清理中间文件: {chunk}")

            # 如果任务目录为空，则删除目录
            if os.path.exists(task_dir) and not os.listdir(task_dir):
                os.rmdir(task_dir)
                print(f"Task {task_id} - 已删除空任务目录: {task_dir}")

        except Exception as e:
            print(f"Task {task_id} - 清理文件时发生错误: {str(e)}")


async def get_task_result_dict_and_files(task_id: str, segments_dir: str):
    """
    获取 result_dict 和 file_list
    """
    async with get_db_async() as conn:
        stmt = select(AITaskResult).where(
            AITaskResult.task_id == task_id
        )
        result = await conn.execute(stmt)
        results = result.scalars().all()

        result_dict = {r.index: r for r in results}

    file_list = []
    for file_name in os.listdir(segments_dir):
        file_path = os.path.join(segments_dir, file_name)
        if not os.path.isfile(file_path):
            print(f"警告：路径 {file_path} 不是文件，已跳过")
            continue
        index = extract_index_from_filename(file_name)
        if index is None:
            print(f"无法从文件名 {file_name} 中提取索引")
            continue
        file_list.append((file_path, file_name))

    return result_dict, file_list


async def read_file(file_path):
    async with aiofiles.open(file_path, 'rb') as f:
        return await f.read()


async def write_files_to_zip_task(zip_file, outer_folder, file_list, result_dict, lock):
    speaker_files = {}
    # 分类 speaker
    for file_path, file_name in file_list:
        index = extract_index_from_filename(file_name)
        result = result_dict.get(index)
        speaker = result.speaker if result else "unknown"
        if speaker not in speaker_files:
            speaker_files[speaker] = []
        speaker_files[speaker].append((file_path, file_name))

    # 写入 ZIP
    for speaker, files in speaker_files.items():
        speaker_folder = os.path.join(outer_folder, speaker)
        for file_path, file_name in files:
            index = extract_index_from_filename(file_name)
            result = result_dict.get(index)

            if not result or not result.text:
                print(f"未找到索引 {index} 的文本信息")
                new_filename = f"{index}.mp3"
            else:
                safe_text = "".join(
                    c if c.isalnum() or c in (" ", "_", "-") else "_" for c in result.text[:50]
                )
                new_filename = f"{safe_text}.mp3"

            content = await read_file(file_path)
            async with lock:
                zip_file.writestr(f"{speaker_folder}/{new_filename}", content)


async def process_download_task(download_task_id: str, task_id: str, indices: list[str] or None, download_type: str):
    print("进入下载异步任务")
    try:
        async with get_db_async() as conn_task:
            task = await get_task_async(conn_task, task_id)
            if not task:
                return

        base_dir = os.path.join(TEMP_DIR, task_id)
        segments_dir = os.path.join(base_dir, "merged_segments")
        if not os.path.exists(segments_dir):
            return

        async with get_db_async() as db:
            # 查询记录
            stmt = select(AIDownloadTask).where(
                AIDownloadTask.task_id == download_task_id
            )
            result = await db.execute(stmt)
            download_task = result.scalars().first()

            if not download_task:
                return

            # 更新字段（方式一：直接赋值并提交）
            download_task.status = 'processing'
            download_task.updated_at = datetime.now()

            # 提交事务（需要启用 sync 模式 flush，因为 AsyncSession 默认不支持 commit）
            await db.commit()
            await db.refresh(download_task)  # 可选：刷新对象以获取最新数据（如自动生成字段）

        zip_buffer = BytesIO()
        lock = asyncio.Lock()
        with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            result_dict, file_list = await get_task_result_dict_and_files(task_id, segments_dir)
            tasks = []

            if download_type == "bulk":
                outer_folder = f"{task_id}_bulk_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                target_indices = indices or []
                filtered_file_list = [
                    (fp, fn) for fp, fn in file_list
                    if extract_index_from_filename(fn) in target_indices
                ]
                tasks.append(write_files_to_zip_task(zip_file, outer_folder, filtered_file_list, result_dict, lock))

            elif download_type == "all":
                outer_folder = f"{task_id}_all_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                tasks.append(write_files_to_zip_task(zip_file, outer_folder, file_list, result_dict, lock))

            await asyncio.gather(*tasks)

        zip_buffer.seek(0)

        download_filename = f"{outer_folder}.zip"
        download_path = os.path.join(DOWNLOAD_DIR, download_filename)
        async with aiofiles.open(download_path, 'wb') as f:
            await f.write(zip_buffer.read())

        async with get_db_async() as db:
            # 查询任务
            stmt = select(AIDownloadTask).where(
                AIDownloadTask.task_id == download_task_id
            )
            result = await db.execute(stmt)
            download_task = result.scalars().first()

            if not download_task:
                return

            # 更新字段
            segment_path = base_url + download_path
            url = segment_path.replace("\\", "/").replace("/root/autodl-fs", "")

            download_task.status = 'completed'
            download_task.progress = 100
            download_task.file_url = url
            download_task.download_path = download_path
            download_task.updated_at = datetime.now()

            # 提交事务（注意：需要 await）
            await db.commit()
            await db.refresh(download_task)  # 可选：刷新对象状态
        print(f"task_id:{task_id},download_task_id:{download_task_id}执行完毕")
    except Exception as e:
        print(f"处理下载任务时发生错误: {e}")
        try:

            stmt = select(AIDownloadTask).where(
                AIDownloadTask.task_id == download_task_id
            )
            result = await db.execute(stmt)
            download_task = result.scalars().first()

            if download_task:
                # 更新字段
                download_task.status = 'failed'
                download_task.updated_at = datetime.now()
                await db.commit()
                await db.refresh(download_task)

        except SQLAlchemyError as e:
            await db.rollback()
            # 可以记录日志或抛出 HTTP 异常
            raise


def merge_with_ffmpeg(task_dir: str, output_path: str):
    """使用FFmpeg合并分片文件"""
    # 生成分片列表文件

    concat_list = os.path.join(task_dir, "concat_list.txt")
    if os.path.exists(concat_list):
        os.remove(concat_list)
        os.remove(os.path.join(task_dir, "merged.wav"))
    with open(concat_list, "w") as f:
        for chunk in sorted(glob.glob(os.path.join(task_dir, "chunk_*.wav"))):
            f.write(f"file '{os.path.basename(chunk)}'\n")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-b:a", "128k",
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


async def load_segments_if_completed(db: AsyncSession, task_id: str):
    """
    异步检查任务是否已完成，如果完成则加载结果
    """
    try:
        # 构建查询语句
        stmt = select(AITaskResult).where(
            AITaskResult.task_id == task_id
        ).order_by(AITaskResult.index)

        # 执行查询
        result = await db.execute(stmt)
        results = result.scalars().all()

        # 转换为 Segment 对象列表
        segment_list = [Segment(**val.__dict__) for val in results]
        return segment_list
    except SQLAlchemyError as e:
        return f"Error loading segments: {str(e)}"


def format_task_merged_filename(task_id: str, index: int, suffix: str = "mp3"):
    """格式化合并后的文件名"""
    filename = f"merged_{task_id}_{index:05d}.{suffix}"
    return filename


def extract_index_from_filename(filename):
    match = re.search(r'merged_\d+_(\d{5})\.mp3$', filename)
    if match:
        return int(match.group(1))
    return None


class TaskQueueManager:
    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.max_concurrent_tasks = 5  # 最大并发任务数
        self.running_tasks = set()
        self.lock = asyncio.Lock()

    async def add_task(self,
                       task_id: str,
                       file_url: str,
                       # original_path: str,
                       # original_ext: str,
                       min_chunk_duration: float,
                       separate: bool
                       ):
        """添加任务到队列"""
        async with self.lock:
            self.task_queue.put_nowait((
                task_id,
                file_url,
                # original_path,
                # original_ext,
                min_chunk_duration,
                separate)
            )
            print(f"Task {task_id} - 已添加到任务队列")
            # 记录任务初始状态
            async with get_db_async() as db:
                await  update_task_status_async(
                    db,
                    {
                        "task_id": task_id,
                        "status": "pending",
                        "message": "任务已添加到队列，等待处理",
                        "progress": 0,

                    }
                )

    async def process_task(self):
        """处理单个任务"""
        while True:
            task = await self.task_queue.get()
            task_id = task[0]

            async with self.lock:
                self.running_tasks.add(task_id)

            try:
                await process_audio_task(*task)
            except Exception as e:
                print(f"Task {task_id} - 处理失败: {str(e)}")
                # 如果失败，则重新添加到队列（根据业务需求决定是否重新调度失败任务）
                if isinstance(e, asyncio.TimeoutError):
                    print(f"Task {task_id} - 超时重试")
                    await self.add_task(*task)
            finally:
                async with self.lock:
                    self.running_tasks.discard(task_id)
                self.task_queue.task_done()  # 标记任务完成

    async def start(self):
        """启动任务队列处理器"""
        for _ in range(self.max_concurrent_tasks):
            asyncio.create_task(self.process_task())
        print("TaskQueueManager - 已启动任务队列处理器")

    async def stop(self):
        """停止任务队列处理器"""
        async with self.lock:
            # 添加特殊任务来停止处理器
            for _ in range(self.max_concurrent_tasks):
                self.task_queue.put_nowait(None)
            await self.task_queue.join()
            print("TaskQueueManager - 已停止任务队列处理器")


class AudioService:
    def __init__(self):
        self.ans_model = None
        self.transcribe_para_former_model = None
        self.model = None

    def load_model(self):
        self.ans_model = pipeline(
            Tasks.acoustic_noise_suppression,
            model='pre_model/speech_frcrn_ans_cirm_16k')

        self.transcribe_para_former_model = pipeline(
            task=Tasks.auto_speech_recognition,
            model="pre_model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="pre_model/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="pre_model/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="pre_model/speech_campplus_sv_zh-cn_16k-common",
            disable_update=True,
            batch_size=4
        )
        # self.ans_model = pipeline(
        #     Tasks.acoustic_noise_suppression,
        #     model='iic/speech_frcrn_ans_cirm_16k')
        #
        # self.transcribe_para_former_model = pipeline(
        #     task=Tasks.auto_speech_recognition,
        #     model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        #     vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        #     punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        #     spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
        #     disable_update=True,
        #     batch_size=4
        # )

        print("Models loaded successfully")

    def transcribe_para_former(self, task_id: str, file_path: str, separate: bool,
                               max_segment_duration: int = 600000 * 6):
        """
        根据 separate 参数决定是否先进行人声分离再进行语音识别。

        :param task_id: 任务ID
        :param file_path: 原始音频文件路径
        :param separate: 是否启用人声与背景音分离
        :param max_segment_duration: 单次处理的最大时长（毫秒） 30 MINUS
        :return: 识别结果
        """
        try:
            if separate:
                self.ans_model(
                    file_path,
                    output_path=file_path
                )
            audio_info = self.get_audio_info(file_path)
            audio_duration = audio_info['duration']
            min_segment_duration = 10 * 60 * 1000  # 每段至少10分钟

            if audio_duration > max_segment_duration:
                # 计算段落数，确保最后一段的长度不会过短
                num_segments = int(audio_duration / max_segment_duration)
                remainder = audio_duration % max_segment_duration
                if remainder > 0:
                    num_segments += 1

                results = []
                for i in range(num_segments):
                    start_time = i * max_segment_duration
                    end_time = min((i + 1) * max_segment_duration, audio_duration)
                    # 确保最后一段至少有 min_segment_duration 的长度
                    if i == num_segments - 1 and (end_time - start_time) < min_segment_duration:
                        start_time = max(0, end_time - min_segment_duration)
                        # 确保不超过音频总时长
                        start_time = min(start_time, audio_duration - min_segment_duration)
                        end_time = audio_duration  # 确保最后一段结束时间不超过音频总时长

                    segment = self.cut_audio_to_memory(file_path, start_time, end_time)
                    segment_result = self.transcribe_segment(
                        segment,
                        start_time,  # 正确传递起始时间
                        file_path
                    )
                    results.append(segment_result)
                    if task_id:
                        progress = int((i + 1) / num_segments * 70) + 15  # 15% 到 85%
                        update_status(task_id, "processing", f"处理中，已完成 {i + 1} / {num_segments} 段", progress)

                final_result = self.merge_results(results)
                return final_result
            else:
                # 不做分离，直接识别原始音频
                result = self.transcribe_para_former_model(
                    input=file_path,
                    batch_size_token=4000,
                    batch_size_token_threshold_s=30,
                    vad=True,
                    punc=True,
                    spk=True
                )
                if task_id:
                    update_status(task_id, "processing", "处理中，已完成", 80)
                return result
        finally:
            if os.path.exists(file_path + "_temp_segment.wav"):
                os.remove(file_path + "_temp_segment.wav")
                print(f"已清理临时音频片段文件: {file_path + '_temp_segment.wav'}")

    def get_audio_info(self, file_path):
        """
        获取音频文件信息（时长等）

        :param file_path: 音频文件路径
        :return: 音频信息字典
        """
        # 使用 torchaudio 获取音频信息
        audio_info = torchaudio.info(file_path)
        sample_rate = audio_info.sample_rate
        num_frames = audio_info.num_frames
        channels = audio_info.num_channels

        # 计算音频时长（毫秒）
        duration_ms = (num_frames / sample_rate) * 1000

        return {
            'duration': duration_ms,
            'sample_rate': sample_rate,
            'channels': channels,
            'num_frames': num_frames
        }

    def cut_audio_to_memory(self, input_file, start_time, end_time):
        """
        截取音频片段并保存到内存

        :param input_file: 输入音频文件路径
        :param start_time: 开始时间（毫秒）
        :param end_time: 结束时间（毫秒）
        :return: 截取后的音频片段对象
        """
        # 使用 pydub 加载音频文件
        audio = AudioSegment.from_file(input_file)

        # 截取音频片段
        segment = audio[start_time:end_time]

        return segment

    def transcribe_segment(self, segment, segment_start_time, file_path):
        """
        对音频片段进行识别

        :param segment: 音频片段对象
        :param segment_start_time: 当前片段的起始时间（毫秒）
        :param file_path: 原始音频文件路径
        :return: 识别结果
        """
        # 将音频片段保存到临时文件
        temp_file = f"{file_path}_temp_segment.wav"
        segment.export(temp_file, format='wav')

        # 对片段进行识别
        segment_result = self.transcribe_para_former_model(
            input=temp_file,
            batch_size_token=4000,
            batch_size_token_threshold_s=30,
            # max_single_segment_time=20000,
            # vad_speech_noise_ratio=0.5,
            hotword=",".join(hotword_list),
            vad=True,
            punc=True,
            spk=True
        )

        return segment_result, segment_start_time

    def merge_results(self, results):
        """
        合并多个片段的结果，调整时间戳

        :param results: 所有片段的识别结果及其起始时间
        :return: 合并后的识别结果
        """
        merged_result = {
            'sentence_info': []
        }

        for result_item in results:
            result, segment_start_time = result_item
            if isinstance(result, list) and len(result) > 0 and 'sentence_info' in result[0]:
                # 遍历片段中的每个句子
                for sentence in result[0]['sentence_info']:
                    # 调整时间戳
                    new_sentence = {
                        'end': segment_start_time + sentence['end'],
                        'spk': sentence['spk'],
                        'start': segment_start_time + sentence['start'],
                        'text': sentence['text'],
                        'timestamp': []
                    }

                    # 调整时间戳数组
                    for timestamp in sentence['timestamp']:
                        new_sentence['timestamp'].append(
                            [segment_start_time + timestamp[0],
                             segment_start_time + timestamp[1]]
                        )

                    merged_result['sentence_info'].append(new_sentence)
            else:

                print(f"结果结构不匹配，跳过处理：{result}")

        return [merged_result] if merged_result['sentence_info'] else []

    def _upload_single_segment(self, segment_path: str, task_id: str, seg: dict, idx: int) -> Dict:
        """单个片段的上传任务"""
        try:
            oss_url = oss_service.upload_file(segment_path, task_id)

            return {
                "index": idx,
                "start": seg["start"],
                "end": seg["end"],
                "url": oss_url,
                "text": seg["text"].strip(),
                "speaker": str(seg.get("spk", "unknown")),
                "local_path": segment_path
            }
        except Exception as e:
            return {
                "index": idx,
                "error": str(e),
                "local_path": segment_path
            }

    def formatted_results_upload(self, merged_segments, file_path, task_id):
        """
        合并片段并上传到OSS，返回格式化后的结果
        """
        formatted_results = []
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        for seg in merged_segments:
            if seg["end"] > duration_ms / 1000:
                raise ValueError("片段超出音频长度")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for idx, merged_seg in enumerate(merged_segments):
                try:
                    segment_path = self._save_merged_segment(
                        audio=audio,
                        duration_ms=duration_ms,
                        original_path=file_path,
                        start=merged_seg["start"],
                        end=merged_seg["end"],
                        index=idx,
                        task_id=task_id
                    )
                    futures.append((idx, executor.submit(
                        self._upload_single_segment,
                        segment_path=segment_path,
                        task_id=task_id,
                        seg=merged_seg,
                        idx=idx
                    )))

                except Exception as e:
                    formatted_results.append({
                        "index": idx,
                        "error": f"合并片段保存失败: {str(e)}",
                        "start": merged_seg["start"],
                        "end": merged_seg["end"]
                    })

            # 处理上传结果
            for idx, future in futures:
                try:
                    upload_result = future.result()
                    formatted_results.append(upload_result)

                    # 上传成功后清理临时文件
                    if upload_result.get("url") and os.path.exists(upload_result["local_path"]):
                        os.remove(upload_result["local_path"])

                except Exception as e:
                    formatted_results.append({
                        "index": idx,
                        "error": f"合并片段上传失败: {str(e)}",
                        "local_path": upload_result.get("local_path"),
                        "start": merged_seg["start"],
                        "end": merged_seg["end"]
                    })
        # 按开始时间排序
        formatted_results.sort(key=lambda x: x["start"])
        return formatted_results

    def merge_segments(self, task_id: str, raw_segments: list, min_chunk_duration: float, progress_callback) -> list:
        """合并连续相同说话人的片段，基于最小时长要求"""
        if not raw_segments:
            progress_callback(task_id, 100, "无片段需要合并")
            return []
        merged = []
        total_segments = len(raw_segments)
        current = {
            "start": raw_segments[0]["start"],
            "end": raw_segments[0]["end"],
            "text": raw_segments[0]["text"].strip(),
            "spk": raw_segments[0].get("spk", "unknown"),
            "count": 1  # 添加计数器
        }
        for i, seg in enumerate(raw_segments[1:]):
            current_duration = (current["end"] - current["start"]) / 1000  # 计算当前片段时长（转为秒）
            # 合并条件：相同说话人 + 间隔 <0.5秒 + 当前片段时长未达到最小要求
            can_merge = (
                    seg.get("spk") == current["spk"] and
                    (seg["start"] - current["end"]) <= 500 and  # 0.5秒转为毫秒
                    current_duration < min_chunk_duration
            )

            if can_merge:
                current["end"] = seg["end"]
                current["text"] += " " + seg["text"].strip()
                current["count"] += 1
            else:
                # 检查当前片段时长是否满足最小要求
                if (current["end"] - current["start"]) / 1000 >= min_chunk_duration:
                    # 保存当前片段（去除count字段）
                    merged.append({
                        "start": current["start"],
                        "end": current["end"],
                        "text": current["text"],
                        "spk": current["spk"]
                    })
                    # 更新进度条
                    progress = int(((i + 1) / total_segments) * 100)
                    if progress % 100 == 0:
                        progress_callback(task_id, progress, f"已合并 {i + 1} 个片段")
                    # 重置current为当前seg
                    current = {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                        "spk": seg.get("spk", "unknown"),
                        "count": 1
                    }
                else:
                    # 如果不满足最小时长，继续尝试合并后续片段
                    current["end"] = seg["end"]
                    current["text"] += " " + seg["text"].strip()
                    current["count"] += 1
        # 处理最后一个片段
        # 检查最后一个片段时长是否满足最小要求
        if (current["end"] - current["start"]) / 1000 >= min_chunk_duration:
            progress_callback(task_id, 100, "片段合并完成")
            merged.append({
                "start": current["start"],
                "end": current["end"],
                "text": current["text"],
                "spk": current["spk"]
            })
        else:
            # 如果最后一个片段不满足最小时长，可以考虑与前一个片段合并或单独处理
            # 这里简单起见，直接添加到结果中，实际应用中可根据需求调整
            merged.append({
                "start": current["start"],
                "end": current["end"],
                "text": current["text"],
                "spk": current["spk"]
            })
            progress_callback(task_id, 100, "片段合并完成（最后一个片段未达最小时长）")
        return merged

    @staticmethod
    def _save_merged_segment(audio, duration_ms, original_path: str, start: float, end: float,
                             index: int, task_id: str) -> str:
        """保存合并后的长片段（修复变量引用问题）"""
        try:
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"音频文件不存在: {original_path}")
            start_ms = int(start)
            end_ms = int(end)
            if start_ms <= 0 or end_ms >= duration_ms:
                raise ValueError(
                    f"时间范围超出音频边界: 0-{duration_ms / 1000:.2f}s "
                    f"(请求范围: {start:.2f}-{end:.2f}s)"
                )
            segment = audio[start_ms:end_ms]
            output_dir = os.path.join(os.path.dirname(original_path), "merged_segments")
            os.makedirs(output_dir, exist_ok=True)
            filename = format_task_merged_filename(task_id, index)
            output_path = os.path.join(output_dir, filename)
            # 6. 导出音频文件
            segment.export(
                output_path,
                format="mp3",
                codec="libmp3lame",
                bitrate="192k",
                tags={
                    'title': f"Merged {index}",
                    'artist': 'Audio Processing System',
                    'comment': f"Original: {start:.2f}-{end:.2f}s"
                }
            )
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
                raise IOError("生成的音频文件无效或为空")
            return output_path
        except Exception as e:
            error_context = (
                f"[文件: {original_path}] "
                f"[时间范围: {start:.2f}s-{end:.2f}s] "
                f"[任务ID: {task_id}] "
                f"[索引: {index}]"
            )
            raise RuntimeError(f"合并片段保存失败: {error_context} → {str(e)}") from e

    def split_segments(self, merged_segments, file_path, task_id, progress_callback):
        segments_paths = []
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        total_segments = len(merged_segments)
        max_workers = min(10, os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 创建任务列表
            futures = []
            for idx, merged_seg in enumerate(merged_segments):
                future = executor.submit(
                    AudioService._save_merged_segment,
                    audio=audio,
                    duration_ms=duration_ms,
                    original_path=file_path,
                    start=merged_seg["start"],
                    end=merged_seg["end"],
                    index=idx,
                    task_id=task_id
                )
                futures.append(future)

            # 使用 map 方法保持结果顺序
            for i, (future, merged_seg) in enumerate(zip(futures, merged_segments)):
                try:
                    segment_path = future.result()
                    segment_url = f"{segment_path}"
                    segments_paths.append((i, segment_url, merged_seg))
                    if i % 100 == 0:
                        progress = int(((i + 1) / total_segments) * 100)
                        progress_callback(task_id, progress, f"已保存 {i + 1} 个片段")
                except Exception as e:
                    print(f"保存片段 {idx} 失败: {str(e)}")
        return segments_paths

    def upload_segments(self, segments_paths, task_id):
        """
        上传分片
        """
        formatted_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for idx, segment_path, merged_seg in segments_paths:
                futures.append((idx, executor.submit(
                    self._upload_single_segment,
                    segment_path=segment_path,
                    task_id=task_id,
                    seg=merged_seg,
                    idx=idx
                )))

            # 处理上传结果
            for idx, future in futures:
                try:
                    upload_result = future.result()
                    formatted_results.append(upload_result)

                    # 上传成功后清理临时文件
                    if upload_result.get("url") and os.path.exists(upload_result["local_path"]):
                        os.remove(upload_result["local_path"])

                except Exception as e:
                    formatted_results.append({
                        "index": idx,
                        "error": f"合并片段上传失败: {str(e)}",
                        "local_path": upload_result.get("local_path"),
                        "start": merged_seg["start"],
                        "end": merged_seg["end"]
                    })

        # 按开始时间排序
        formatted_results.sort(key=lambda x: x.get("start", 0))
        return formatted_results

    async def audio_download_progress_callback(self, task_id: str, file_url: str, task_dir: str) -> tuple:
        """
        音频下载进度回调函数

        :param task_id: 任务ID
        :param file_url: 文件URL
        :param task_dir: 任务目录
        :return: 下载的音频文件路径和文件名
        :raises HTTPException: 如果任务已存在、下载失败或目录创建失败
        """
        async with get_db_async() as db:
            # 检查任务是否已存在
            existing_task = await get_task_async(db, task_id)
            if existing_task:
                # 清理任务目录（如果存在）
                shutil.rmtree(task_dir, ignore_errors=True)
                raise HTTPException(status_code=409, detail="任务已存在")

        # 创建任务目录（如果不存在）
        try:
            os.makedirs(task_dir, exist_ok=True)
        except Exception as e:
            await update_status(task_id, "error ", f"创建任务目录失败: {str(e)}", 100)
            raise HTTPException(status_code=500, detail=f"创建任务目录失败: {str(e)}")

        downloader = UCloudFileDownloader()
        download_path = None
        audio_filename = None

        try:
            # 解析文件名并验证格式
            parsed_url = urlparse(file_url)
            audio_filename = os.path.basename(parsed_url.path)
            audio_filename_lower = audio_filename.lower()

            if not audio_filename_lower.endswith(('.wav', '.mp3')):
                await update_status(task_id, "error ",
                                    f"不支持的音频格式，仅支持 .wav 或 .mp3", 100)
                raise HTTPException(status_code=400, detail="不支持的音频格式，仅支持 .wav 或 .mp3")

            download_path = os.path.join(task_dir, audio_filename)

            # 下载音频文件
            download_success = downloader.download_file(file_url, download_path)
            if not download_success:
                await update_status(task_id, "error ", "下载音频文件失败", 100)
                raise HTTPException(status_code=500, detail="从UCloud下载音频文件失败")

            # 创建任务记录
            async with get_db_async() as db:
                task_created = await create_task_async(db, {
                    "task_id": task_id,
                    "status": "pending",
                    "message": "音频文件已下载，等待处理",
                    "progress": 20,
                    "original_path": download_path,
                    "created_at": datetime.now().isoformat(),
                    "start_time": None
                })
                if not task_created:
                    await update_status(task_id, "error ", "创建任务记录失败", 100)
                    raise HTTPException(status_code=500, detail="创建任务记录失败")

            print(f"音频文件已下载到: {download_path}")
            return download_path, audio_filename

        except Exception as e:

            shutil.rmtree(task_dir, ignore_errors=True)
            await update_status(task_id, "error ", f"下载音频文件失败: {str(e)}", 100)
            raise HTTPException(status_code=500, detail=f"下载音频文件失败: {str(e)}")


audio_service = AudioService()

task_queue_manager = TaskQueueManager()
# asyncio.create_task(task_queue_manager.start())
