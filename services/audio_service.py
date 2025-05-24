"""
@Project ：pythonProject
@File    ：audio_service.py.py
@IDE     ：PyCharm
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51
"""
import concurrent
import threading
from datetime import datetime
import glob
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from modelscope import pipeline, Tasks
from pydub import AudioSegment
from config import hotword_list, TEMP_DIR, pipeline_queue
import time
# from funasr import AutoModel
import os
import subprocess
import time
from models.schemas import Segment
from services.oss_service import oss_service
from utils.mysql_db import update_task_status, get_db_connection
import asyncio

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# # 定义最大并发数和全局锁
# MAX_CONCURRENT_TASKS = 20
# TRANSCRIBE_LOCK = asyncio.Lock()

# 使用信号量限制总并发数量
# semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
# ------------------------新解决方案----------------------

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


def update_status(task_id, status: str, message: str, progress: int, complete_time=None):
    """
    更新任务状态
    :param task_id: 任务ID
    :param status: 状态
    :param message: 消息
    :param progress: 进度
    :return: None
    """
    if complete_time:
        update_task_status({
            "task_id": task_id,
            "status": status,
            "message": message,
            "progress": progress,
            "complete_time": complete_time
        })
    update_task_status({
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
    update_status(task_id, "processing", f"保存片段: {message}", 60 + int(progress * 0.3))


def merge_progress_callback(task_id, progress, message):
    """
    合并进度回调函数
    :param task_id: 任务ID
    :param progress: 进度百分比
    :param message: 进度消息
    :return: None
    """
    update_status(task_id, "processing", f"合并片段: {message}", 30 + int(progress * 0.3))


async def process_audio_task(task_id: str, original_path: str, original_ext: str, min_chunk_duration: float):
    task_dir = os.path.join(TEMP_DIR, task_id)
    start_time = time.time()  # 开始计时
    audio_service = pipeline_queue.get()
    try:
        update_status(task_id, "processing", "开始处理音频文件", 0)
        stage_start_time = time.time()
        if original_ext.lower() != '.wav':
            update_status(task_id, "processing", "正在转换音频格式", 10)
            wav_path = os.path.join(task_dir, "audio.wav")
            convert_to_wav(original_path, wav_path)
            processing_path = wav_path
        else:
            processing_path = original_path
            update_status(task_id, "processing", "音频格式无需转换", 10)
        print(f"Task {task_id} - 音频格式处理耗时: {time.time() - stage_start_time:.2f}秒")
        update_status(task_id, "processing", "开始语音识别", 20)
        stage_start_time = time.time()
        result = audio_service.transcribe_para_former(processing_path)
        print(f"Task {task_id} - 语音识别耗时: {time.time() - stage_start_time:.2f}秒")
        # 发音人合并
        update_status(task_id, "processing", "合并发音人信息", 30)
        stage_start_time = time.time()
        raw_segments = result[0]
        raw_segments = raw_segments['sentence_info']
        merged_segments = audio_service.merge_segments(
            task_id,
            raw_segments,
            min_chunk_duration,
            merge_progress_callback
        )
        print(f"Task {task_id} - 合并发音人信息耗时: {time.time() - stage_start_time:.2f}秒")

        update_status(task_id, "processing", "格式化识别结果", 60)
        stage_start_time = time.time()

        # 保存所有分片到本地
        segments_paths = audio_service.split_segments(
            merged_segments,
            processing_path,
            task_id,
            split_progress_callback
        )
        print(f"Task {task_id} - 分割音频片段耗时: {time.time() - stage_start_time:.2f}秒")

        # 结果保存阶段
        update_status(task_id, "processing", "保存识别结果", 95)
        stage_start_time = time.time()
        from config import base_url
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                for idx, segment_path, merged_seg in segments_paths:
                    cursor.execute('''
                        INSERT INTO ai_task_results (task_id, `index`, start, `end`, text, speaker, `url`)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        task_id,
                        idx,
                        merged_seg.get("start"),
                        merged_seg.get("end"),
                        merged_seg.get("text"),
                        str(merged_seg.get("spk")),
                        base_url + segment_path.replace("\\", "/")
                    ))
                conn.commit()
        print(f"Task {task_id} - 保存识别结果耗时: {time.time() - stage_start_time:.2f}秒")

        update_status(task_id, "completed", "处理完成", 100, complete_time=datetime.now().isoformat())
        print(f"Task {task_id} - 总耗时: {time.time() - start_time:.2f}秒")
    finally:
        pipeline_queue.put(audio_service)


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


def load_segments_if_completed(conn, task_id):
    """
    检查任务是否已完成，如果完成则加载结果
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM ai_task_results WHERE task_id = %s ORDER BY `index`", (task_id,))
            results = cursor.fetchall()
        return [Segment(**val) for val in results]
    except Exception as e:
        return f"Error loading segments: {str(e)}"


def format_task_merged_filename(task_id: str, index: int, suffix: str = "mp3"):
    """格式化合并后的文件名"""
    filename = f"merged_{task_id}_{index:05d}.{suffix}"
    return filename


def extract_index_from_filename(filename):
    """
    从文件名中提取索引
    """
    match = re.search(r'merged_.*?_(\d{5})\.mp3$', filename)
    if match:
        return int(match.group(1))
    return None


class AudioService:
    def __init__(self):
        self.transcribe_para_former_model = None
        self.model = None
        self.inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="pre_model/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="pre_model/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="pre_model/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="pre_model/speech_campplus_sv_zh-cn_16k-common",
            disable_update=True,
        )

    def transcribe_para_former(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        asr_result = self.inference_pipeline(input=file_path, batch_size_token=4000,
                                             batch_size_token_threshold_s=30,
                                             max_single_segment_time=5000)
        return asr_result

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

    def _save_merged_segment(self, audio, duration_ms, original_path: str, start: float, end: float,
                             index: int, task_id: str) -> str:
        """保存合并后的长片段（修复变量引用问题）"""
        try:
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"音频文件不存在: {original_path}")
            # audio = AudioSegment.from_file(original_path)
            # duration_ms = len(audio)
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
                segments_paths.append((idx, segment_path, merged_seg))
                progress_callback(task_id, int(((idx + 1) / total_segments) * 100), f"已保存 {idx + 1} 个片段")
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
