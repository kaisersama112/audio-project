"""
@Project ：pythonProject
@File    ：audio_service.py.py
@IDE     ：PyCharm
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from pydub import AudioSegment

from config import settings, hotword_list

from funasr import AutoModel
import os

from services.oss_service import oss_service

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class AudioService:
    def __init__(self):
        self.transcribe_para_former_model = None
        self.model = None

    def load_model(self):
        self.transcribe_para_former_model = AutoModel(
            model="paraformer-zh",
            model_revision="v2.0.4",
            vad_model="fsmn-vad",
            vad_model_revision="v2.0.4",
            punc_model="ct-punc-c",
            punc_model_revision="v2.0.4",
            spk_model="cam++",
            spk_model_revision="v2.0.2"
        )
        print("Models loaded successfully")

    def _save_segment(self, original_path: str, seg: dict, index: int, task_id: str) -> str:
        """保存音频片段到临时文件"""
        audio = AudioSegment.from_file(original_path)
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        segment = audio[start_ms:end_ms]

        # 创建存储目录
        output_dir = os.path.join(os.path.dirname(original_path), "segments")
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        filename = f"seg_{task_id}_{index:04d}.mp3"
        output_path = os.path.join(output_dir, filename)

        # 导出文件
        segment.export(output_path,
                       format="mp3",
                       bitrate="192k",
                       tags={
                           'title': f"Segment {index}",
                           'artist': 'Audio Processing System'
                       })

        return output_path

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

    def merge_segments(self, raw_segments: list, min_chunk_duration: float) -> list:
        """合并连续相同说话人的片段，基于最小时长要求"""
        if not raw_segments:
            return []
        merged = []
        current = {
            "start": raw_segments[0]["start"],
            "end": raw_segments[0]["end"],
            "text": raw_segments[0]["text"].strip(),
            "spk": raw_segments[0].get("spk", "unknown"),
            "count": 1  # 添加计数器
        }
        for seg in raw_segments[1:]:
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
        return merged
    def _save_merged_segment(self, original_path: str, start: float, end: float,
                             index: int, task_id: str) -> str:
        """保存合并后的长片段（修复变量引用问题）"""
        output_path = None  # 提前初始化变量
        try:
            # 1. 校验输入参数
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"音频文件不存在: {original_path}")

            # 2. 加载音频文件
            audio = AudioSegment.from_file(original_path)

            duration_ms = len(audio)  # 音频总时长（毫秒）

            # 3. 校验时间范围有效性
            start_ms = int(start )
            end_ms = int(end )
            if start_ms < 0 or end_ms > duration_ms:
                raise ValueError(
                    f"时间范围超出音频边界: 0-{duration_ms / 1000:.2f}s "
                    f"(请求范围: {start:.2f}-{end:.2f}s)"
                )

            # 4. 切割音频片段
            segment = audio[start_ms:end_ms]

            # 5. 准备输出路径
            output_dir = os.path.join(os.path.dirname(original_path), "merged_segments")
            os.makedirs(output_dir, exist_ok=True)

            filename = f"merged_{task_id}_{index:03d}.mp3"
            output_path = os.path.join(output_dir, filename)  # 明确赋值位置

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

            # 7. 二次校验文件有效性
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
                raise IOError("生成的音频文件无效或为空")

            return output_path

        except Exception as e:
            # 增强错误信息
            error_context = (
                f"[文件: {original_path}] "
                f"[时间范围: {start:.2f}s-{end:.2f}s] "
                f"[任务ID: {task_id}] "
                f"[索引: {index}]"
            )
            raise RuntimeError(f"合并片段保存失败: {error_context} → {str(e)}") from e

    def transcribe_para_former(self, file_path: str):

        result = self.transcribe_para_former_model.generate(
            input=file_path,
            batch_size_s=2000,
            hotword=",".join(hotword_list),
            vad=True,
            punc=True,
            spk=True,
            precision="float16"
        )
        return result
    def formatted_results_upload(self,merged_segments,file_path,task_id):
        formatted_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for idx, merged_seg in enumerate(merged_segments):
                try:
                    segment_path = self._save_merged_segment(
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


audio_service = AudioService()
