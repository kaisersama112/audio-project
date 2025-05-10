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
            spk_model_revision="v2.0.2",
            batch_size=4
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

    def _merge_segments(self, raw_segments: list) -> list:
        """合并连续相同说话人的片段"""
        if not raw_segments:
            return []

        merged = []
        current = {
            "start": raw_segments[0]["start"],
            "end": raw_segments[0]["end"],
            "text": raw_segments[0]["text"].strip(),
            "spk": raw_segments[0].get("spk", "unknown")
        }

        for seg in raw_segments[1:]:
            # 合并条件：相同说话人且间隔小于0.5秒
            if (seg.get("spk") == current["spk"] and
                    seg["start"] - current["end"] <= 0.5):
                current["end"] = seg["end"]
                current["text"] += " " + seg["text"].strip()
            else:
                merged.append(current)
                current = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "spk": seg.get("spk", "unknown")
                }

        merged.append(current)
        return merged

    def _save_merged_segment(self, original_path: str, start: float, end: float,
                             index: int, task_id: str) -> str:
        """保存合并后的长片段"""
        try:
            audio = AudioSegment.from_file(original_path)
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            segment = audio[start_ms:end_ms]

            output_dir = os.path.join(os.path.dirname(original_path), "merged_segments")
            os.makedirs(output_dir, exist_ok=True)

            filename = f"merged_{task_id}_{index:03d}.mp3"
            output_path = os.path.join(output_dir, filename)

            segment.export(output_path,
                           format="mp3",
                           bitrate="192k",
                           tags={
                               'title': f"Merged {index}",
                               'artist': 'Audio Processing System',
                               'comment': f"Original: {start:.2f}-{end:.2f}s"
                           })
            return output_path
        except Exception as e:
            raise RuntimeError(f"合并片段保存失败: {str(e)}")

    def transcribe_para_former(self, file_path: str, task_id: str):
        result = self.transcribe_para_former_model.generate(
            input=file_path,
            batch_size_s=1000,
            hotword=",".join(hotword_list),
            vad=True,
            punc=True,
            spk=True
        )

        raw_segments = result[0]["sentence_info"]
        merged_segments = self._merge_segments(raw_segments)

        formatted_results = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []

            # 处理合并后的片段
            for idx, merged_seg in enumerate(merged_segments):
                try:
                    # 生成合并后的音频文件路径
                    segment_path = self._save_merged_segment(
                        original_path=file_path,
                        start=merged_seg["start"],
                        end=merged_seg["end"],
                        index=idx,
                        task_id=task_id
                    )

                    # 提交上传任务
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
