"""
@Project ：pythonProject 
@File    ：audio_service.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51 
"""
import bisect
import os

import torch
from faster_whisper import WhisperModel
from config import settings, hotword_list
from pyannote.audio import Pipeline
from pyannote.audio import Inference
from funasr import AutoModel
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class AudioService:
    def __init__(self):
        self.transcribe_para_former_model = None
        self.model = None

    def load_model(self):
        # self.model = WhisperModel(
        #     model_size_or_path=settings.model_path,
        #     device="cuda",
        #     compute_type="float16"
        # )

        # self.diarization_pipeline = Pipeline.from_pretrained(
        #     "pyannote/speaker-diarization-3.1",
        #     use_auth_token=os.getenv("HF_API_TOKEN")
        # ).to(torch.device("cuda"))
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

    def transcribe(self, file_path: str):
        diarization = self.diarization_pipeline(file_path)
        print("Diarization结果:", diarization)
        segments, _ = self.model.transcribe(
            file_path,
            beam_size=5,
            best_of=5,
            patience=1.5,
            length_penalty=1.2,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            initial_prompt="这是一段普通话的直播带货内容",
            language="zh",
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=150,
                threshold=0.5
            )
        )

        speaker_segments = list(diarization.itertracks(yield_label=True))
        speaker_starts = [spk_seg.start for (spk_seg, _, _) in speaker_segments]
        aligned = []
        for idx, seg in enumerate(segments):
            seg_start = seg.start
            seg_end = seg.end
            current_speakers = []

            start_idx = bisect.bisect_right(speaker_starts, seg_start) - 1
            end_idx = bisect.bisect_left(speaker_starts, seg_end)

            for i in range(max(0, start_idx), min(end_idx + 1, len(speaker_segments))):
                spk_seg, _, speaker = speaker_segments[i]

                overlap_start = max(seg_start, spk_seg.start)
                overlap_end = min(seg_end, spk_seg.end)
                overlap = overlap_end - overlap_start

                if overlap > 0:
                    current_speakers.append((speaker, overlap))
            speaker_label = "未知"
            if current_speakers:
                sorted_speakers = sorted(current_speakers, key=lambda x: x[1], reverse=True)
                total_overlap = sum(overlap for _, overlap in sorted_speakers)
                total_duration = seg_end - seg_start
                main_threshold = max(0.4, 0.7 - 0.1 * len(sorted_speakers))
                main_speaker = sorted_speakers[0]
                if (main_speaker[1] / total_duration) >= main_threshold:
                    speaker_label = main_speaker[0]
                else:
                    valid_speakers = [
                        (spk, ovlp)
                        for spk, ovlp in sorted_speakers
                        if ovlp / total_duration >= 0.2
                    ]
                    if valid_speakers:
                        total_valid = max(sum(ovlp for _, ovlp in valid_speakers), 1e-6)
                        contributions = []
                        for spk, ovlp in valid_speakers[:2]:  # 安全截断前两名
                            percent = (ovlp / total_valid) * 100
                            contributions.append(f"{spk}({percent:.0f}%)")
                        if contributions:
                            if len(contributions) > 1:
                                speaker_label = f"混合[{'+'.join(contributions)}]"
                            else:
                                speaker_label = contributions[0].split('(')[0]  # 提取SPEAKER_XX
                        else:
                            speaker_label = "未知"
                    else:
                        speaker_label = "未知"
            else:
                speaker_label = "未知"
            aligned.append({
                "index": idx,
                "start": seg_start,
                "end": seg_end,
                "suffix": file_path.rsplit("."),
                "text": seg.text.strip(),
                "speaker": speaker_label,
                "confidence": seg.avg_logprob  # 保留识别置信度
            })

        return aligned

    def transcribe_para_former(self, file_path: str):
        result = self.transcribe_para_former_model.generate(
            input=file_path,
            batch_size_s=1000,
            hotword=",".join(hotword_list),
            vad=True,
            punc=True,
            spk=True
        )

        formatted_results = []
        for idx, seg in enumerate(result[0]["sentence_info"]):
            formatted = {
                "index": idx,
                "start": seg["start"],
                "end": seg["end"],
                "suffix": os.path.splitext(file_path)[-1].lstrip('.'),
                "text": seg["text"].strip(),
                "speaker": seg.get("spk", "unknown")
            }
            formatted_results.append(formatted)

        print(formatted_results)
        return formatted_results


audio_service = AudioService()
