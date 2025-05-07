"""
@Project ：pythonProject 
@File    ：audio_service.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:51 
"""

from config import settings, hotword_list

from funasr import AutoModel
import os

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
