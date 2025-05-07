import os
import numpy as np
from scipy.io import wavfile
import ffmpeg
import traceback
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm

import argparse


# This function is obtained from librosa.
def get_rms(
        y,
        frame_length=2048,
        hop_length=512,
        pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    """
    音频切分
    """

    def __init__(
            self,
            sr: int,
            threshold: float = -40.0,
            min_length: int = 5000,
            min_interval: int = 300,
            hop_size: int = 20,
            max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                   :, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)
                   ]
        else:
            return waveform[
                   begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)
                   ]

    # @timeit
    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                    i - silence_start >= self.min_interval
                    and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                      i - self.max_sil_kept: silence_start + self.max_sil_kept + 1
                      ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                        rms_list[
                        silence_start: silence_start + self.max_sil_kept + 1
                        ].argmin()
                        + silence_start
                )
                pos_r = (
                        rms_list[i - self.max_sil_kept: i + 1].argmin()
                        + i
                        - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                        rms_list[
                        silence_start: silence_start + self.max_sil_kept + 1
                        ].argmin()
                        + silence_start
                )
                pos_r = (
                        rms_list[i - self.max_sil_kept: i + 1].argmin()
                        + i
                        - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if (
                silence_start is not None
                and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        ####音频+起始时间+终止时间
        if len(sil_tags) == 0:
            return [[waveform, 0, int(total_frames * self.hop_size)]]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append([self._apply_slice(waveform, 0, sil_tags[0][0]), 0, int(sil_tags[0][0] * self.hop_size)])
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    [self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]),
                     int(sil_tags[i][1] * self.hop_size), int(sil_tags[i + 1][0] * self.hop_size)]
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    [self._apply_slice(waveform, sil_tags[-1][1], total_frames), int(sil_tags[-1][1] * self.hop_size),
                     int(total_frames * self.hop_size)]
                )
            return chunks


class Denoise:
    """音频降噪处理器"""

    def __init__(self, denoise_path='tools/denoise-model/speech_frcrn_ans_cirm_16k'):
        self.converter = AudioConverter(target_sr=16000)
        self.path_denoise = denoise_path if os.path.exists(denoise_path) else "damo/speech_frcrn_ans_cirm_16k"
        self.ans = pipeline(Tasks.acoustic_noise_suppression, model=self.path_denoise)

    def execute_single(self, input_file, output_file):
        """处理含格式转换的降噪流程"""
        # 强制输出为有效WAV格式


        # 转换输入文件
        converted_path = self.converter.convert_to_wav(input_file)

        try:
            # 读取转换后的音频数据
            with open(converted_path, 'rb') as f:
                audio_data = f.read()

            # 执行降噪处理
            result = self.ans(audio_data)

            # 获取音频参数（关键修改点）
            sample_rate = 16000  # 根据模型要求设置为16kHz
            bit_depth = 16  # 模型输出为16位PCM
            channels = 1  # 单声道

            # 将PCM数据转换为numpy数组
            pcm_data = np.frombuffer(result["output_pcm"], dtype=np.int16)

            # 使用scipy写入标准WAV文件（自动添加文件头）
            wavfile.write(
                output_file,
                sample_rate,
                pcm_data
            )

            # 二次验证文件有效性
            try:
                _ = wavfile.read(output_file)  # 尝试读取文件
            except Exception as e:
                raise RuntimeError(f"生成的文件无效: {str(e)}")

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"降噪处理失败: {str(e)}")

        finally:
            if os.path.exists(converted_path):
                os.remove(converted_path)


class AudioConverter:
    """通用音频格式转换器"""

    def __init__(self, target_sr=16000, target_channels=1):
        self.target_sr = target_sr
        self.target_channels = target_channels

    def convert_to_wav(self, input_path, output_path=None):
        """
        将任意音频转换为标准WAV格式
        :param input_path: 输入文件路径
        :param output_path: 输出路径（可选）
        :return: 转换后的临时文件路径
        """
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + '_converted.wav'

        try:
            (
                ffmpeg.input(input_path)
                .output(output_path,
                        acodec='pcm_s16le',
                        ar=str(self.target_sr),
                        ac=str(self.target_channels))
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
        except ffmpeg.Error as e:
            error_message = f"格式转换失败: {e.stderr.decode().strip()}"
            raise RuntimeError(error_message)

        return output_path


class AudioSlicer:
    """音频切片处理器"""

    def __init__(self, ):
        self.converter = AudioConverter(target_sr=32000)  # 切片使用32kHz
        self.slicer = Slicer(sr=32000, # 切片使用32kHz
                             threshold=-45, # 降低阈值
                             min_length=5000, # 降低最小长度
                             min_interval=300, # 降低最小间隔
                             hop_size=20, # 降低步长
                             max_sil_kept=800 # 降低最大静默保留长度
                             )

    def slice_audio(self, input_path, output_dir):
        """支持多格式的切片流程"""
        converted_path = self.converter.convert_to_wav(input_path)

        try:
            # 执行切片
            audio = self.load_audio(converted_path)
            base_name = os.path.splitext(os.path.basename(input_path))[0]

            for chunk_info in self.slicer.slice(audio):
                chunk, start_frame, end_frame = chunk_info
                self.process_chunk(chunk, base_name, start_frame, end_frame, output_dir)

        finally:
            if os.path.exists(converted_path):
                os.remove(converted_path)

    @staticmethod
    def load_audio(file):
        """加载音频文件"""
        file = clean_path(file)
        if not os.path.exists(file):
            raise FileNotFoundError(f"音频文件 {file} 不存在")
        try:
            out, _ = (ffmpeg.input(file, threads=0)
                      .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=32000)
                      .run(cmd=["ffmpeg", "-nostdin"],
                           capture_stdout=True,
                           capture_stderr=True))
            return np.frombuffer(out, np.float32).flatten()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"音频加载失败: {str(e)}")

    def process_chunk(self, chunk, base_name, start_frame, end_frame, output_dir):
        """处理并保存音频片段"""
        # 幅度归一化处理
        tmp_max = np.abs(chunk).max()
        if tmp_max > 1:
            chunk /= tmp_max

        # 混合处理
        _max = 0.9
        alpha = 0.25
        chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk

        # 计算时间戳
        start_time = int(start_frame * self.slicer.hop_size)
        end_time = int(end_frame * self.slicer.hop_size)

        # 保存文件
        filename = f"{os.path.splitext(base_name)[0]}_{start_time:010d}_{end_time:010d}.wav"
        wavfile.write(
            os.path.join(output_dir, filename),
            32000,
            (chunk * 32767).astype(np.int16)
        )


def process_audio(input_path, output_dir,
                  denoise_params=None, slicer_params=None):
    """
    音频处理总入口
    :param input_path: 输入音频路径
    :param output_dir: 输出目录
    :param denoise_params: 降噪参数字典
    :param slicer_params: 切片参数字典
    """
    # 初始化参数
    denoise_params = denoise_params or {}

    # 创建输出目录
    denoised_dir = os.path.join(output_dir, "denoised")
    sliced_dir = os.path.join(output_dir, "sliced")
    os.makedirs(denoised_dir, exist_ok=True)
    os.makedirs(sliced_dir, exist_ok=True)

    # 执行降噪
    denoiser = Denoise(**denoise_params)
    denoised_path = os.path.join(denoised_dir, os.path.basename(input_path))
    denoised_path = os.path.splitext(denoised_path)[0] + '.wav'
    denoiser.execute_single(input_path, denoised_path)

    # 验证降噪结果
    if not os.path.exists(denoised_path):
        raise RuntimeError("降噪处理失败，未生成输出文件")

    # 执行切片
    slicer = AudioSlicer()
    slicer.slice_audio(denoised_path, sliced_dir)


def clean_path(path_str: str):
    """路径清洗函数"""
    path_str = path_str.replace('/', os.sep).replace('\\', os.sep)
    path_str = path_str.strip(" \t\n\r\"'")
    return os.path.normpath(path_str)


if __name__ == "__main__":

    audio_input = "../temp/1111.mp4"
    audio_output = "../temp_audio_files/slicer_opt"
    try:
        process_audio(
            input_path=clean_path(audio_input),
            output_dir=clean_path(audio_output),
            slicer_params={
                'threshold': -40,
                'min_length': 3000
            }
        )
        print("处理完成，结果保存在:", os.path.abspath(audio_output))
    except Exception as e:
        print(f"处理失败: {str(e)}")
        traceback.print_exc()
