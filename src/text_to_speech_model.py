import numpy as np
import torch
from numpy.typing import NDArray
from torch._prims_common import DeviceLikeType
from TTS.api import TTS
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsArgs, XttsAudioConfig, XttsConfig


class TextToSpeechModel:
    PRODUCED_AUDIO_SAMPLE_RATE = 24000

    def __init__(self, device: DeviceLikeType) -> None:
        with torch.serialization.safe_globals(
            [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
        ):
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    def __call__(self, text: str) -> tuple[NDArray, int]:
        return np.array(
            self.tts.tts(
                text=text,
                language="en",
                speaker="Ana Florence",
            )
        ), self.PRODUCED_AUDIO_SAMPLE_RATE
