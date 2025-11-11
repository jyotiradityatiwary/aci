import sys
from typing import Callable, Any

import torch
from numpy.typing import NDArray
from torch._prims_common import DeviceLikeType
from transformers import (
    Wav2Vec2Processor,
    Speech2TextProcessor,
    Speech2TextForConditionalGeneration
)
from dataclasses import dataclass

from emotion_prediction import load_emotion_model, predict_from_signal
from emotion_prediction_dimensional import EmotionModel

class EmotionalState:
    def encode_as_dict(self) -> dict[str, float]:
        return {}

    def encode_for_model(self) -> str:
        return ''

class EmotionPredictionModel(Callable[[NDArray, int|float], EmotionalState]):
    required_sampling_rate: int | None = None

    def __init__(self, device: DeviceLikeType):
        self.device = device

    def __call__(self, signal: NDArray, sample_rate: int | float) -> EmotionalState:
        return EmotionalState()

@dataclass
class CategoricalEmotionalState(EmotionalState):
    predicted_emotion: str
    emotion_probabilities: dict[str, float]

    def encode_as_dict(self) -> dict[str, float]:
        return self.emotion_probabilities

    def encode_for_model(self) -> str:
        return self.predicted_emotion

class CategoricalEmotionPredictionModel(EmotionPredictionModel):
    required_sampling_rate = 48000

    def __init__(self, device: DeviceLikeType):
        super().__init__(device)
        self.model = load_emotion_model()

        if self.model is None:
            print("Exiting due to model loading failure.")
            exit()

    def __call__(
        self, signal: NDArray, sample_rate: int | float
    ) -> CategoricalEmotionalState:
        print("Running inference...")
        predicted_emotion, probabilities = predict_from_signal(signal, sample_rate, self.model, self.device)
        return CategoricalEmotionalState(
            predicted_emotion=predicted_emotion,
            emotion_probabilities=probabilities,
        )

@dataclass
class DimensionalEmotionalState:
    valence: float
    arousal: float
    dominance: float

    def encode_as_dict(self) -> dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }

    def encode_for_model(self) -> str:
        return f'valence={self.valence:.1f}, arousal={self.arousal:.1f}, dominance={self.dominance:.1f}'


class DimensionalEmotionPredictionModel(EmotionPredictionModel):
    required_sampling_rate = 16000

    emotion_recognition_model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'

    def __init__(self, device: DeviceLikeType):
        super().__init__(device)
        self.processor = Wav2Vec2Processor.from_pretrained(self.emotion_recognition_model_name)
        self.model = EmotionModel.from_pretrained(self.emotion_recognition_model_name).to(device)

    def __call__(
            self, signal: NDArray, sample_rate: int | float
    ) -> DimensionalEmotionalState:
        if sample_rate != self.required_sampling_rate:
            print(f"WARNING: sampling rate mismatch in {self.__class__.__name__}.", file=sys.stderr)

        processed_audio = self.processor(
            signal,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_values = processed_audio.input_values.to(self.device)

        # Run the model
        with torch.no_grad():
            logits = self.model(input_values)[1]  # Get the second return value (logits)

        scores = logits.detach().cpu().numpy()[0]
        valence, arousal, dominance = scores.flatten()
        return DimensionalEmotionalState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
        )

EMOTION_PREDICTION_MODEL_DICT = {
    "Disabled": EmotionPredictionModel,
    "Categorical": CategoricalEmotionPredictionModel,
    "Dimensional": DimensionalEmotionPredictionModel,
}