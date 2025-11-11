from numpy.typing import NDArray
from torch._prims_common import DeviceLikeType

from emotion_prediction import load_emotion_model, predict_from_signal


class EmotionPredictionModel:
    required_sampling_rate = 48000

    def __init__(self, device: DeviceLikeType):
        self.device = device
        self.model = load_emotion_model()

        if self.model is None:
            print("Exiting due to model loading failure.")
            exit()

    def __call__(
        self, signal: NDArray, sample_rate: int | float
    ) -> tuple[str, dict[str, float]]:
        print("Running inference...")
        return predict_from_signal(signal, sample_rate, self.model, self.device)
