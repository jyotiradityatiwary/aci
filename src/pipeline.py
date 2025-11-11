from dataclasses import dataclass
from io import BytesIO

import librosa
from numpy.typing import NDArray

from emotion_prediction_model import EmotionPredictionModel
from response_generator import LlmResponseGenerator
from speech_to_text_model import SpeechToTextModel
from text_to_speech_model import TextToSpeechModel


@dataclass
class PipelineResult:
    speech_to_text_transcript: str
    predicted_emotion: str
    emotion_probabilities: dict[str, float]
    llm_response: str
    audio_output: NDArray
    audio_output_sample_rate: int


@dataclass
class Pipeline:
    speech_to_text_model: SpeechToTextModel
    emotion_prediction_model: EmotionPredictionModel
    llm_response_generator: LlmResponseGenerator
    text_to_speech_model: TextToSpeechModel

    def __call__(self, file: BytesIO) -> PipelineResult:
        signal, sample_rate = librosa.load(
            file, sr=SpeechToTextModel.required_sampling_rate
        )
        speech_to_text_transcript = self.speech_to_text_model(signal, sample_rate)
        if sample_rate != EmotionPredictionModel.required_sampling_rate:
            print(
                "Reloading audio with required sampling rate for Emotion Prediction Model"
            )
            file.seek(0)
            signal, sample_rate = librosa.load(
                file, sr=EmotionPredictionModel.required_sampling_rate
            )
        predicted_emotion, emotion_probabilities = self.emotion_prediction_model(
            signal, sample_rate
        )
        llm_response = self.llm_response_generator(
            user_input=speech_to_text_transcript, emotion_label=predicted_emotion
        )
        audio_output, audio_output_sample_rate = self.text_to_speech_model(llm_response)
        return PipelineResult(
            speech_to_text_transcript=speech_to_text_transcript,
            predicted_emotion=predicted_emotion,
            emotion_probabilities=emotion_probabilities,
            llm_response=llm_response,
            audio_output=audio_output,
            audio_output_sample_rate=audio_output_sample_rate,
        )
