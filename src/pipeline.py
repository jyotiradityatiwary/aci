from dataclasses import dataclass
from io import BytesIO

import librosa
from numpy.typing import NDArray

from emotion_prediction_model import EmotionPredictionModel
from response_generator import LlmResponseGenerator
from speech_to_text_model import SpeechToTextModel
from emotion_prediction_model import EmotionalState
from text_to_speech_model import TextToSpeechModel


@dataclass
class PipelineResult:
    speech_to_text_transcript: str
    predicted_emotional_state: EmotionalState
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
        # load audio
        signal, sample_rate = librosa.load(
            file, sr=self.speech_to_text_model.required_sampling_rate
        )

        # Speech to Text
        speech_to_text_transcript = self.speech_to_text_model(signal, sample_rate)

        # Emotional State Detection
        required_sampling_rate = self.emotion_prediction_model.required_sampling_rate
        if required_sampling_rate is not None and required_sampling_rate != sample_rate:
            print(
                "Reloading audio with required sampling rate for Emotion Prediction Model"
            )
            file.seek(0)
            signal, sample_rate = librosa.load(
                file, sr=EmotionPredictionModel.required_sampling_rate
            )
        predicted_emotional_state = self.emotion_prediction_model(
            signal, sample_rate
        )

        # LLM Response (text)
        llm_response = self.llm_response_generator(
            user_input=speech_to_text_transcript, emotion_label=predicted_emotional_state.encode_for_model()
        )

        # Response text to speech
        audio_output, audio_output_sample_rate = self.text_to_speech_model(llm_response)

        # Return final result
        return PipelineResult(
            speech_to_text_transcript=speech_to_text_transcript,
            predicted_emotional_state=predicted_emotional_state,
            llm_response=llm_response,
            audio_output=audio_output,
            audio_output_sample_rate=audio_output_sample_rate,
        )
