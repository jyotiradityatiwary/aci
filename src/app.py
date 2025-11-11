from io import BytesIO

import streamlit as st
import torch
from torch._prims_common import DeviceLikeType

from evaluator import Evaluator, EvaluatorResult
from pipeline import (
    EmotionPredictionModel,
    Pipeline,
    PipelineResult,
    SpeechToTextModel,
    TextToSpeechModel,
)
from response_generator import LlmResponseGenerator, SYSTEM_PROMPT_DICT
from utils import get_device

device = get_device()
if device == "cpu":
    st.warning("Using CPU for inference")

should_show_intermediate_steps: bool = st.toggle("Show intermediate steps")
should_evaluate: bool = st.toggle("Evaluate results against metrics")
_llm_instruction_level = st.segmented_control(
    label="Set instruction level given to LLM",
    options=SYSTEM_PROMPT_DICT.keys(),
    selection_mode='single',
    default="Full",
)


@st.cache_resource(show_spinner="Loading Speech to Text Model")
def get_speech_to_text_model() -> SpeechToTextModel:
    return SpeechToTextModel()


@st.cache_resource(show_spinner="Loading Emotion Prediction Model")
def get_emotion_prediction_model() -> EmotionPredictionModel:
    return EmotionPredictionModel(device=device)


@st.cache_resource(show_spinner="Loading Llm Response Generator")
def get_llm_response_generator(llm_instruction_level: str) -> LlmResponseGenerator:
    return LlmResponseGenerator(
        provider="google_genai",
        model="gemini-2.5-flash",
        system_prompt=SYSTEM_PROMPT_DICT[llm_instruction_level],
        should_use_system_prompt=llm_instruction_level != "None"
    )


@st.cache_resource(show_spinner="Loading Text to Speech Model")
def get_text_to_speech_model() -> TextToSpeechModel:
    return TextToSpeechModel(device=device)


@st.cache_resource(show_spinner='Initializing pipeline')
def get_pipeline(llm_instruction_level: str):
    return Pipeline(
        speech_to_text_model=get_speech_to_text_model(),
        emotion_prediction_model=get_emotion_prediction_model(),
        llm_response_generator=get_llm_response_generator(llm_instruction_level=llm_instruction_level),
        text_to_speech_model=get_text_to_speech_model(),
    )


@st.cache_resource(show_spinner="Initializing response evaluator")
def get_evaluator() -> Evaluator:
    return Evaluator()


pipeline = get_pipeline(llm_instruction_level=_llm_instruction_level)


@st.cache_resource(show_spinner="Processing your audio... 🎧")
def call_pipeline_with_cache(uploaded_file: BytesIO) -> PipelineResult:
    return pipeline(uploaded_file)


@st.cache_resource(show_spinner="Evaluating LLM output")
def get_evaluation_result(filename: str, response_text: str) -> EvaluatorResult:
    evaluator = get_evaluator()
    return evaluator(filename=filename, response_text=response_text)


uploaded_file = st.file_uploader(
    "Upload an audio file to process", type=["mp3", "wav", "m4a", "ogg", ".flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    result = call_pipeline_with_cache(uploaded_file)

    if should_show_intermediate_steps:
        st.subheader("Speech to Text Transcription:")
        st.write(result.speech_to_text_transcript)

        st.subheader("Speech Emotion Recognition Results")
        predicted_em = result.predicted_emotion
        confidence = result.emotion_probabilities[result.predicted_emotion]
        st.table(
            {
                "Detected Emotion": predicted_em,
                "Confidence": confidence,
            }
        )
        st.bar_chart(result.emotion_probabilities)

        st.subheader("LLM Response Text")
        st.write(result.llm_response)

    st.subheader("Response")
    st.audio(result.audio_output, sample_rate=result.audio_output_sample_rate)

    if should_evaluate:
        evaluator_result = get_evaluation_result(
            filename=uploaded_file.name,
            response_text=result.llm_response,
        )

        st.subheader("Evaluation Results")
        st.table(evaluator_result.as_flat_dict())
