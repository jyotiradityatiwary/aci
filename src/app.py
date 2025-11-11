from io import BytesIO

import streamlit as st
import torch

from evaluator import Evaluator, EvaluatorResult
from pipeline import (
    EmotionPredictionModel,
    Pipeline,
    PipelineResult,
    SpeechToTextModel,
    TextToSpeechModel,
)
from response_generator import LlmResponseGenerator
from utils import get_device


@st.cache_resource(show_spinner=False)
def get_pipeline():
    device = get_device()
    if device == "cpu":
        st.warning("Using CPU for inference")
    with st.spinner("Loading Speech to Text model"):
        speech_to_text_model = SpeechToTextModel()
    with st.spinner("Loading Emotion Prediction model"):
        emotion_prediction_model = EmotionPredictionModel(device)
    with st.spinner("Loading Llm Response Generator"):
        llm_response_generator = LlmResponseGenerator(
            provider="google_genai", model="gemini-2.5-flash"
        )
    with st.spinner("Loading Text to Speech model"):
        text_to_speech_model = TextToSpeechModel(device)
    with st.spinner("Initializing pipeline"):
        pipeline = Pipeline(
            speech_to_text_model=speech_to_text_model,
            emotion_prediction_model=emotion_prediction_model,
            llm_response_generator=llm_response_generator,
            text_to_speech_model=text_to_speech_model,
        )
    return pipeline


@st.cache_resource(show_spinner="Initializing response evaluator")
def get_evaluator() -> Evaluator:
    return Evaluator()


pipeline = get_pipeline()

should_show_intermediate_steps: bool = st.toggle("Show intermediate steps")
should_evaluate: bool = st.toggle("Evaluate results against metrics")


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
