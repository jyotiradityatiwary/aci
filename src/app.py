from io import BytesIO

import streamlit as st

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


pipeline = get_pipeline()

verbose_output: bool = st.toggle("verbose output")


@st.cache_data(show_spinner="Processing your audio... 🎧")
def call_pipeline_with_cache(uploaded_file: BytesIO) -> PipelineResult:
    return pipeline(uploaded_file)


uploaded_file = st.file_uploader(
    "Upload an audio file to process", type=["mp3", "wav", "m4a", "ogg", ".flac"]
)

if uploaded_file is None:
    st.info("Please upload an audio file to get started.")
else:
    st.audio(uploaded_file, format="audio/wav")

    result = call_pipeline_with_cache(uploaded_file)

    if verbose_output:
        st.success("Processing finished!")
        st.subheader("Speech to Text Transcription:")
        st.write(result.speech_to_text_transcript)

        st.subheader("Speech Emotion Recognition Results")
        predicted_em = result.predicted_emotion
        confidence = result.emotion_probabilities[result.predicted_emotion]
        st.write(
            f"Detected emotion: {predicted_em} with confidence {confidence * 100:.2f}%"
        )
        st.bar_chart(result.emotion_probabilities)

        st.subheader("LLM Response Text")
        st.write(result.llm_response)

    st.subheader("Response")
    st.audio(result.audio_output, sample_rate=result.audio_output_sample_rate)
