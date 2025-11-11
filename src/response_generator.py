from dataclasses import dataclass

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import API_KEYS, TARGET_RESPONSES_DIR

INTERMEDIATE_PROMPTS_CSV_PATH = "outputs/intermediate_prompts.csv"
RESPONSE_OUTPUT_PATH = "outputs/emotional_responses.csv"


class CorrectAnswers:
    def __init__(self):
        self._df = self._build_df()

    @staticmethod
    def _build_df() -> pd.DataFrame:
        content = []
        with open(TARGET_RESPONSES_DIR / "1.txt") as f:
            content.append(f.read().strip())
        with open(TARGET_RESPONSES_DIR / "2.txt") as f:
            content.append(f.read().strip())
        content = "\n\n".join(content)
        content = [p.split("\n")[4] for p in content.split("\n\n")]

        df = pd.DataFrame(
            {
                "correct_answer": content,
                "emotion": [
                    "angry",
                    "fearful",
                    "disgust",
                    "surprise",
                    "neutral",
                    "calm",
                    "happy",
                    "sad",
                ]
                * 2
                * 2,
                "statement": (
                    ["Kids are talking by the door"] * 8
                    + ["Dogs are sitting by the door"] * 8
                )
                * 2,
            }
        )

        df.set_index(["emotion", "statement"])
        df.sort_index()
        return df

    def get(self, emotion: str, statement: str):
        """Todo: fix fear-fearful datapoint"""
        if emotion == "fear":
            emotion = "fearful"
        return self._df.loc[(emotion, statement)]["correct_answer"].sample(n=1).iloc[0]


@dataclass
class LlModel:
    model_name: str
    provider_name: str


ll_models = [
    LlModel(model_name="gpt-4o", provider_name="openai"),
    LlModel(model_name="gemini-2.5-flash", provider_name="google_genai"),
    LlModel(model_name="gemini-1.5-pro", provider_name="google_genai"),
    LlModel(model_name="mistral-large-latest", provider_name="mistralai"),
]


provider_required_api_key_map = {
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
}


class LlmResponseGenerator:
    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider
        self.model = model
        self.kwargs = kwargs
        print(
            f"🤖 Initializing Emotion-Aware Processor for {provider} with model {model}..."
        )
        # self._setup_api_keys()
        self.system_prompt = (
            "You are an advanced AI assistant with a high degree of emotional intelligence. Your function is to analyze and respond to user inputs that contain both transcribed text and a specific emotional label detected from the user's voice.\n"
            "You will be provided with the user's text and an emotional signifier from the following list: angry, fearful, disgust, surprise, neutral, calm, happy, or sad.\n"
            "Your primary goal is to generate a response that is not only contextually relevant to the user's words but also emotionally attuned to their state. You must adapt your tone, language, and the substance of your reply to appropriately address the identified emotion.\n"
            "For negative emotions like angry, fearful, or sad, adopt a supportive, patient, and calming tone.\n"
            "For positive emotions like happy or calm, respond with an engaging and encouraging tone that matches the user's energy.\n"
            "For emotions like surprise or disgust, your response should be validating and help clarify the situation.\n"
            "For a neutral state, maintain a standard, helpful, and clear tone.\n"
            "Your ability to craft empathetic and suitable responses based on this emotional context is critical to your function."
        )
        self._setup_chain()

    def _setup_chain(self):
        # This method is unchanged
        self.llm = init_chat_model(
            model=self.model,
            model_provider=self.provider,
            temperature=self.kwargs.get("temperature", 0.7),
            api_key=API_KEYS[provider_required_api_key_map[self.provider]],
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "User Input:\n{user_input}\nDetected Emotion:\n{emotion_label}",
                ),
            ]
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        print("✅ AI Chain initialized with emotional intelligence.")

    def __call__(
        self,
        user_input: str,
        emotion_label: str,
    ) -> str:
        try:
            return self.chain.invoke(
                {"user_input": user_input, "emotion_label": emotion_label}
            )
        except Exception as e:
            return f"ERROR: {e}"
