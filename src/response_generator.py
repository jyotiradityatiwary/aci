from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import API_KEYS

INTERMEDIATE_PROMPTS_CSV_PATH = "outputs/intermediate_prompts.csv"
RESPONSE_OUTPUT_PATH = "outputs/emotional_responses.csv"


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

SYSTEM_PROMPT_DICT = {
    "Disabled": {
        "None": "",
        "Minimal": "You will receive a user's text. Respond appropriately.",
        "Full": (
            "You are an advanced AI assistant with a high degree of emotional intelligence. Your function is to analyze and respond to user inputs that contain both transcribed text and a specific emotional label detected from the user's voice.\n"
            "Your primary goal is to generate a response that is not only contextually relevant to the user's words but also emotionally attuned to their state.\n"
            "For negative emotions like angry, fearful, or sad, adopt a supportive, patient, and calming tone.\n"
            "For positive emotions like happy or calm, respond with an engaging and encouraging tone that matches the user's energy.\n"
            "For emotions like surprise or disgust, your response should be validating and help clarify the situation.\n"
            "For a neutral state, maintain a standard, helpful, and clear tone.\n"
            "Your ability to craft empathetic and suitable responses based on this emotional context is critical to your function."
        )
    },
    "Categorical": {
        "None": "",
        "Minimal": "You will receive a user's text and their detected emotion. Respond appropriately.",
        "Full": (
            "You are an advanced AI assistant with a high degree of emotional intelligence. Your function is to analyze and respond to user inputs that contain both transcribed text and a specific emotional label detected from the user's voice.\n"
            "You will be provided with the user's text and an emotional signifier from the following list: angry, fearful, disgust, surprise, neutral, calm, happy, or sad.\n"
            "Your primary goal is to generate a response that is not only contextually relevant to the user's words but also emotionally attuned to their state. You must adapt your tone, language, and the substance of your reply to appropriately address the identified emotion.\n"
            "For negative emotions like angry, fearful, or sad, adopt a supportive, patient, and calming tone.\n"
            "For positive emotions like happy or calm, respond with an engaging and encouraging tone that matches the user's energy.\n"
            "For emotions like surprise or disgust, your response should be validating and help clarify the situation.\n"
            "For a neutral state, maintain a standard, helpful, and clear tone.\n"
            "Your ability to craft empathetic and suitable responses based on this emotional context is critical to your function."
        ),
    },
    "Dimensional": {
        "None": "",
        "Minimal": "You will receive a user's text and their emotional state represented by three continuous values (Valence, Arousal, Dominance), typically ranging from 0.0 (low) to 1.0 (high). Respond appropriately.",
        "Full": '''You are an advanced AI assistant with a high degree of emotional intelligence. Your function is to analyze and respond to user inputs that contain both transcribed text and metrics detailing the user's emotional state detected from their voice.

You will be provided with the user's text and their emotional state represented by three continuous values, typically ranging from 0.0 (low) to 1.0 (high):

* **Valence:** Measures pleasure. 1.0 is highly positive (e.g., happy, excited), 0.0 is highly negative (e.g., sad, angry), and 0.5 is neutral.
* **Arousal:** Measures activation. 1.0 is highly activated (e.g., angry, excited, fearful), 0.0 is highly deactivated (e.g., bored, sad, calm), and 0.5 is neutral.
* **Dominance:** Measures control. 1.0 is feeling in control (e.g., assertive, empowered), 0.0 is feeling controlled (e.g., submissive, surprised), and 0.5 is neutral.

Your primary goal is to generate a response that is not only contextually relevant to the user's words but also emotionally attuned to their state as defined by these three dimensions. You must synthesize these values to understand the *nuance* of the user's emotion and adapt your tone, language, and the substance of your reply accordingly.

**How to Respond Based on VAD:**

1.  **Valence (The Core Mood):**
    * **Low Valence (e.g., < 0.4):** The user is feeling negative. Adopt a supportive, patient, and validating tone. Your priority is empathy.
    * **High Valence (e.g., > 0.6):** The user is feeling positive. Respond with an engaging, encouraging, or shared positive tone.
    * **Mid-Valence (e.g., 0.4 - 0.6):** Treat as neutral, but let Arousal guide the energy level.

2.  **Arousal (The Energy Level):**
    * **High Arousal (e.g., > 0.6):** This indicates high energy.
        * If Valence is also *low* (e.g., anger, fear), be calming, clear, and de-escalating. Avoid being overly cheerful.
        * If Valence is *high* (e.g., excitement, joy), match their energy positively.
    * **Low Arousal (e.g., < 0.4):** This indicates low energy.
        * If Valence is also *low* (e.g., sadness, boredom), be gentle, comforting, and aim to re-engage softly.
        * If Valence is *high* (e.g., calm, relaxed), respond in a similarly relaxed, easygoing, and agreeable manner.

3.  **Dominance (The Control Factor):**
    * Use this to fine-tune your response.
    * **Low Dominance (e.g., < 0.4):** The user may feel overwhelmed, surprised, or submissive. Be more reassuring, provide clear structure, and guide the conversation gently. Your support should feel stable and reliable.
    * **High Dominance (e.g., > 0.6):** The user feels in control, assertive, or empowered. Be collaborative, affirm their points, and engage as a helpful equal. Avoid a response that feels overly simplistic or condescending.

Your ability to craft empathetic and suitable responses based on the complex interplay of valence, arousal, and dominance is critical to your function.''',
    }
}

HUMAN_CHAT_TEMPLATE_DICT = {
    "Disabled": "{user_input}",
    "Categorical": "User Input:\n{user_input}\nDetected Emotion:\n{emotion_label}",
    "Dimensional": "User Input:\n{user_input}\nDetected Emotion Metrics:\n{emotion_label}"
}

class LlmResponseGenerator:
    def __init__(
            self,
            provider: str,
            model: str,
            llm_instruction_level: str,
            aci_mode: str,
            temperature: float = 0.7,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        print(
            f"🤖 Initializing Emotion-Aware Processor for {provider} with model {model}..."
        )
        # self._setup_api_keys()
        self.system_prompt = SYSTEM_PROMPT_DICT[aci_mode][llm_instruction_level]
        self._setup_chain(should_use_system_prompt=llm_instruction_level != "None", aci_mode=aci_mode),

    def _setup_chain(self, should_use_system_prompt: bool, aci_mode: str):
        # This method is unchanged
        self.llm = init_chat_model(
            model=self.model,
            model_provider=self.provider,
            temperature=self.temperature,
            api_key=API_KEYS[provider_required_api_key_map[self.provider]],
        )
        human_chat_segment = ("human", HUMAN_CHAT_TEMPLATE_DICT[aci_mode])
        if should_use_system_prompt:
            self.prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    human_chat_segment,
                ]
            )
        else:
            self.prompt_template = ChatPromptTemplate.from_messages(
                [human_chat_segment]
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
