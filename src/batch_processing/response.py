import csv
import getpass
import os
from time import sleep
from typing import Dict, List

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import INTERMEDIATE_PROMPTS_CSV_PATH, RESPONSE_OUTPUT_PATH

"""## Prepare data

TODO: Optimize using Multi-Index
"""


def prepare_correct_answers():
    with open("/content/drive/MyDrive/colab/results_all_1.txt") as f:
        content = f.read().strip()
    with open("/content/drive/MyDrive/colab/results_all_2.txt") as f:
        content += "\n\n" + f.read().strip()
    content = [p.split("\n")[4] for p in content.split("\n\n")]

    return pd.DataFrame(
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


correct_answers = prepare_correct_answers()
correct_answers.head()

"""Todo: fix fear-fearful datapoint"""


def get_correct_answer(emotion: str, statement: str):
    if emotion == "fear":
        emotion = "fearful"
    return (
        correct_answers[
            (correct_answers["emotion"] == emotion)
            & (correct_answers["statement"] == statement)
        ]["correct_answer"]
        .sample(n=1)
        .iloc[0]
    )


get_correct_answer("fearful", "Kids are talking by the door")

# random sample N data points
n_data_points = 100

model_predictions = pd.read_csv(
    "/content/drive/MyDrive/colab/model_predictions.csv"
).sample(n=n_data_points)
model_predictions.head()

"""Prepare prompts CSV in the format required by `ColabEmotionalProcessor`"""


def prepare_prompts_csv(path: str):
    pd.DataFrame(
        {
            "correct_answer": [
                get_correct_answer(emotion, statement)
                for (emotion, statement) in zip(
                    model_predictions["correct_emotion"], model_predictions["statement"]
                )
            ],
            "emotion_label": model_predictions["predicted_emotion"],
            "user_input": model_predictions["statement"],
        }
    ).to_csv(path)


prepare_prompts_csv(INTERMEDIATE_PROMPTS_CSV_PATH)


"""## LLM Calling

### The Core Logic: `ColabEmotionalProcessor` Class

This class contains all the functions for handling prompts,
interacting with the LLM, and managing the workflow using `CSV` files.
"""


class ColabEmotionalProcessor:
    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider
        self.model = model
        self.kwargs = kwargs
        print(
            f"🤖 Initializing Emotion-Aware Processor for {provider} with model {model}..."
        )
        self._setup_api_keys()
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

    def _setup_api_keys(self):
        # This method is unchanged
        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "google_genai": "GOOGLE_API_KEY",
            "mistralai": "MISTRAL_API_KEY",
        }
        required_key = api_key_map.get(self.provider)
        if not required_key:
            return
        if not os.getenv(required_key):
            os.environ[required_key] = getpass.getpass(f"Enter your {required_key}: ")

    def _setup_chain(self):
        # This method is unchanged
        self.llm = init_chat_model(
            model=self.model,
            model_provider=self.provider,
            temperature=self.kwargs.get("temperature", 0.7),
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

    def read_prompts_from_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parses a CSV file with 'user_input', 'emotion_label', and 'correct_answer' columns.
        """
        try:
            df = pd.read_csv(file_path)
            # Ensure the necessary columns are present
            required_cols = {"user_input", "emotion_label", "correct_answer"}
            if not required_cols.issubset(df.columns):
                print(
                    f"❌ Error: Your CSV must contain the columns: {', '.join(required_cols)}"
                )
                return []

            # Convert DataFrame to a list of dictionaries
            prompts = df.to_dict("records")
            print(f"✅ Parsed {len(prompts)} prompts from '{file_path}'.")
            return prompts
        except FileNotFoundError:
            print(f"❌ Error: The file '{file_path}' was not found.")
            return []
        except Exception as e:
            print(f"❌ An error occurred while reading the CSV file: {e}")
            return []

    def process_prompts(
        self, prompts: List[Dict[str, str]], use_batch: bool
    ) -> List[str]:
        # This method is unchanged
        if use_batch:
            print("📦 Starting batch processing...")
            return self.chain.batch(prompts)

        SEQUENTIAL_PROCESSING_DELAY: float = 6  # seconds
        print("🔄 Starting sequential processing...")
        print(f"Delay between requests = {SEQUENTIAL_PROCESSING_DELAY} seconds")
        responses = []
        for i, p in enumerate(prompts, 1):
            print(f"  - Processing prompt {i}/{len(prompts)}...")
            try:
                responses.append(self.chain.invoke(p))
            except Exception as e:
                responses.append(f"ERROR: {e}")
            sleep(SEQUENTIAL_PROCESSING_DELAY)
        return responses

    def run_workflow(self, use_batch: bool):
        """
        Runs the full workflow: upload CSV, process prompts, and save results to a new CSV.
        """
        # print("\n📁 Please upload your prompts CSV file.")
        # uploaded = files.upload()
        # if not uploaded:
        #     print("❌ No file uploaded.")
        #     return
        #
        # input_file = list(uploaded.keys())[0]
        # if not input_file.lower().endswith('.csv'):
        #     print(f"❌ Error: Please upload a CSV file. You uploaded '{input_file}'.")
        #     return
        input_file = INTERMEDIATE_PROMPTS_CSV_PATH
        prompts = self.read_prompts_from_file(input_file)
        if not prompts:
            print("❌ Workflow halted: No valid prompts were parsed from the file.")
            return

        # Get AI responses
        responses = self.process_prompts(prompts, use_batch)

        # Write the results to a new CSV file
        output_filename = RESPONSE_OUTPUT_PATH
        try:
            with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
                # Define the column headers for the output file
                fieldnames = [
                    "user_input",
                    "emotion_label",
                    "ai_response",
                    "correct_answer",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for i, response in enumerate(responses):
                    writer.writerow(
                        {
                            "user_input": prompts[i]["user_input"],
                            "emotion_label": prompts[i]["emotion_label"],
                            "ai_response": response,
                            "correct_answer": prompts[i]["correct_answer"],
                        }
                    )

            print(f"\n🎉 Success! Results saved to '{output_filename}'.")
        except Exception as e:
            print(f"❌ An error occurred while writing the output CSV file: {e}")


print("✅ Main Processor Class defined for CSV input and CSV output.")

# --- 5. The "Quick Start" Function ---
# This function provides the interactive menu for easy use.


def quick_setup():
    print("\n" + "=" * 50 + "\n🎯 Emotion-Aware Batch LLM Processor Setup\n" + "=" * 50)
    providers = {
        "1": ("openai", "gpt-4o"),
        "2": ("google_genai", "gemini-2.5-flash"),
        "3": ("google_genai", "gemini-1.5-pro"),
        "4": ("mistralai", "mistral-large-latest"),
    }
    for k, (p, m) in providers.items():
        print(f"  {k}. {p.replace('_', ' ').title()} - {m}")
    choice = input(f"Enter your choice (1-{len(providers)}): ").strip()
    provider, model = providers.get(choice, providers["1"])
    use_batch = input(
        "Use batch processing for speed? (y/n, default y): "
    ).lower() not in ["n", "no"]
    try:
        processor = ColabEmotionalProcessor(provider=provider, model=model)
        processor.run_workflow(use_batch=use_batch)
    except Exception as e:
        print(f"❌ Setup failed: {e}")


print("✅ Interactive setup function defined.")

# --- 6. Run Everything! ---
# This will start the interactive setup, which will then guide you
# through uploading your file and processing it.

quick_setup()

"""## BERT AND BLEU SCORE"""

# Step 2: Import libraries
import io

import bert_score
import pandas as pd

# CORRECTED: Import SmoothingFunction as well
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

print("\nLibraries installed and imported.")


# The 'uploaded' object is a dictionary where keys are filenames and values are the file content
input_filename = RESPONSE_OUTPUT_PATH
output_filename = "scored_" + input_filename

print(f"\nProcessing '{input_filename}'...")

# Step 4: Load the CSV data directly from the uploaded content
# We use io.BytesIO to read the file content as if it were a file on disk
df = pd.read_csv(INTERMEDIATE_PROMPTS_CSV_PATH)
print(f"Successfully loaded {len(df)} rows.")

# Step 5: Calculate BLEU score for each row
print("Calculating BLEU scores...")

# CORRECTED: Instantiate the SmoothingFunction
smoother = SmoothingFunction()


def calculate_bleu(row):
    # Ensure that the inputs are strings before splitting
    ref_text = str(row.get("correct_answer", ""))
    cand_text = str(row.get("ai_response", ""))

    reference = [ref_text.split()]
    candidate = cand_text.split()

    return sentence_bleu(
        reference,
        candidate,
        smoothing_function=smoother.method4,  # A good general-purpose smoothing
    )


df["BLEU_score"] = df.apply(calculate_bleu, axis=1)

# Step 6: Calculate BERT scores in a single batch
print("Calculating BERT scores... This may take a moment.")

# Ensure all items in the list are strings to avoid errors
candidates = df["ai_response"].astype(str).tolist()
references = df["correct_answer"].astype(str).tolist()

P, R, F1 = bert_score.score(
    candidates,
    references,
    lang="en",
    verbose=True,  # Shows a progress bar
)

# Step 7: Add the BERT scores to the DataFrame
df["BERT_F1"] = F1.numpy()
df["BERT_Precision"] = P.numpy()
df["BERT_Recall"] = R.numpy()

# Step 8: Save the results to a new CSV in memory
# and then trigger a download for the user
print(f"\nScoring complete. Preparing '{output_filename}' for download.")
df.to_csv(output_filename, index=False)


print(f"Mean BLEU score: {df.BLEU_score.mean()}")
print(f"Mean BERT F1 score: {df.BERT_F1.mean()}")
print(f"Mean BERT Precision: {df.BERT_Precision.mean()}")
print(f"Mean BERT Recall: {df.BERT_Recall.mean()}")
