import os

import librosa
from pandas import DataFrame, Index
from parse_data import SpeechDetails
from transformers.models.speech_to_text.modeling_speech_to_text import (
    Speech2TextForConditionalGeneration,
)
from transformers.models.speech_to_text.processing_speech_to_text import (
    Speech2TextProcessor,
)

from config import RAVDESS_AUDIO_DIR

model = Speech2TextForConditionalGeneration.from_pretrained(
    "facebook/s2t-small-librispeech-asr"
)
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

filenames = librosa.util.find_files(directory=RAVDESS_AUDIO_DIR, ext=["wav"], limit=10)

signals = [librosa.load(file, sr=16000)[0] for file in filenames]
inputs = processor(signals, sampling_rate=16000, return_tensors="pt", padding=True)

generated_ids = model.generate(
    inputs["input_features"], attention_mask=inputs["attention_mask"]
)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

df = DataFrame(
    {
        "predicted_statement": transcription,
        "original_statement": [
            SpeechDetails.from_ravdess_filename(os.path.basename(fname)).statement
            for fname in filenames
        ],
    },
    index=Index(filenames, name="filename"),
)
df.to_csv("outputs/speech_to_text.csv")
print(df.head())
