import os

import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from config import EMOTION_PREDICTION_CSV_PATH, MODELS_DIR, RAVDESS_AUDIO_DIR

# Constants (keep same as original)
EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    0: "surprise",
}
SAMPLE_RATE = 48000

# Statement mapping based on filename format
STATEMENTS = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}


# Model definition (same as original)
class ParallelModel(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # 2. conv block
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 3. conv block
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 4. conv block
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2, 4], stride=[2, 4])
        transf_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=512, dropout=0.4, activation="relu"
        )
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)

        # Linear softmax layer
        self.out_linear = nn.Linear(320, num_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # conv embedding
        conv_embedding = self.conv2Dblock(x)  # (b,channel,freq,time)
        conv_embedding = torch.flatten(
            conv_embedding, start_dim=1
        )  # do not flatten batch dimension

        # transformer embedding
        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(
            2, 0, 1
        )  # requires shape = (time,batch,embedding)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)

        # concatenate
        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)

        # final Linear
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)

        return output_logits, output_softmax


# Function to get mel spectrogram (same as original)
def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        win_length=512,
        window="hamming",
        hop_length=256,
        n_mels=128,
        fmax=sample_rate / 2,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


print("Starting inference on all files...")

# Load all files and extract information
all_data = []
all_spectrograms = []

for dirname, _, filenames in os.walk(RAVDESS_AUDIO_DIR):
    for filename in filenames:
        if filename.endswith(".wav"):  # only process wav files
            file_path = os.path.join(dirname, filename)
            try:
                # Parse filename according to the format provided
                identifiers = filename.split(".")[0].split("-")
                emotion = int(identifiers[2])
                if emotion == 8:  # relabel surprise to 0 instead of 8
                    emotion = 0
                statement_id = int(identifiers[4])

                # Load and process audio
                audio, sample_rate = librosa.load(
                    file_path, duration=3, offset=0.5, sr=SAMPLE_RATE
                )
                signal = np.zeros(
                    (
                        int(
                            SAMPLE_RATE * 3,
                        )
                    )
                )
                signal[: len(audio)] = audio

                # Get mel spectrogram
                mel_spectrogram = getMELspectrogram(signal, SAMPLE_RATE)

                # Store data
                all_spectrograms.append(mel_spectrogram)
                all_data.append(
                    {
                        "filename": filename,
                        "correct_emotion": EMOTIONS[emotion],
                        "statement": STATEMENTS[statement_id],
                    }
                )

                print(f"\r Processed {len(all_data)} files", end="")

            except Exception as e:
                print(f"\nSkipping {filename}: {e}")

print(f"\nTotal files processed: {len(all_spectrograms)}")

# Convert to numpy arrays and prepare for inference
X = np.stack(all_spectrograms, axis=0)
X = np.expand_dims(X, 1)  # Add channel dimension

# Scale data (fit scaler on all data since we don't have the original training scaler)
scaler = StandardScaler()
b, c, h, w = X.shape
X_reshaped = np.reshape(X, newshape=(b, -1))
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = np.reshape(X_scaled, newshape=(b, c, h, w))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# Load the saved model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = ParallelModel(len(EMOTIONS)).to(device)
model.load_state_dict(
    torch.load(
        os.path.join(MODELS_DIR, "cnn_transf_parallel_model.pt"), map_location=device
    )
)
model.eval()
print("Model loaded successfully")

# Run inference
predictions = []
batch_size = 32

print("Running inference...")
with torch.no_grad():
    for i in range(0, len(X_scaled), batch_size):
        batch_end = min(i + batch_size, len(X_scaled))
        X_batch = torch.tensor(X_scaled[i:batch_end], device=device).float()

        output_logits, output_softmax = model(X_batch)
        batch_predictions = torch.argmax(output_softmax, dim=1)

        for pred in batch_predictions.cpu().numpy():
            predictions.append(EMOTIONS[pred])

        print(
            f"\r Processing batch {i // batch_size + 1}/{(len(X_scaled) - 1) // batch_size + 1}",
            end="",
        )

print("\nInference completed!")

# Create results DataFrame
results_data = []
for i, data_item in enumerate(all_data):
    results_data.append(
        {
            "filename": data_item["filename"],
            "correct_emotion": data_item["correct_emotion"],
            "predicted_emotion": predictions[i],
            "statement": data_item["statement"],
        }
    )

results_df = pd.DataFrame(results_data)

# Save to CSV
output_path = EMOTION_PREDICTION_CSV_PATH
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Display summary
print("\nSummary:")
print(f"Total files processed: {len(results_df)}")
correct_predictions = sum(
    results_df["correct_emotion"] == results_df["predicted_emotion"]
)
accuracy = correct_predictions / len(results_df) * 100
print(f"Correct predictions: {correct_predictions}/{len(results_df)} ({accuracy:.2f}%)")

# Show first few rows
print("\nFirst 10 results:")
print(results_df.head(10))
