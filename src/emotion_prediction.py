import os

import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn

from config import MODELS_DIR

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

MODEL_SAMPLE_RATE = 48000  # The sample rate the model was trained on


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


def predict_from_signal(signal, sample_rate, model, device) -> tuple[str, dict[str, float]]:
    """
    Predicts the emotion from a raw audio signal.

    Args:
        signal (np.array): Audio time series.
        sample_rate (int): Sample rate of the audio.
        model (torch.nn.Module): The loaded emotion recognition model.
        device (str): The device to run inference on ('cpu' or 'cuda').

    Returns:
        tuple: (predicted_emotion_str, emotion_probabilities_dict)
    """

    # 1. Resample if necessary
    if sample_rate != MODEL_SAMPLE_RATE:
        print(
            f"Warning: Input sample rate ({sample_rate}Hz) differs from model's ({MODEL_SAMPLE_RATE}Hz). Resampling..."
        )
        signal = librosa.resample(
            y=signal, orig_sr=sample_rate, target_sr=MODEL_SAMPLE_RATE
        )
        sample_rate = MODEL_SAMPLE_RATE  # Update sample_rate for processing

    # 2. Pad or truncate to 3 seconds
    # Note: The original script used a 0.5s offset and 3s duration.
    # Here, we just ensure the signal is 3s long.
    target_len = int(MODEL_SAMPLE_RATE * 3)

    # Apply offset (0.5s) and duration (3s)
    start_sample = int(MODEL_SAMPLE_RATE * 0.5)
    end_sample = start_sample + target_len

    # Create a 3-second zero-padded signal
    processed_signal = np.zeros((target_len,))

    # Check if the original signal (after offset) is long enough
    if len(signal) > start_sample:
        # Get the slice from 0.5s to 3.5s
        signal_slice = signal[start_sample:end_sample]
        # Copy this slice into our 3s buffer
        processed_signal[: len(signal_slice)] = signal_slice
    else:
        # If the signal is shorter than 0.5s, just take what we can
        # (This will result in a mostly empty signal)
        print("Warning: Signal is shorter than 0.5s offset.")
        pass  # signal remains zeros

    # 3. Get mel spectrogram
    mel_spectrogram = getMELspectrogram(processed_signal, MODEL_SAMPLE_RATE)

    # Add batch and channel dimensions: (h, w) -> (1, 1, h, w)
    X = np.expand_dims(mel_spectrogram, axis=0)
    X = np.expand_dims(X, axis=0)

    # 4. Scale data
    b, c, h, w = X.shape
    X_reshaped = np.reshape(X, newshape=(b, -1))
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    X_scaled = scaler.transform(X_reshaped)  # Use transform, NOT fit_transform
    X_scaled = np.reshape(X_scaled, newshape=(b, c, h, w))

    X_tensor = torch.tensor(X_scaled, device=device).float()

    # 6. Run inference
    with torch.no_grad():
        output_logits, output_softmax = model(X_tensor)

        # Get the prediction
        prediction_idx = torch.argmax(output_softmax, dim=1).item()
        predicted_emotion = EMOTIONS[int(prediction_idx)]

        # Get probabilities
        probabilities = torch.squeeze(output_softmax).cpu().numpy()

    # Format probabilities into a dictionary
    emotion_probabilities: dict[str, float] = {EMOTIONS[i]: prob.item() for i, prob in enumerate(probabilities)}

    return predicted_emotion, emotion_probabilities


def load_emotion_model():
    """
    Loads the ParallelModel from a file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(MODELS_DIR, "cnn_transf_parallel_model.pt")
    try:
        model = ParallelModel(len(EMOTIONS)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
