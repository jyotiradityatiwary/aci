from numpy.typing import NDArray
from transformers.models.speech_to_text.modeling_speech_to_text import (
    Speech2TextForConditionalGeneration,
)
from transformers.models.speech_to_text.processing_speech_to_text import (
    Speech2TextProcessor,
)


class SpeechToTextModel:
    _model_name = "facebook/s2t-small-librispeech-asr"
    required_sampling_rate = 16000

    def __init__(self):
        self.model = Speech2TextForConditionalGeneration.from_pretrained(
            self._model_name
        )
        self.processor = Speech2TextProcessor.from_pretrained(self._model_name)

    def __call__(self, signal: NDArray, sampling_rate: int | float) -> str:
        inputs = self.processor(
            [signal], sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        generated_ids = self.model.generate(
            inputs["input_features"], attention_mask=inputs["attention_mask"]
        )
        transcriptions = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return transcriptions[0]
