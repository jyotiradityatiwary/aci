from dataclasses import dataclass
from importlib import import_module

from correct_answers import CorrectAnswers
from parse_data import SpeechDetails


class _BleuEvaluator:
    def __init__(self) -> None:
        nltk = import_module("nltk")
        bleu_score = nltk.translate.bleu_score
        self.smoother = bleu_score.SmoothingFunction()
        self.sentence_bleu = bleu_score.sentence_bleu

    def __call__(self, correct_answers: list[str], ai_response: str) -> float:
        ref_texts = correct_answers
        cand_text = ai_response

        reference = [ref_text.split() for ref_text in ref_texts]
        candidate = cand_text.split()

        return self.sentence_bleu(
            reference,
            candidate,
            smoothing_function=self.smoother.method4,  # A good general-purpose smoothing
        )


@dataclass
class BertScore:
    precision: float
    recall: float
    f1: float


class _BertEvaluator:
    def __init__(self) -> None:
        import_module("torch")
        self.bert_score = import_module("bert_score")

    def __call__(self, ai_response: str, correct_answers: list[str]) -> BertScore:
        candidates = [ai_response]
        references = correct_answers

        P, R, F1 = self.bert_score.score(
            candidates,
            references,
            lang="en",
            verbose=True,  # Shows a progress bar
        )

        return BertScore(
            precision=P[0].item(),
            recall=R[0].item(),
            f1=F1.item(),
        )


@dataclass
class EvaluatorResult:
    bleu_score: float
    bert_score: BertScore

    def as_flat_dict(self) -> dict[str, list[str] | list[float]]:
        return {
            "Metric": ["BLEU Score", "BERT F1 Score", "BERT Precision", "BERT Recall"],
            "Result": [
                self.bleu_score,
                self.bert_score.f1,
                self.bert_score.precision,
                self.bert_score.recall,
            ],
        }


class Evaluator:
    def __init__(self):
        self.correct_answers = CorrectAnswers()
        self.bleu_evaluator = _BleuEvaluator()
        self.bert_evaluator = _BertEvaluator()

    def __call__(self, filename: str, response_text: str) -> EvaluatorResult:
        speech_details = SpeechDetails.from_ravdess_filename(filename)
        correct_answer = self.correct_answers.get(
            emotion=speech_details.emotion.value,
            statement=speech_details.statement.value,
        )
        return EvaluatorResult(
            bleu_score=self.bleu_evaluator(
                correct_answers=[correct_answer],
                ai_response=response_text,
            ),
            bert_score=self.bert_evaluator(
                correct_answers=[correct_answer],
                ai_response=response_text,
            ),
        )
