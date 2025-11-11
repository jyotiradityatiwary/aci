from dataclasses import dataclass

import bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


class BleuEvaluator:
    def __init__(self) -> None:
        self.smoother = SmoothingFunction()

    def calculate_score(self, correct_answers: list[str], ai_response: str) -> float:
        ref_texts = correct_answers
        cand_text = ai_response

        reference = [ref_text.split() for ref_text in ref_texts]
        candidate = cand_text.split()

        return sentence_bleu(
            reference,
            candidate,
            smoothing_function=self.smoother.method4,  # A good general-purpose smoothing
        )


@dataclass
class BertScore:
    precision: float
    recall: float
    f1: float


def calculate(ai_response: str, correct_answers: list[str]) -> BertScore:
    candidates = [ai_response]
    references = correct_answers

    P, R, F1 = bert_score.score(
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
