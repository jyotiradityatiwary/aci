from pandas import DataFrame

from config import TARGET_RESPONSES_DIR


class CorrectAnswers:
    def __init__(self):
        self._df = self._build_df()

    @staticmethod
    def _build_df() -> DataFrame:
        content = []
        with open(TARGET_RESPONSES_DIR / "1.txt") as f:
            content.append(f.read().strip())
        with open(TARGET_RESPONSES_DIR / "2.txt") as f:
            content.append(f.read().strip())
        content = "\n\n".join(content)
        content = [p.split("\n")[4] for p in content.split("\n\n")]

        df = DataFrame(
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

        df = df.set_index(["emotion", "statement"])
        df = df.sort_index()
        return df

    def get(self, emotion: str, statement: str) -> str:
        """Todo: fix fear-fearful datapoint"""
        if emotion == "fear":
            emotion = "fearful"
        filtered_df = self._df.loc[(emotion, statement)]
        assert isinstance(filtered_df, DataFrame)
        correct_answers = filtered_df["correct_answer"]
        correct_answer = correct_answers.sample(n=1).iloc[0]
        return correct_answer
