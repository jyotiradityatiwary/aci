import re
from enum import Enum

from pydantic import BaseModel, PrivateAttr


class Emotion(str, Enum):
    "Range of emotions that we are working with"

    neutral = "neutral"
    calm = "calm"
    happy = "happy"
    sad = "sad"
    angry = "angry"
    fearful = "fearful"
    disgust = "disgust"
    surprise = "surprise"

    @staticmethod
    def from_id(id: int):
        return [
            Emotion.neutral,
            Emotion.calm,
            Emotion.happy,
            Emotion.sad,
            Emotion.angry,
            Emotion.fearful,
            Emotion.disgust,
            Emotion.surprise,
        ][id - 1]


class EmotionalIntensity(str, Enum):
    'note: no "strong" intensity for "neutral" emotion'

    normal = "normal"
    strong = "strong"

    @staticmethod
    def from_id(id: int):
        return [
            EmotionalIntensity.normal,
            EmotionalIntensity.strong,
        ][id - 1]


class Statement(str, Enum):
    "statement said in the speech audio sample"

    kids = "Kids are talking by the door"
    dogs = "Dogs are sitting by the door"

    @staticmethod
    def from_id(id: int):
        return [
            Statement.kids,
            Statement.dogs,
        ][id - 1]


class Gender(str, Enum):
    male = "male"
    female = "female"


class Actor(BaseModel):
    actor_id: int
    gender: Gender

    @classmethod
    def from_id(cls, id: int) -> "Actor":
        return cls(
            actor_id=id,
            gender=Gender.female if id % 2 == 0 else Gender.male,
        )


class SpeechDetails(BaseModel):
    emotion: Emotion
    emotional_intensity: EmotionalIntensity
    statement: Statement
    actor: Actor

    _ravdess_filename_pattern: re.Pattern = PrivateAttr(
        re.compile(
            r"03-01-(?P<emotion_id>\d*)-(?P<emotion_intensity_id>\d*)-(?P<statement_id>\d*)-(?:\d*)-(?P<actor_id>\d*).wav"
        )
    )

    @classmethod
    def from_ravdess_filename(cls: type, filename: str) -> "SpeechDetails":
        pattern_match = cls._ravdess_filename_pattern.default.fullmatch(filename)
        if not pattern_match:
            raise ValueError(
                f"Invalid filename (does not match expected pattern): {repr(filename)}"
            )
        group = pattern_match.group
        return cls(
            emotion=Emotion.from_id(int(group("emotion_id"))),
            emotional_intensity=EmotionalIntensity.from_id(
                int(group("emotion_intensity_id"))
            ),
            statement=Statement.from_id(int(group("statement_id"))),
            actor=Actor.from_id(int(group("actor_id"))),
        )
