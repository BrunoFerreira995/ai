"""Text classification and translation baselines."""

from __future__ import annotations

from collections.abc import Iterable

from phase20_portuguese_nlp.assistant import PortugueseAssistant


class PortugueseTextClassifier:
    """Reuse the Portuguese intent classifier as a text-classification model."""

    def __init__(self):
        self.assistant = PortugueseAssistant()

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "PortugueseTextClassifier":
        self.assistant.fit(texts, labels)
        return self

    def predict(self, text: str) -> tuple[str, float]:
        return self.assistant.predict_intent(text)


class DictionaryTranslator:
    """Small deterministic translation baseline for common educational terms."""

    translations = {
        ("português", "inglês"): {"olá": "hello", "matemática": "mathematics", "física": "physics", "obrigado": "thank you"},
        ("inglês", "português"): {"hello": "olá", "mathematics": "matemática", "physics": "física", "thank you": "obrigado"},
    }

    def translate(self, text: str, source: str = "português", target: str = "inglês") -> str:
        dictionary = self.translations.get((source.lower(), target.lower()), {})
        result = text
        for original, translated in dictionary.items():
            result = result.replace(original, translated)
        return result
