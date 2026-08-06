"""A lightweight Portuguese intent and entity understanding assistant."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .lexicon import canonicalize_text
from .education import detect_subject, subject_response
from .literacy import literacy_response


def normalize_text(text: str) -> str:
    """Normalize whitespace, case, and punctuation while preserving accents."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\sÀ-ÿ]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", canonicalize_text(text)).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize Portuguese words, including accented characters."""
    return re.findall(r"[\wÀ-ÿ]+", normalize_text(text), flags=re.UNICODE)


def load_intent_dataset(path: str | Path) -> tuple[list[str], list[str]]:
    """Load a JSONL dataset with ``text`` and ``intent`` fields."""
    texts, intents = [], []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            texts.append(str(row["text"]))
            intents.append(str(row["intent"]))
        except (json.JSONDecodeError, KeyError) as error:
            raise ValueError(f"invalid dataset row at line {line_number}") from error
    if not texts or len(set(intents)) < 2:
        raise ValueError("dataset must contain text examples for at least two intents")
    return texts, intents


def extract_entities(text: str) -> list[dict[str, str]]:
    """Extract common entities using Portuguese-friendly deterministic patterns."""
    entities: list[dict[str, str]] = []
    patterns = {
        "EMAIL": r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        "DATE": r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
        "NUMBER": r"(?<![/-])\b\d+(?:[,.]\d+)?\b(?![/-])",
    }
    for entity_type, pattern in patterns.items():
        entities.extend({"text": match.group(0), "type": entity_type} for match in re.finditer(pattern, text))
    return entities


def evaluate_classifier(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    """Return accuracy, precision, recall, and macro F1 for intent predictions."""
    true, predicted = list(y_true), list(y_pred)
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "precision": float(precision_score(true, predicted, average="macro", zero_division=0)),
        "recall": float(recall_score(true, predicted, average="macro", zero_division=0)),
        "f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
    }


class PortugueseAssistant:
    """Intent classifier with lightweight context and Portuguese responses."""

    responses = {
        "saudacao": "Olá! Como posso ajudar você?",
        "despedida": "Até mais!",
        "treinamento": "Para treinar o modelo, use ./train.sh e informe seus dados em formato NPZ.",
        "execucao_modelo": "Para executar o modelo, use .venv/bin/python run.py.",
        "duvida_matematica": "Posso ajudar com álgebra, cálculo, probabilidade e outros temas de Matemática.",
        "duvida_ciencias": "Posso ajudar a organizar estudos de Física, Química e Biologia.",
        "duvida_humanas": "Posso ajudar com História, Geografia, Filosofia e Sociologia.",
        "duvida_linguagens": "Posso ajudar com Língua Portuguesa, Literatura, Redação e idiomas.",
        "desconhecido": "Ainda não entendi completamente. Pode reformular a pergunta?",
        "alfabetizacao": "Posso ajudar com letras, formação de sílabas, leitura e classificação de palavras. Diga uma palavra para eu analisá-la.",
    }

    keyword_intents = {
        "oi": "saudacao",
        "olá": "saudacao",
        "ola": "saudacao",
        "bom dia": "saudacao",
        "boa tarde": "saudacao",
        "boa noite": "saudacao",
        "tudo bem": "saudacao",
        "como você está": "saudacao",
        "como voce esta": "saudacao",
        "tchau": "despedida",
        "até logo": "despedida",
        "ate logo": "despedida",
        "treinamento": "treinamento",
        "treinar": "treinamento",
        "treino": "treinamento",
        "executar": "execucao_modelo",
        "modelo": "execucao_modelo",
        "matemática": "duvida_matematica",
        "matematica": "duvida_matematica",
        "álgebra": "duvida_matematica",
        "cálculo": "duvida_matematica",
        "probabilidade": "duvida_matematica",
        "física": "duvida_ciencias",
        "fisica": "duvida_ciencias",
        "química": "duvida_ciencias",
        "quimica": "duvida_ciencias",
        "biologia": "duvida_ciencias",
        "história": "duvida_humanas",
        "historia": "duvida_humanas",
        "geografia": "duvida_humanas",
        "filosofia": "duvida_humanas",
        "sociologia": "duvida_humanas",
        "português": "duvida_linguagens",
        "portugues": "duvida_linguagens",
        "literatura": "duvida_linguagens",
        "redação": "duvida_linguagens",
        "redacao": "duvida_linguagens",
        "inglês": "duvida_linguagens",
        "ingles": "duvida_linguagens",
        "letra": "alfabetizacao",
        "alfabeto": "alfabetizacao",
        "sílaba": "alfabetizacao",
        "silaba": "alfabetizacao",
        "palavra": "alfabetizacao",
        "separação silábica": "alfabetizacao",
        "separacao silabica": "alfabetizacao",
    }

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.history: list[tuple[str, str]] = []
        self._trained = False

    def fit(self, texts: Iterable[str], intents: Iterable[str]) -> "PortugueseAssistant":
        texts, intents = list(texts), list(intents)
        if len(texts) != len(intents) or len(set(intents)) < 2:
            raise ValueError("texts and intents must have equal length and at least two classes")
        features = self.vectorizer.fit_transform(texts)
        self.classifier.fit(features, intents)
        self._trained = True
        return self

    def predict_intent(self, text: str) -> tuple[str, float]:
        if not self._trained:
            raise RuntimeError("assistant must be trained before prediction")
        features = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(features)[0]
        index = int(np.argmax(probabilities))
        intent, confidence = str(self.classifier.classes_[index]), float(probabilities[index])
        normalized = normalize_text(text)
        for keyword, keyword_intent in self.keyword_intents.items():
            if re.search(rf"\b{re.escape(keyword)}\b", normalized):
                return keyword_intent, max(confidence, 0.9)
        if confidence < 0.35:
            return "desconhecido", confidence
        return intent, confidence

    def respond(self, text: str) -> dict[str, object]:
        """Predict an intent, retain conversation context, and generate a response."""
        intent, confidence = self.predict_intent(text)
        subject = detect_subject(text)
        if subject:
            intent = f"educacao_{subject.lower().replace(' ', '_')}"
            confidence = max(confidence, 0.9)
        if self.history and len(tokenize(text)) <= 4:
            previous_intent = self.history[-1][1]
            intent = intent if confidence >= 0.5 else previous_intent
        self.history.append((text, intent))
        literacy_word = None
        if intent == "alfabetizacao":
            words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
            ignored = {"o", "que", "é", "ser", "e", "uma", "um", "a", "as", "os", "como", "se", "da", "de", "do", "para", "com", "por", "letra", "letras", "alfabeto", "sílaba", "silaba", "sílabas", "silabas", "palavra", "palavras", "separação", "separacao", "silábica", "silabica"}
            candidates = [word for word in words if normalize_text(word) not in ignored]
            literacy_word = candidates[-1] if candidates else None
        return {
            "text": text,
            "intent": intent,
            "confidence": confidence,
            "entities": extract_entities(text),
            "subject": subject,
            "response": literacy_response(literacy_word) if literacy_word else (subject_response(subject) if subject else self.responses.get(intent, "Entendi. Pode explicar um pouco mais a sua dúvida?")),
        }
