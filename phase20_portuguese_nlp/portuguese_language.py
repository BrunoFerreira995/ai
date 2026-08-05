"""Starter tools for Portuguese reading, grammar, orthography, and semantics."""

from __future__ import annotations

import re
from collections import Counter


PORTUGUESE_TOPICS = {
    "Leitura e interpretação de textos": {
        "summary": "identificação de tema, tese, informações explícitas e inferências",
        "skills": ("tema", "ideia principal", "inferência", "evidência textual", "ponto de vista"),
    },
    "Gramática": {
        "summary": "estudo do funcionamento das palavras e das relações entre elas",
        "skills": ("classes gramaticais", "concordância", "regência", "crase", "pontuação"),
    },
    "Ortografia e acentuação": {
        "summary": "escrita correta, uso de letras, hífen e regras de acentuação",
        "skills": ("oxítonas", "paroxítonas", "proparoxítonas", "hiato", "hífen"),
    },
    "Morfologia e sintaxe": {
        "summary": "estrutura das palavras e organização dos termos na oração",
        "skills": ("radical e afixos", "classes de palavras", "sujeito", "predicado", "orações"),
    },
    "Semântica e variação linguística": {
        "summary": "sentidos das palavras e diversidade de usos da língua portuguesa",
        "skills": ("sinonímia", "antonímia", "polissemia", "sentido figurado", "variação regional"),
    },
}


COMMON_ORTHOGRAPHY = {
    "voce": "você",
    "tambem": "também",
    "matematica": "matemática",
    "fisica": "física",
    "quimica": "química",
    "portugues": "português",
    "lingua": "língua",
    "interpretacao": "interpretação",
    "acentuacao": "acentuação",
    "gramatica": "gramática",
    "redacao": "redação",
}

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "de", "do", "da", "dos", "das", "e", "ou", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "que", "se", "é", "foi", "ao", "à",
}


def portuguese_topic_summary(topic: str) -> str:
    """Return a compact lesson outline for one Portuguese topic."""
    if topic not in PORTUGUESE_TOPICS:
        raise KeyError(f"unknown Portuguese topic: {topic}")
    content = PORTUGUESE_TOPICS[topic]
    return f"{content['summary']}. Tópicos: {', '.join(content['skills'])}."


def tokenize_words(text: str) -> list[str]:
    """Extract Portuguese words while preserving accents."""
    return re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())


def interpret_text(text: str) -> dict[str, object]:
    """Produce basic reading statistics useful for interpretation exercises."""
    sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    words = tokenize_words(text)
    content_words = [word for word in words if word not in STOPWORDS]
    return {
        "sentences": sentences,
        "word_count": len(words),
        "content_words": content_words,
        "most_common_words": Counter(content_words).most_common(5),
        "question": "Qual é o tema principal e quais evidências aparecem no texto?",
    }


def correct_common_orthography(text: str) -> str:
    """Correct a small explicit dictionary of frequent unaccented words."""
    return re.sub(
        r"\b[\wÀ-ÿ]+\b",
        lambda match: COMMON_ORTHOGRAPHY.get(match.group(0).lower(), match.group(0)),
        text,
        flags=re.UNICODE,
    )


def morphological_analysis(text: str) -> list[dict[str, str]]:
    """Provide a transparent heuristic classification of common word classes."""
    result = []
    for word in tokenize_words(text):
        if word in STOPWORDS:
            category = "palavra funcional"
        elif word.endswith(("ar", "er", "ir")):
            category = "verbo no infinitivo provável"
        elif word.endswith(("mente",)):
            category = "advérbio provável"
        elif word.endswith(("ção", "dade", "mento")):
            category = "substantivo provável"
        else:
            category = "palavra lexical"
        result.append({"word": word, "category": category})
    return result


def syntactic_analysis(text: str) -> dict[str, object]:
    """Identify simple sentence structure markers using deterministic heuristics."""
    words = tokenize_words(text)
    verbs = [word for word in words if word.endswith(("ar", "er", "ir", "ou", "am", "em"))]
    return {
        "subject_hint": words[0] if words else None,
        "verb_hints": verbs,
        "predicate_hint": words[1:] if len(words) > 1 else [],
        "note": "Análise introdutória; uma análise sintática completa requer parser linguístico especializado.",
    }


def semantic_relations(word: str) -> dict[str, object]:
    """Return starter synonym, antonym, and polysemy examples."""
    relations = {
        "feliz": {"synonyms": ["contente", "alegre"], "antonyms": ["triste"]},
        "rápido": {"synonyms": ["veloz", "ágil"], "antonyms": ["lento"]},
        "claro": {"synonyms": ["evidente", "luminoso"], "antonyms": ["escuro"]},
    }
    return relations.get(word.lower(), {"synonyms": [], "antonyms": [], "polysemy": []})
