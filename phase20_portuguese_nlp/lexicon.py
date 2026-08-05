"""Small Portuguese verb and synonym lexicon for text normalization."""

from __future__ import annotations

import re


VERB_DICTIONARY: dict[str, tuple[str, ...]] = {
    "ser": ("sou", "é", "somos", "são", "era", "eram", "foi", "foram", "serei", "será"),
    "estar": ("estou", "está", "estamos", "estão", "estava", "estavam", "esteve"),
    "ter": ("tenho", "tem", "temos", "têm", "tinha", "tinham", "teve", "terei"),
    "fazer": ("faço", "faz", "fazemos", "fazem", "fazia", "fez", "fizeram", "farei"),
    "querer": ("quero", "quer", "queremos", "querem", "queria", "quis", "quererei"),
    "poder": ("posso", "pode", "podemos", "podem", "podia", "pôde", "poderia"),
    "precisar": ("preciso", "precisa", "precisamos", "precisam", "precisava"),
    "estudar": ("estudo", "estuda", "estudamos", "estudam", "estudei", "estudando"),
    "aprender": ("aprendo", "aprende", "aprendemos", "aprendem", "aprendi", "aprendendo"),
    "ajudar": ("ajudo", "ajuda", "ajudamos", "ajudam", "ajudei", "ajudando"),
    "explicar": ("explico", "explica", "explicamos", "explicam", "expliquei"),
    "treinar": ("treino", "treina", "treinamos", "treinam", "treinei", "treinando"),
    "executar": ("executo", "executa", "executamos", "executam", "executei"),
    "criar": ("crio", "cria", "criamos", "criam", "criei", "criando"),
    "conversar": ("converso", "conversa", "conversamos", "conversam", "conversei"),
    "entender": ("entendo", "entende", "entendemos", "entendem", "entendi"),
    "falar": ("falo", "fala", "falamos", "falam", "falei", "falando"),
    "escrever": ("escrevo", "escreve", "escrevemos", "escrevem", "escrevi"),
}

SYNONYM_DICTIONARY: dict[str, tuple[str, ...]] = {
    "ajudar": ("auxiliar", "apoiar", "orientar", "socorrer"),
    "aprender": ("estudar", "compreender", "assimilar", "conhecer"),
    "entender": ("compreender", "interpretar", "saber", "perceber"),
    "explicar": ("ensinar", "esclarecer", "detalhar", "demonstrar"),
    "executar": ("rodar", "iniciar", "acionar"),
    "criar": ("desenvolver", "produzir", "construir", "elaborar"),
    "falar": ("dizer", "comunicar", "conversar", "comentar"),
    "estudar": ("revisar", "pesquisar", "aprender", "praticar"),
    "rápido": ("veloz", "ligeiro", "ágil"),
    "difícil": ("complexo", "complicado", "árduo"),
    "importante": ("relevante", "essencial", "fundamental"),
}


def build_lexicon() -> dict[str, str]:
    """Return every known conjugation/synonym mapped to its canonical term."""
    mapping: dict[str, str] = {}
    for canonical, forms in VERB_DICTIONARY.items():
        mapping[canonical] = canonical
        mapping.update({form: canonical for form in forms})
    for canonical, synonyms in SYNONYM_DICTIONARY.items():
        mapping[canonical] = canonical
        mapping.update({synonym: canonical for synonym in synonyms})
    return mapping


LEXICON = build_lexicon()


def canonicalize_text(text: str) -> str:
    """Replace known verb forms and synonyms with canonical vocabulary."""
    words = re.findall(r"[\wÀ-ÿ]+|[^\wÀ-ÿ]+", text.lower(), flags=re.UNICODE)
    return "".join(LEXICON.get(word, word) for word in words)
