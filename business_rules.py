"""Business rules that turn classifier classes into Portuguese responses."""

from __future__ import annotations

DEFAULT_RESPONSES = {
    "Língua Portuguesa": "Posso ajudar com interpretação de textos, gramática, ortografia e redação.",
    "Matemática": "Posso ajudar a resolver problemas de Matemática passo a passo.",
    "Ciências da Natureza": "Posso explicar conteúdos de Física, Química e Biologia.",
    "Ciências Humanas": "Posso ajudar com História, Geografia, Filosofia e Sociologia.",
    "Língua Inglesa e Idiomas": "Posso ajudar com vocabulário, gramática e interpretação de idiomas.",
    "Artes": "Posso conversar sobre Artes Visuais, Música, Teatro, Dança e História da Arte.",
    "Educação Física e Itinerários": "Posso ajudar com esportes, saúde, qualidade de vida e projetos escolares.",
}


def response_for_class(class_name: str) -> str:
    """Return the business response associated with a predicted class."""
    return DEFAULT_RESPONSES.get(
        class_name,
        f"A classificação recebida foi {class_name}. Como posso ajudar?",
    )
