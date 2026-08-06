"""Starter educational knowledge base and subject routing."""

from __future__ import annotations

import re


EDUCATIONAL_KNOWLEDGE: dict[str, dict[str, object]] = {
    "Língua Portuguesa": {
        "aliases": ("português", "portugues", "língua portuguesa", "gramática"),
        "topics": ("leitura e interpretação de textos", "gramática", "ortografia e acentuação", "morfologia e sintaxe", "semântica e variação linguística"),
        "summary": "estudo da língua, leitura, gramática, ortografia e comunicação",
    },
    "Literatura": {
        "aliases": ("literatura", "literatura brasileira", "literatura portuguesa"),
        "topics": ("gêneros literários", "escolas literárias", "literatura brasileira", "literatura portuguesa", "análise de obras e autores"),
        "summary": "leitura e análise de obras, autores, gêneros e movimentos literários",
    },
    "Redação": {
        "aliases": ("redação", "redacao", "produção textual", "texto dissertativo"),
        "topics": ("estrutura textual", "tese e planejamento", "argumentação", "coesão e coerência", "revisão e conclusão"),
        "summary": "planejamento, argumentação, coesão e revisão de textos",
    },
    "Matemática": {
        "aliases": ("matemática", "matematica", "álgebra", "geometria", "probabilidade"),
        "topics": ("aritmética", "álgebra", "geometria", "funções", "estatística e probabilidade"),
        "summary": "raciocínio quantitativo, álgebra, geometria, funções e probabilidade",
    },
    "Física": {
        "aliases": ("física", "fisica", "mecânica", "eletricidade"),
        "topics": ("cinemática", "dinâmica e leis de Newton", "trabalho e energia", "ondas e óptica", "eletricidade e magnetismo"),
        "summary": "estudo da matéria, movimento, energia, forças, ondas e eletricidade",
    },
    "Química": {
        "aliases": ("química", "quimica", "química orgânica"),
        "topics": ("estrutura atômica", "tabela periódica", "ligações químicas", "reações e estequiometria", "química orgânica"),
        "summary": "estudo da matéria, suas propriedades, transformações e reações",
    },
    "Biologia": {
        "aliases": ("biologia", "genética", "ecologia", "célula"),
        "topics": ("citologia", "genética", "evolução", "ecologia", "fisiologia"),
        "summary": "estudo da vida, células, organismos, genética, evolução e ecossistemas",
    },
    "História": {
        "aliases": ("história", "historia", "história do brasil"),
        "topics": ("antiguidade", "idade média", "idade moderna", "história do Brasil", "mundo contemporâneo"),
        "summary": "análise de sociedades, acontecimentos, permanências e transformações históricas",
    },
    "Geografia": {
        "aliases": ("geografia", "cartografia", "geopolítica"),
        "topics": ("cartografia", "população e demografia", "urbanização e industrialização", "geopolítica", "meio ambiente e sustentabilidade"),
        "summary": "estudo do espaço geográfico, sociedade, território, ambiente e paisagem",
    },
    "Filosofia": {
        "aliases": ("filosofia", "ética", "epistemologia"),
        "topics": ("filosofia antiga", "ética", "política", "epistemologia", "lógica e argumentação"),
        "summary": "investigação crítica sobre conhecimento, ética, política e existência",
    },
    "Sociologia": {
        "aliases": ("sociologia", "sociedade", "cultura"),
        "topics": ("cultura e socialização", "instituições sociais", "classes e desigualdades", "trabalho e economia", "cidadania e movimentos sociais"),
        "summary": "estudo das relações sociais, instituições, cultura, trabalho e desigualdades",
    },
    "Língua Inglesa ou outro idioma": {
        "aliases": ("inglês", "ingles", "língua inglesa", "idioma", "espanhol", "francês"),
        "topics": ("vocabulário", "gramática", "leitura e interpretação", "escrita e conversação", "pronúncia e compreensão auditiva"),
        "summary": "desenvolvimento de leitura, escrita, escuta, fala e vocabulário em idiomas",
    },
    "Educação Física": {
        "aliases": ("educação física", "educacao fisica", "esportes", "atividade física"),
        "topics": ("esportes e regras", "jogos e brincadeiras", "corpo e movimento", "saúde e qualidade de vida", "inclusão e cooperação"),
        "summary": "estudo do corpo, movimento, práticas esportivas, saúde e qualidade de vida",
    },
    "Artes": {
        "aliases": ("artes", "arte", "música", "teatro", "dança", "artes visuais"),
        "topics": ("artes visuais", "música", "teatro", "dança", "história e crítica da arte"),
        "summary": "expressão, criação e apreciação em artes visuais, música, teatro e dança",
    },
    "Itinerários Formativos": {
        "aliases": ("itinerário formativo", "itinerarios formativos", "itinerários", "projeto de vida"),
        "topics": ("projeto de vida", "eletivas", "aprofundamento de áreas", "projetos de pesquisa e intervenção", "orientação profissional"),
        "summary": "percursos de aprofundamento e projetos definidos conforme a escola",
    },
}


def detect_subject(text: str) -> str | None:
    """Return the first educational subject mentioned in a message."""
    normalized = text.lower()
    matches = []
    for subject, content in EDUCATIONAL_KNOWLEDGE.items():
        for alias in content["aliases"]:
            if re.search(rf"\b{re.escape(alias)}\b", normalized, flags=re.IGNORECASE):
                matches.append((normalized.index(alias.lower()), subject))
    return min(matches)[1] if matches else None


def subject_response(subject: str) -> str:
    """Generate a concise study-oriented response for a subject."""
    content = EDUCATIONAL_KNOWLEDGE[subject]
    topics = ", ".join(content["topics"][:4])
    return f"Posso ajudar com {subject}: {content['summary']}. Podemos começar por {topics}."
