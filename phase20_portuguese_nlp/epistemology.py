"""Training knowledge about epistemology and the subject–object relation."""

from __future__ import annotations


EPISTEMOLOGY_ROWS = [
    {"question": "O que é teoria do conhecimento?", "answer": "Teoria do conhecimento, ou epistemologia, investiga como o conhecimento é produzido, justificado e limitado. Ela analisa a relação entre sujeito cognoscente, objeto conhecido, crença, verdade e evidência."},
    {"question": "O que é o sujeito do conhecimento?", "answer": "O sujeito do conhecimento é quem conhece, interpreta, investiga ou formula juízos. Ele participa ativamente do processo cognitivo por meio de percepção, linguagem, memória, razão e experiência."},
    {"question": "O que é o objeto do conhecimento?", "answer": "O objeto do conhecimento é aquilo que pode ser investigado ou conhecido. Pode ser um fenômeno, uma ideia, um acontecimento, uma entidade, uma relação ou um problema estudado pelo sujeito."},
    {"question": "Qual é a relação entre sujeito e objeto do conhecimento?", "answer": "A relação entre sujeito e objeto é o processo pelo qual alguém procura compreender algo. O sujeito interpreta o objeto usando capacidades cognitivas, linguagem, métodos e evidências; o conhecimento não deve ser confundido automaticamente com o objeto em si."},
    {"question": "O sujeito cria o objeto do conhecimento?", "answer": "Depende da perspectiva filosófica. O sujeito pode organizar, interpretar e construir representações do objeto, mas isso não significa necessariamente que ele crie a realidade investigada. A epistemologia estuda essa relação entre realidade, experiência e representação."},
    {"question": "O que é conhecimento justificado?", "answer": "Conhecimento justificado é uma crença apoiada por razões, evidências ou métodos adequados. Uma opinião verdadeira por acaso não é, por si só, conhecimento bem fundamentado."},
    {"question": "Qual é a diferença entre sujeito e objeto?", "answer": "O sujeito é quem realiza o ato de conhecer; o objeto é aquilo que é conhecido ou investigado. Essa distinção ajuda a analisar como perspectivas, métodos e evidências influenciam o conhecimento."},
    {"question": "Como o empirismo explica o conhecimento?", "answer": "O empirismo enfatiza a experiência e a observação como fontes importantes do conhecimento. A observação ainda precisa ser interpretada e organizada por conceitos, linguagem e métodos."},
    {"question": "Como o racionalismo explica o conhecimento?", "answer": "O racionalismo destaca o papel da razão, dos conceitos e da argumentação na construção do conhecimento. A experiência pode fornecer dados, mas a razão ajuda a relacioná-los e avaliá-los."},
    {"question": "O que é criticismo na teoria do conhecimento?", "answer": "O criticismo examina conjuntamente as contribuições da experiência e da razão, perguntando quais são as condições, possibilidades e limites do conhecimento."},
]


def epistemology_rows() -> list[dict[str, str]]:
    return list(EPISTEMOLOGY_ROWS)
