# Compreensão do Português

Implementação inicial de NLP em português com:

- Dataset JSONL de intenções em português
- Normalização e tokenização com suporte a acentos
- Dicionário de verbos, conjugações e sinônimos
- Classificação de intenções com TF-IDF + regressão logística
- Extração de e-mails, datas e números
- Histórico simples para contexto de conversa
- Geração de respostas por intenção
- Métricas de accuracy, precision, recall e F1

Treinar e testar:

```bash
.venv/bin/python -m unittest discover -s phase20_portuguese_nlp -p 'test_*.py'
```

Esta é uma base de compreensão/classificação local. Para respostas abertas e
contexto avançado, será necessário evoluir para um modelo de linguagem treinado
ou ajustado com dados em português.
