# Phase 22 — Language Model

Implementação TensorFlow de um protótipo de geração causal:

- tokenizer de caracteres e vocabulário persistente;
- Transformer decoder-only;
- RoPE e Multi-Query Attention;
- geração autoregressiva com cache de chaves/valores;
- greedy decoding e beam search;
- detecção de Flash Attention com fallback seguro;
- perdas opcionais DPO e RLHF/reward model;
- `InferenceEngine` para integrar o modelo ao chat.

Teste:

```bash
.venv/bin/python -m unittest phase22_language_model.test_phase22
```

Treine o protótipo com um arquivo UTF-8, uma pergunta/resposta por linha:

```bash
.venv/bin/python -m phase22_language_model.train_lm \
  --text data/lm_train.txt \
  --output-dir artifacts/language_model
```

O dataset QA padrão está em `data/qa_train.jsonl`. Para treinar com ele:

```bash
.venv/bin/python -m phase22_language_model.train_qa \
  --data data/qa_train.jsonl \
  --output-dir artifacts/language_model
```

O treino também carrega o vocabulário pt-BR em `data/pt_br_dictionary`,
licenciado sob MIT, e salva a contagem em `dictionary_stats.json`.
O treino faz primeiro um pré-treino lexical com todas as palavras empacotadas
em sequências e depois ajusta o modelo nas perguntas/respostas.

Para gerar uma explicação estrutural de todas as palavras, incluindo letras,
sílabas e subsequências contíguas, e usar esse material no treino:

```bash
.venv/bin/python -m phase20_portuguese_nlp.build_word_explanations \
  --output data/word_explanations.jsonl
.venv/bin/python -m phase22_language_model.train_qa \
  --data data/qa_phase20.jsonl \
  --word-explanations data/word_explanations.jsonl \
  --output-dir artifacts/language_model
```

São usadas subsequências contíguas de até 6 caracteres. Todas as subsequências
não contíguas não são materializadas porque crescem exponencialmente e o
dicionário não fornece definições semânticas confiáveis.

Para treinar com todo o conhecimento educacional da Phase 20:

```bash
.venv/bin/python -m phase20_portuguese_nlp.build_qa_dataset \
  --output data/qa_phase20.jsonl \
  --dictionary-variants 4
.venv/bin/python -m phase22_language_model.train_qa \
  --data data/qa_phase20.jsonl \
  --output-dir artifacts/language_model
```

Depois do treino, o adaptador próprio para o `lm-eval` pode ser executado com:

```bash
.venv/bin/python -m phase22_language_model.run_lm_eval \
  --model-dir artifacts/language_model \
  --tasks mmlu,gpqa,aime24,bbh
```

Esta implementação fornece a arquitetura. Para obter pontuação em MMLU,
GPQA, AIME e demais benchmarks, ainda é necessário treinar o modelo com um
dataset amplo de perguntas/respostas compatível com cada benchmark. O
`qa_train.jsonl` é apenas um dataset inicial de demonstração e não produz
pontuações competitivas.
