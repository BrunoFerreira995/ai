# Plano do Causal LM: 25% → 50% no MMLU

Este plano separa conhecimento textual, especialização e comportamento de
resposta. `qa_phase20.jsonl` e referências bibliográficas não substituem o
texto integral das fontes.

## Estágios

1. **Pretraining:** corpus curado em português e inglês, livros licenciados,
   artigos com texto integral autorizado, material educacional, matemática,
   código e Wikipedia. Objetivo: next-token prediction.
2. **Continued pretraining:** misturas com maior proporção para os domínios
   prioritários. Manter uma fração do corpus geral para reduzir esquecimento.
3. **Instruction tuning:** QA, instrução-resposta, problemas resolvidos e
   múltipla escolha com explicação. Separar exemplos de treino e avaliação.
4. **Benchmark:** executar MMLU e conjuntos independentes, sem colocar itens
   de avaliação no treino.

## Experimentos incrementais

Registrar cada checkpoint em uma pasta própria e avançar somente quando os
resultados forem reproduzíveis:

| Marco | Critério mínimo de decisão |
|---|---|
| 25% | baseline reproduzível, loss/validação e categorias registradas |
| 30% | ganho global sem queda relevante em matemática e humanidades |
| 40% | ganho confirmado em duas execuções com seeds diferentes |
| 50% | ganho confirmado em MMLU e em um conjunto externo não usado no ajuste |

Em cada execução registrar: tokens vistos, batch efetivo, learning rate,
scheduler, seed, loss de treino, loss de validação, perplexidade e acerto por
categoria. O script `prepare_corpus.py` remove duplicatas e verifica
vazamento entre treino e validação.

## Ponto de partida no repositório

```bash
.venv/bin/python -m phase22_language_model.prepare_corpus \
  --input corpus/livros.txt \
  --input corpus/artigos_completos.jsonl \
  --validation-input corpus/validation.txt \
  --train-output data/corpus_train.txt \
  --validation-output data/corpus_validation.txt

.venv/bin/python -m phase22_language_model.train_lm \
  --text data/corpus_train.txt \
  --validation-text data/corpus_validation.txt \
  --tokenizer sentencepiece \
  --output-dir artifacts/language_model/pretraining
```

Continued pretraining reutiliza tokenizer, configuração e pesos:

```bash
.venv/bin/python -m phase22_language_model.train_lm \
  --init-dir artifacts/language_model/pretraining \
  --text data/domain_math_train.txt \
  --validation-text data/domain_math_validation.txt \
  --output-dir artifacts/language_model/continued_math
```

Depois, instruction tuning parte do checkpoint especializado:

```bash
.venv/bin/python -m phase22_language_model.train_qa \
  --init-dir artifacts/language_model/continued_math \
  --data data/instructions_train.jsonl \
  --output-dir artifacts/language_model/instruction_tuning
```

As métricas por categoria são geradas separadamente:

```bash
.venv/bin/python -m phase22_language_model.evaluate_categories \
  --model-dir artifacts/language_model/instruction_tuning \
  --data data/mmlu_validation.jsonl
```

O texto integral deve ser fornecido pelo usuário a partir de fontes com
licença adequada; metadados, DOI e resumos não são tratados como artigo.

O pipeline agora agrupa corpus textual por parágrafos/blocos, ajusta o
tokenizer somente no treino, usa early stopping por `val_loss` e salva
perplexidade em cada checkpoint. O instruction tuning aceita
`--validation-data` e produz as mesmas métricas.
