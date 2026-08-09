# Phase 22 — Language Model

Ative o ambiente antes de executar os comandos:

```bash
source .venv/bin/activate
```

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

Para pretraining, prepare primeiro um corpus textual real. O comando aceita
TXT, JSON e JSONL, remove duplicatas e verifica vazamento entre treino e
validação:

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

`train_lm.py` registra loss, `val_loss`, perplexidade e perplexidade de
validação em `training_metrics.json`. O arquivo de validação deve ser mantido
fora do instruction tuning e dos benchmarks.

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
O dataset educacional inclui 60 referências científicas por subtema, totalizando
4.500 registros bibliográficos. Cada artigo vira um exemplo QA individual, para
que nenhuma referência seja perdida por truncamento da sequência.
O dataset também contém teoria do conhecimento sobre sujeito, objeto e a
relação entre experiência, razão, evidência e conhecimento.

Para novos treinamentos, use o tokenizer BPE SentencePiece:

```bash
.venv/bin/python -m phase22_language_model.train_qa \
  --data data/qa_phase20.jsonl \
  --tokenizer sentencepiece \
  --vocab-size 32000 \
  --output-dir artifacts/language_model
```

Valide o adapter e o cálculo de probabilidades das alternativas:

```bash
.venv/bin/python -m phase22_language_model.validate_adapter \
  --model-dir artifacts/language_model
```

O checkpoint antigo continua compatível com o tokenizer de caracteres. Para
usar SentencePiece, é necessário treinar um novo checkpoint.

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
  --dictionary-variants 4 \
  --references data/educational_references_60.json
.venv/bin/python -m phase22_language_model.train_qa \
  --data data/qa_phase20.jsonl \
  --reference-explanations data/reference_explanations.jsonl \
  --output-dir artifacts/language_model
```

Para incluir as explicações bibliográficas das 4.500 referências:

```bash
.venv/bin/python -m phase20_portuguese_nlp.build_reference_explanations \
  --source data/educational_references_60.json \
  --output data/reference_explanations.jsonl
.venv/bin/python -m phase22_language_model.train_qa \
  --data data/qa_phase20.jsonl \
  --reference-explanations data/reference_explanations.jsonl \
  --output-dir artifacts/language_model
```

Arquivos `.jsonl.gz` também são aceitos. Para o corpus de 1 milhão de linhas,
comece com um subconjunto para não esgotar a memória:

```bash
.venv/bin/python -m phase22_language_model.train_qa \
  --data data/exam_qa_1m.jsonl.gz \
  --max-examples 100000 \
  --output-dir artifacts/language_model
```

É possível combinar datasets repetindo `--data`:

```bash
.venv/bin/python -m phase22_language_model.train_qa \
  --data data/exam_qa_1m.jsonl.gz \
  --data data/qa_phase20.jsonl \
  --max-examples 100000 \
  --dictionary-epochs 2 \
  --word-explanations data/word_explanations.jsonl \
  --reference-explanations data/reference_explanations.jsonl \
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
