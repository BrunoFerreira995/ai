# Corpus textual

Este diretório recebe os textos usados no pretraining. Os arquivos de exemplo
abaixo são apenas um contrato de estrutura; substitua-os por conteúdo real e
licenciado antes de treinar o modelo.

Estrutura esperada:

```text
corpus/
├── livros.txt                 # um documento ou trecho por linha/bloco
├── artigos_completos.jsonl    # cada linha: {"text": "..."}
└── validation.txt             # documentos separados do treino
```

`validation.txt` deve conter documentos que não aparecem em nenhum arquivo de
treino. Linhas idênticas encontradas nos dois conjuntos são removidas da
validação. Títulos, DOI, URLs, resumos e referências bibliográficas não substituem
o texto integral de livros ou artigos.

Para um teste local rápido usando documentação já existente no projeto:

```bash
python -m phase22_language_model.prepare_corpus \
  --input README.md \
  --input docs/ARCHITECTURE.md \
  --validation-input docs/TRAINING.md \
  --train-output data/corpus_train.txt \
  --validation-output data/corpus_validation.txt
```

Esse teste valida o pipeline, mas não é um corpus suficiente para melhorar o
MMLU.
