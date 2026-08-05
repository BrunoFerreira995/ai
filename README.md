# AI System with TensorFlow

Projeto de estudo e implementação de uma IA com TensorFlow, classificação de
intenções em português e conhecimentos educacionais.

## Começar

```bash
./install.sh
./train.sh
./run.py
./chat.py
```

O treinamento acima usa o dataset de demonstração e cria os arquivos em
`artifacts/`. Não use `--data` sem ter criado um arquivo `.npz`.

## Ativar o ambiente Python

Depois da instalação, ative a `.venv` com:

```bash
source .venv/bin/activate
```

Para sair do ambiente:

```bash
deactivate
```

## Scripts

| Script | Função |
| --- | --- |
| `install.sh` | Instala Python 3.11, ambiente virtual e dependências |
| `train.sh` | Treina o classificador e exporta o modelo |
| `run.py` | Executa uma previsão |
| `chat.py` | Abre o chat educacional em português |

Para ajustar o treinamento:

```bash
./train.sh --epochs 10 --batch-size 32
```

Para executar uma entrada própria:

```bash
./run.py --input data/sample.npy
```

## Dados próprios

O `train.py` espera um arquivo `.npz` com arrays numéricos `x` e `y`:

```python
import numpy as np
np.savez("data/dataset.npz", x=x, y=y)
```

Depois:

```bash
./train.sh --data data/dataset.npz --epochs 10
```

## Chat em português

```bash
./chat.py
```

O chat reconhece intenções, verbos, sinônimos, entidades simples e disciplinas
educacionais. Digite `sair` para encerrar.

As áreas educacionais incluem Língua Portuguesa, Literatura, Redação,
Matemática, Física, Química, Biologia, História, Geografia, Filosofia,
Sociologia, idiomas, Educação Física, Artes e Itinerários Formativos.

## Testes

```bash
.venv/bin/python -m unittest discover -p 'test_*.py'
```

## Artefatos

O treinamento gera:

```text
artifacts/
├── checkpoints/best.keras
├── metrics.json
├── model.onnx
├── model.tflite
└── saved_model/
```
