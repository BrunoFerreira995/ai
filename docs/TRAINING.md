# Guia de treinamento

## Dataset de demonstração

```bash
./install.sh
./train.sh --epochs 10 --batch-size 32
```

## Dataset próprio

Crie um `.npz` com `x` e `y`, mantendo o mesmo número de amostras:

```python
import numpy as np
np.savez("data/dataset.npz", x=x, y=y)
```

Treine:

```bash
./train.sh --data data/dataset.npz --epochs 10 --output-dir artifacts
```

O processo divide os dados em treino, validação e teste, treina o modelo,
calcula métricas e exporta SavedModel, TFLite e ONNX.
