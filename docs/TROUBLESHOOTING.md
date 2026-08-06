# Troubleshooting

## `Dataset não encontrado`

Use o dataset de demonstração:

```bash
./train.sh
```

Ou crie `data/dataset.npz` com arrays `x` e `y` antes de usar `--data`.

## `No module named tensorflow`

Ative ou recrie o ambiente:

```bash
./install.sh
source .venv/bin/activate
```

## `Modelo não encontrado`

Treine o modelo antes da inferência:

```bash
./train.sh
./run.py
```

## API retorna `401`

Se `API_KEY` estiver definida, envie o header:

```bash
-H 'X-API-Key: sua-chave'
```

## Avisos de GPU ou optimizer no Mac

Esses avisos são informativos. O projeto usa CPU quando CUDA/ROCm não estão
disponíveis.
