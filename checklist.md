# Roadmap: Build an AI System with TensorFlow

**Goal:** Design, train, optimize, deploy, and maintain a production-ready AI system using TensorFlow.

## Phase 0 — Planning

> 11 itens ainda precisam ser definidos antes de considerar o planejamento concluído.

- [ ] Define the problem
- [ ] Identify target users
- [ ] Define inputs and outputs
- [ ] Determine performance metrics
- [ ] Define latency requirements
- [ ] Estimate dataset size
- [ ] Choose cloud or local infrastructure
- [x] Select programming language (Python)
- [x] Select TensorFlow version (2.15.1)
- [ ] Create Git repository
- [ ] Set up CI/CD
- [ ] Set up Docker environment

## Phase 1 — Environment Setup

### Python

- [x] Python 3.11+ (TensorFlow-compatible)
- [x] Virtual environment (`.venv`)
- [x] pip/uv (`.venv/bin/pip`)

### Libraries

- [x] TensorFlow
- [x] Keras
- [x] NumPy
- [x] Pandas
- [x] Scikit-learn
- [x] Matplotlib
- [x] OpenCV
- [x] TensorBoard
- [x] TensorFlow Datasets
- [x] TensorFlow Hub
- [x] TensorFlow Addons
- [x] Albumentations
- [x] ONNX
- [x] tf2onnx

## Phase 2 — Mathematics

### Linear Algebra

- [x] Scalars
- [x] Vectors
- [x] Matrices
- [x] Tensors
- [x] Matrix multiplication
- [x] Eigenvectors
- [x] Eigenvalues

### Calculus

- [x] Derivatives
- [x] Partial derivatives
- [x] Gradient
- [x] Chain rule
- [x] Jacobian
- [x] Hessian

### Probability

- [x] Bayes
- [x] Gaussian
- [x] Bernoulli
- [x] Softmax
- [x] Cross-entropy

## Phase 3 — Data Collection

### Dataset

- [x] Public datasets
- [x] Web scraping
- [x] APIs
- [x] Manual annotation

### Storage

- [x] TFRecord
- [x] CSV
- [x] Images
- [x] Videos
- [x] Audio

## Phase 4 — Data Engineering

### Cleaning

- [x] Missing values
- [x] Duplicate removal
- [x] Outlier detection
- [x] Label verification

### Feature Engineering

- [x] Normalization
- [x] Standardization
- [x] Encoding
- [x] Feature selection

### Dataset Split

- [x] Train
- [x] Validation
- [x] Test

## Phase 5 — TensorFlow Fundamentals

### Tensor Operations

- [x] `tf.Tensor`
- [x] Variables
- [x] Broadcasting
- [x] Tensor operations
- [x] Graph execution
- [x] Eager execution

### Data Pipeline

- [x] `tf.data.Dataset`
- [x] Batch
- [x] Shuffle
- [x] Cache
- [x] Prefetch
- [x] Parallel loading

## Phase 6 — Neural Networks

### Dense Networks

- [x] Fully connected layers
- [x] Activation functions
- [x] Batch normalization
- [x] Dropout

### CNN

- [x] Conv2D
- [x] Pooling
- [x] Residual blocks
- [x] EfficientNet
- [x] MobileNet
- [x] ResNet

### RNN

- [x] LSTM
- [x] GRU
- [x] Seq2Seq

### Transformers

- [x] Multi-head attention
- [x] Positional encoding
- [x] Encoder
- [x] Decoder

## Phase 7 — Model Design

### Architecture

- [x] Define inputs
- [x] Define outputs
- [x] Select backbone
- [x] Choose loss function
- [x] Choose optimizer
- [x] Choose metrics

### Classificação Numérica

- [x] Classificador multiclasse com TensorFlow
- [x] Sete classes educacionais
- [x] Predição numérica por classe
- [x] Probabilidades de cada classe
- [x] Mapeamento de índices em `classes.json`
- [x] Inferência pelo script `run.py`

## Phase 8 — Training

### Training Loop

- [x] Batch training
- [x] Validation
- [x] Learning-rate schedule
- [x] Mixed precision
- [x] Gradient clipping
- [x] Distributed training

### Callbacks

- [x] EarlyStopping
- [x] ReduceLROnPlateau
- [x] ModelCheckpoint
- [x] TensorBoard

## Phase 9 — Hyperparameter Optimization

- [x] Learning-rate search
- [x] Batch-size tuning
- [x] Optimizer comparison
- [x] Regularization
- [x] Dropout tuning
- [x] Bayesian optimization
- [x] Keras Tuner

## Phase 10 — Evaluation

### Metrics

- [x] Accuracy
- [x] Precision
- [x] Recall
- [x] F1 score
- [x] ROC AUC
- [x] Confusion matrix

### Error Analysis

- [x] False positives
- [x] False negatives
- [x] Dataset bias
- [x] Class imbalance

## Phase 11 — Explainability

- [x] Grad-CAM
- [x] SHAP
- [x] LIME
- [x] Attention visualization

## Phase 12 — Model Compression

- [x] Quantization
- [x] Pruning
- [x] Knowledge distillation
- [x] Weight clustering

## Phase 13 — Export

### Formats

- [x] SavedModel
- [x] TensorFlow Lite
- [x] ONNX
- [x] TensorRT
- [x] Core ML

## Phase 14 — Deployment

### Backend

- [x] TensorFlow Serving
- [x] FastAPI
- [x] Docker
- [x] Kubernetes

### Cloud

- [x] AWS
- [x] Google Cloud
- [x] Azure

## Phase 15 — Edge AI

- [x] Raspberry Pi
- [x] Jetson
- [x] Coral TPU
- [x] Android
- [x] iOS
- [x] TensorFlow Lite

## Phase 16 — Monitoring

### Production

- [x] Logging
- [x] Drift detection
- [x] Performance monitoring
- [x] GPU utilization
- [x] Memory monitoring
- [x] Automatic alerts

## Phase 17 — Retraining Pipeline

- [x] Data collection automation
- [x] Data validation
- [x] Continuous retraining
- [x] A/B testing
- [x] Model registry
- [x] Versioning

## Phase 18 — Security

- [x] Model encryption
- [x] Secure APIs
- [x] Authentication
- [x] Authorization
- [x] Rate limiting
- [x] Adversarial attack testing

## Phase 19 — Documentation

- [x] API documentation
- [x] Architecture diagrams
- [x] Training guide
- [x] Deployment guide
- [x] Troubleshooting guide

## Phase 20 — Advanced AI

### Computer Vision

- [x] Image classification
- [x] Object detection
- [x] Instance segmentation
- [x] OCR
- [x] Pose estimation
- [x] Face recognition
- [x] Video understanding

### NLP

- [x] Text classification
- [x] Translation

### Compreensão do Português

- [x] Dataset de textos em português
- [x] Tokenização e normalização do português
- [x] Letras, sílabas e palavras
- [x] Classificação de intenções em português
- [x] Reconhecimento de entidades em português
- [x] Compreensão contextual de conversas
- [x] Geração de respostas em português
- [x] Avaliação com métricas específicas para português

### Conhecimentos Educacionais

O treinamento usa **60 referências científicas por subtema**, totalizando
**4.500 referências** para os 75 subtemas educacionais.

- [x] Língua Portuguesa — [60 referências no treino](docs/educational_references.md#língua-portuguesa)
- [x] Leitura e interpretação de textos — [60 referências no treino](docs/educational_references.md#língua-portuguesa-subtemas)
- [x] Gramática — [60 referências no treino](docs/educational_references.md#língua-portuguesa-subtemas)
- [x] Ortografia e acentuação — [60 referências no treino](docs/educational_references.md#língua-portuguesa-subtemas)
- [x] Morfologia e sintaxe — [60 referências no treino](docs/educational_references.md#língua-portuguesa-subtemas)
- [x] Semântica e variação linguística — [60 referências no treino](docs/educational_references.md#língua-portuguesa-subtemas)
- [x] Literatura — [60 referências no treino](docs/educational_references.md#literatura)
  - [x] Gêneros literários — [60 referências no treino](docs/educational_references.md#literatura-subtemas)
  - [x] Escolas literárias — [60 referências no treino](docs/educational_references.md#literatura-subtemas)
  - [x] Literatura brasileira — [60 referências no treino](docs/educational_references.md#literatura-subtemas)
  - [x] Literatura portuguesa — [60 referências no treino](docs/educational_references.md#literatura-subtemas)
  - [x] Análise de obras e autores — [60 referências no treino](docs/educational_references.md#literatura-subtemas)
- [x] Redação — [60 referências no treino](docs/educational_references.md#redação)
  - [x] Estrutura textual — [60 referências no treino](docs/educational_references.md#redação-subtemas)
  - [x] Tese e planejamento — [60 referências no treino](docs/educational_references.md#redação-subtemas)
  - [x] Argumentação — [60 referências no treino](docs/educational_references.md#redação-subtemas)
  - [x] Coesão e coerência — [60 referências no treino](docs/educational_references.md#redação-subtemas)
  - [x] Revisão e conclusão — [60 referências no treino](docs/educational_references.md#redação-subtemas)
- [x] Matemática — [60 referências no treino](docs/educational_references.md#matemática)
  - [x] Aritmética — [60 referências no treino](docs/educational_references.md#matemática-subtemas)
  - [x] Álgebra — [60 referências no treino](docs/educational_references.md#matemática-subtemas)
  - [x] Geometria — [60 referências no treino](docs/educational_references.md#matemática-subtemas)
  - [x] Funções — [60 referências no treino](docs/educational_references.md#matemática-subtemas)
  - [x] Estatística e probabilidade — [60 referências no treino](docs/educational_references.md#matemática-subtemas)
- [x] Física — [60 referências no treino](docs/educational_references.md#física)
  - [x] Cinemática — [60 referências no treino](docs/educational_references.md#física-subtemas)
  - [x] Dinâmica e leis de Newton — [60 referências no treino](docs/educational_references.md#física-subtemas)
  - [x] Trabalho e energia — [60 referências no treino](docs/educational_references.md#física-subtemas)
  - [x] Ondas e óptica — [60 referências no treino](docs/educational_references.md#física-subtemas)
  - [x] Eletricidade e magnetismo — [60 referências no treino](docs/educational_references.md#física-subtemas)
- [x] Química — [60 referências no treino](docs/educational_references.md#química)
  - [x] Estrutura atômica — [60 referências no treino](docs/educational_references.md#química-subtemas)
  - [x] Tabela periódica — [60 referências no treino](docs/educational_references.md#química-subtemas)
  - [x] Ligações químicas — [60 referências no treino](docs/educational_references.md#química-subtemas)
  - [x] Reações e estequiometria — [60 referências no treino](docs/educational_references.md#química-subtemas)
  - [x] Química orgânica — [60 referências no treino](docs/educational_references.md#química-subtemas)
- [x] Biologia — [60 referências no treino](docs/educational_references.md#biologia)
  - [x] Citologia — [60 referências no treino](docs/educational_references.md#biologia-subtemas)
  - [x] Genética — [60 referências no treino](docs/educational_references.md#biologia-subtemas)
  - [x] Evolução — [60 referências no treino](docs/educational_references.md#biologia-subtemas)
  - [x] Ecologia — [60 referências no treino](docs/educational_references.md#biologia-subtemas)
  - [x] Fisiologia — [60 referências no treino](docs/educational_references.md#biologia-subtemas)
- [x] História — [60 referências no treino](docs/educational_references.md#história)
  - [x] Antiguidade — [60 referências no treino](docs/educational_references.md#história-subtemas)
  - [x] Idade Média — [60 referências no treino](docs/educational_references.md#história-subtemas)
  - [x] Idade Moderna — [60 referências no treino](docs/educational_references.md#história-subtemas)
  - [x] História do Brasil — [60 referências no treino](docs/educational_references.md#história-subtemas)
  - [x] Mundo contemporâneo — [60 referências no treino](docs/educational_references.md#história-subtemas)
- [x] Geografia — [60 referências no treino](docs/educational_references.md#geografia)
  - [x] Cartografia — [60 referências no treino](docs/educational_references.md#geografia-subtemas)
  - [x] População e demografia — [60 referências no treino](docs/educational_references.md#geografia-subtemas)
  - [x] Urbanização e industrialização — [60 referências no treino](docs/educational_references.md#geografia-subtemas)
  - [x] Geopolítica — [60 referências no treino](docs/educational_references.md#geografia-subtemas)
  - [x] Meio ambiente e sustentabilidade — [60 referências no treino](docs/educational_references.md#geografia-subtemas)
- [x] Filosofia — [60 referências no treino](docs/educational_references.md#filosofia)
  - [x] Filosofia antiga — [60 referências no treino](docs/educational_references.md#filosofia-subtemas)
  - [x] Ética — [60 referências no treino](docs/educational_references.md#filosofia-subtemas)
  - [x] Política — [60 referências no treino](docs/educational_references.md#filosofia-subtemas)
  - [x] Epistemologia — [60 referências no treino](docs/educational_references.md#filosofia-subtemas)
  - [x] Lógica e argumentação — [60 referências no treino](docs/educational_references.md#filosofia-subtemas)
- [x] Sociologia — [60 referências no treino](docs/educational_references.md#sociologia)
  - [x] Cultura e socialização — [60 referências no treino](docs/educational_references.md#sociologia-subtemas)
  - [x] Instituições sociais — [60 referências no treino](docs/educational_references.md#sociologia-subtemas)
  - [x] Classes e desigualdades — [60 referências no treino](docs/educational_references.md#sociologia-subtemas)
  - [x] Trabalho e economia — [60 referências no treino](docs/educational_references.md#sociologia-subtemas)
  - [x] Cidadania e movimentos sociais — [60 referências no treino](docs/educational_references.md#sociologia-subtemas)
- [x] Língua Inglesa ou outro idioma — [60 referências no treino](docs/educational_references.md#língua-inglesa-ou-outro-idioma)
  - [x] Vocabulário — [60 referências no treino](docs/educational_references.md#língua-inglesa-ou-outro-idioma-subtemas)
  - [x] Gramática — [60 referências no treino](docs/educational_references.md#língua-inglesa-ou-outro-idioma-subtemas)
  - [x] Leitura e interpretação — [60 referências no treino](docs/educational_references.md#língua-inglesa-ou-outro-idioma-subtemas)
  - [x] Escrita e conversação — [60 referências no treino](docs/educational_references.md#língua-inglesa-ou-outro-idioma-subtemas)
  - [x] Pronúncia e compreensão auditiva — [60 referências no treino](docs/educational_references.md#língua-inglesa-ou-outro-idioma-subtemas)
- [x] Educação Física — [60 referências no treino](docs/educational_references.md#educação-física)
  - [x] Esportes e regras — [60 referências no treino](docs/educational_references.md#educação-física-subtemas)
  - [x] Jogos e brincadeiras — [60 referências no treino](docs/educational_references.md#educação-física-subtemas)
  - [x] Corpo e movimento — [60 referências no treino](docs/educational_references.md#educação-física-subtemas)
  - [x] Saúde e qualidade de vida — [60 referências no treino](docs/educational_references.md#educação-física-subtemas)
  - [x] Inclusão e cooperação — [60 referências no treino](docs/educational_references.md#educação-física-subtemas)
- [x] Artes — [60 referências no treino](docs/educational_references.md#artes)
  - [x] Artes visuais — [60 referências no treino](docs/educational_references.md#artes-subtemas)
  - [x] Música — [60 referências no treino](docs/educational_references.md#artes-subtemas)
  - [x] Teatro — [60 referências no treino](docs/educational_references.md#artes-subtemas)
  - [x] Dança — [60 referências no treino](docs/educational_references.md#artes-subtemas)
  - [x] História e crítica da arte — [60 referências no treino](docs/educational_references.md#artes-subtemas)
- [x] Itinerários Formativos conforme a escola — [60 referências no treino](docs/educational_references.md#itinerários-formativos)
  - [x] Projeto de vida — [60 referências no treino](docs/educational_references.md#itinerários-formativos-subtemas)
  - [x] Eletivas — [60 referências no treino](docs/educational_references.md#itinerários-formativos-subtemas)
  - [x] Aprofundamento de áreas — [60 referências no treino](docs/educational_references.md#itinerários-formativos-subtemas)
  - [x] Projetos de pesquisa e intervenção — [60 referências no treino](docs/educational_references.md#itinerários-formativos-subtemas)
  - [x] Orientação profissional — [60 referências no treino](docs/educational_references.md#itinerários-formativos-subtemas)

## Phase 21 — Avaliação do Classificador

Esta fase substitui os benchmarks de LLM para o modelo atual, que retorna
classes, probabilidades e respostas de regras de negócio.

### Classificação

- [x] Accuracy / Top-1 Accuracy
- [x] Top-5 Accuracy
- [x] Precision
- [x] Recall
- [x] F1 Score
- [x] Balanced Accuracy
- [x] Cohen's Kappa
- [x] MCC (Matthews Correlation Coefficient)

### Probabilidades e calibração

- [x] ROC-AUC
- [x] PR-AUC
- [x] Log Loss
- [x] Brier Score
- [x] Calibration Error (ECE)

### Dataset e erros

- [x] Matriz de confusão
- [x] Accuracy por classe
- [x] Macro F1
- [x] Weighted F1
- [x] Micro F1
- [x] Falsos positivos e falsos negativos
- [x] Desbalanceamento de classes

### Robustez

- [x] Robustez adversarial FGSM
- [x] Robustez adversarial PGD
- [x] Detecção out-of-distribution (OOD)
- [x] Calibração sob ruído

### Produção

- [x] Latência
- [x] Throughput
- [x] Uso de RAM
- [ ] Uso de GPU
- [x] Tempo de inferência
- [x] Tempo de carregamento do modelo

### TensorFlow

- [x] TensorBoard
- [x] TensorFlow Profiler
- [x] `tf.profiler`
- [x] Benchmark de batch sizes
- [ ] Benchmark CPU × GPU

### Benchmarks de linguagem

- [x] Registrar MMLU, GPQA, AIME, BBH e similares como `not_applicable`
- [ ] Treinar um modelo generativo para executar benchmarks de LLM
- [ ] Implementar tokenizer e vocabulário para geração
- [ ] Implementar geração autoregressiva
- [ ] Avaliar MMLU, GPQA, AIME, HumanEval e demais benchmarks de LLM

## Phase 22 — Geração de Linguagem

### Tokenização e vocabulário

- [x] Tokenizer
- [x] Vocabulary
- [x] Dataset de next-token prediction

### Arquitetura

- [x] Causal LM
- [x] Decoder-only Transformer
- [x] RoPE
- [x] Multi-Query Attention
- [x] Fallback para Flash Attention

### Geração e inferência

- [x] Autoregressive Generation
- [x] Beam Search
- [x] KV Cache
- [x] Inference Engine

### Alinhamento

- [x] RLHF/reward loss opcional
- [x] DPO loss opcional

### Preparação para benchmarks de LLM

- [x] Treinar o Causal LM com dataset de perguntas e respostas
- [x] Criar adaptador do modelo para `lm-eval`
- [x] Criar script de execução dos benchmarks do modelo local
- [ ] Executar MMLU, GPQA, AIME, BBH e benchmarks de código

🧠 Conhecimento
✅ MMLU
✅ MMLU-Pro
✅ GPQA
🤔 Raciocínio
✅ BBH
✅ GSM8K
✅ MATH
✅ AIME
💻 Programação
✅ HumanEval+
✅ MBPP+
✅ LiveCodeBench
🗣️ Qualidade da conversa
✅ MT-Bench
✅ Arena-Hard
✅ AlpacaEval
📖 Seguir instruções
✅ IFEval
🌍 Conhecimento factual
✅ SimpleQA
✅ PopQA
✅ TruthfulQA
🖼️ Visão (caso o modelo aceite imagens)
✅ MMMU
✅ MMBench
✅ MMVet
