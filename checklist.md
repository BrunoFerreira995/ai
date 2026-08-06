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

- [x] Língua Portuguesa
- [x] Leitura e interpretação de textos
- [x] Gramática
- [x] Ortografia e acentuação
- [x] Morfologia e sintaxe
- [x] Semântica e variação linguística
- [x] Literatura
  - [x] Gêneros literários
  - [x] Escolas literárias
  - [x] Literatura brasileira
  - [x] Literatura portuguesa
  - [x] Análise de obras e autores
- [x] Redação
  - [x] Estrutura textual
  - [x] Tese e planejamento
  - [x] Argumentação
  - [x] Coesão e coerência
  - [x] Revisão e conclusão
- [x] Matemática
  - [x] Aritmética
  - [x] Álgebra
  - [x] Geometria
  - [x] Funções
  - [x] Estatística e probabilidade
- [x] Física
  - [x] Cinemática
  - [x] Dinâmica e leis de Newton
  - [x] Trabalho e energia
  - [x] Ondas e óptica
  - [x] Eletricidade e magnetismo
- [x] Química
  - [x] Estrutura atômica
  - [x] Tabela periódica
  - [x] Ligações químicas
  - [x] Reações e estequiometria
  - [x] Química orgânica
- [x] Biologia
  - [x] Citologia
  - [x] Genética
  - [x] Evolução
  - [x] Ecologia
  - [x] Fisiologia
- [x] História
  - [x] Antiguidade
  - [x] Idade Média
  - [x] Idade Moderna
  - [x] História do Brasil
  - [x] Mundo contemporâneo
- [x] Geografia
  - [x] Cartografia
  - [x] População e demografia
  - [x] Urbanização e industrialização
  - [x] Geopolítica
  - [x] Meio ambiente e sustentabilidade
- [x] Filosofia
  - [x] Filosofia antiga
  - [x] Ética
  - [x] Política
  - [x] Epistemologia
  - [x] Lógica e argumentação
- [x] Sociologia
  - [x] Cultura e socialização
  - [x] Instituições sociais
  - [x] Classes e desigualdades
  - [x] Trabalho e economia
  - [x] Cidadania e movimentos sociais
- [x] Língua Inglesa ou outro idioma
  - [x] Vocabulário
  - [x] Gramática
  - [x] Leitura e interpretação
  - [x] Escrita e conversação
  - [x] Pronúncia e compreensão auditiva
- [x] Educação Física
  - [x] Esportes e regras
  - [x] Jogos e brincadeiras
  - [x] Corpo e movimento
  - [x] Saúde e qualidade de vida
  - [x] Inclusão e cooperação
- [x] Artes
  - [x] Artes visuais
  - [x] Música
  - [x] Teatro
  - [x] Dança
  - [x] História e crítica da arte
- [x] Itinerários Formativos conforme a escola
  - [x] Projeto de vida
  - [x] Eletivas
  - [x] Aprofundamento de áreas
  - [x] Projetos de pesquisa e intervenção
  - [x] Orientação profissional

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
