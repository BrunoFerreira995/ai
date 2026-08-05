# Phase 7 — Model Design

This package centralizes the decisions required to define a supervised model:

- Input and output shapes
- Backbone selection: dense, CNN, residual CNN, EfficientNet, MobileNet, or ResNet
- Loss function
- Optimizer and learning rate
- Evaluation metrics

Example:

```python
from phase7_model_design import ModelConfig, build_classifier, compile_model

config = ModelConfig(
    input_shape=(32, 32, 3),
    num_classes=10,
    backbone="residual_cnn",
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=("accuracy",),
)
model = compile_model(build_classifier(config), config)
```

Run the tests:

```bash
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -m unittest discover -s phase7_model_design -p 'test_*.py'
```
