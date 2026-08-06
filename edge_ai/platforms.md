# Edge platform setup

## Raspberry Pi

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tflite-runtime numpy
```

## Jetson

Use the NVIDIA JetPack-provided runtime and validate the model with the
device's TensorRT/TFLite installation before deployment.

## Coral TPU

```bash
sudo apt install libedgetpu1-std
pip install tflite-runtime
```

Then pass the Edge TPU shared library to `EdgeInterpreter`.

## Android and iOS

Use the native TensorFlow Lite APIs and keep the model input preprocessing
identical to the Python training pipeline.
