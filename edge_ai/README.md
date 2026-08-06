# Phase 15 — Edge AI

## TFLite inference

The portable executor works with TensorFlow Lite on CPU and can load a
hardware delegate when the device provides one:

```python
import numpy as np
from edge_ai import EdgeInterpreter

model = EdgeInterpreter("artifacts/model.tflite")
output = model.predict(np.zeros(model.input_shape, dtype=np.float32))
print(output)
```

## Platform targets

- **Raspberry Pi:** install `tflite-runtime` and run the same Python executor.
- **Jetson:** use the Jetson TensorFlow/TFLite runtime or convert the model to TensorRT.
- **Coral TPU:** install `tflite-runtime` and pass the Edge TPU delegate path,
  usually `/usr/lib/aarch64-linux-gnu/libedgetpu.so.1`.
- **Android:** use the TensorFlow Lite Android Interpreter and bundle `model.tflite` in assets.
- **iOS:** use the TensorFlowLite Swift pod and add `model.tflite` to the app bundle.

Hardware delegates are optional; the CPU path remains the portable fallback.
