# Target Platform Capabilities (TPC)

## About 

TPC is our way of describing the hardware that will be used to run and infer with models that are
optimized using the MCT.

The TPC includes different parameters that are relevant to the
 hardware during inference (e.g., number of bits used
in some operator for its weights/activations, fusing patterns, etc.)


## Supported Target Platform Models 

Currently, MCT contains three target-platform models
(new models can be created and used by users as demonstrated [here](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/target_platform_capabilities/tpc_models/imx500_tpc/v1/tpc.py)):
- [IMX500](https://developer.sony.com/develop/imx500/)
- [TFLite](https://www.tensorflow.org/lite/performance/quantization_spec)
- [QNNPACK](https://github.com/pytorch/QNNPACK)

The default target-platform model is [IMX500](https://developer.sony.com/develop/imx500/), quantizes activations using 8 bits with power-of-two thresholds for 
activations and symmetric threshold for weights.
For mixed-precision quantization it uses either 2, 4, or 8 bits for quantizing the operators.
One may view the full default target-platform model and its parameters [here](./tpc_models/imx500_tpc/v1/tpc.py).

[TFLite](./tpc_models/tflite_tpc/v1/tpc.py) and [QNNPACK](./tpc_models/qnnpack_tpc/v1/tpc.py) models were created similarly and were used to create two TPCs: One for Keras TPC and one for PyTorch TPC (for each model, this 8 in total).

## Usage

The simplest way to initiate a TPC and use it in MCT is by using the function [get_target_platform_capabilities](https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/methods/get_target_platform_capabilities.html#ug-get-target-platform-capabilities).

For example:

```python
from tensorflow.keras.applications.mobilenet import MobileNet
import model_compression_toolkit as mct
import numpy as np

# Get a TPC object that models the hardware for the quantized model inference.
# The model determines the quantization methods to use during the MCT optimization process.
# Here, we use the default (imx500) target-platform model attached to a Tensorflow
# layers representation.
target_platform_cap = mct.get_target_platform_capabilities('tensorflow', 'default')

quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(MobileNet(),
                                                                              lambda: [np.random.randn(1, 224, 224, 3)],
                                                                              # Random representative dataset 
                                                                              target_platform_capabilities=target_platform_cap)
```

Similarly, you can retrieve IMX500, TFLite and QNNPACK target-platform models for Keras and PyTorch frameworks.

For more information and examples, we highly recommend you to visit our [project website](https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/modules/target_platform_capabilities.html#ug-target-platform-capabilities).
