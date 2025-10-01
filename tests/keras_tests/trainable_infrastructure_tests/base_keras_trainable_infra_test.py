# Copyright 2023 Sony Semiconductor Solutions, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import TensorShape
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from mct_quantizers import QuantizationTarget, mark_quantizer
from model_compression_toolkit.trainable_infrastructure import BaseKerasTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC])
class IdentityWeightsQuantizer(BaseKerasTrainableQuantizer):
    """
    A dummy quantizer for test usage - "quantize" the layer's weights to the original weights
    """
    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        super().__init__(quantization_config)

    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: Any) -> Dict[str, tf.Variable]:
        return {}

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        return inputs


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC])
class ZeroWeightsQuantizer(BaseKerasTrainableQuantizer):
    """
    A dummy quantizer for test usage - "quantize" the layer's weights to 0
    """
    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        super().__init__(quantization_config)

    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: Any) -> Dict[str, tf.Variable]:
        return {}

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        return inputs * 0


@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC])
class ZeroActivationsQuantizer(BaseKerasTrainableQuantizer):
    """
    A dummy quantizer for test usage - "quantize" the layer's activation to 0
    """
    def __init__(self, quantization_config: TrainableQuantizerActivationConfig):
        super().__init__(quantization_config)

    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: Any) -> Dict[str, tf.Variable]:
        return {}

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool = True) -> tf.Tensor:
        return inputs * 0


class BaseKerasTrainableInfrastructureTest:
    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3)):
        self.unit_test = unit_test
        self.val_batch_size = val_batch_size
        self.num_calibration_iter = num_calibration_iter
        self.num_of_inputs = num_of_inputs
        self.input_shape = (val_batch_size,) + input_shape

    def generate_inputs(self):
        return [np.random.randn(*in_shape) for in_shape in self.get_input_shapes()]

    def get_input_shapes(self):
        return [self.input_shape for _ in range(self.num_of_inputs)]

    def get_wrapper(self, layer, weight_quantizers={}, activation_quantizers=[]):
        return KerasTrainableQuantizationWrapper(layer, weight_quantizers, activation_quantizers)

    def get_weights_quantization_config(self):
        return TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                               weights_n_bits=9,
                                               weights_quantization_params={},
                                               enable_weights_quantization=True,
                                               weights_channels_axis=3,
                                               weights_per_channel_threshold=True,
                                               min_threshold=0)

    def get_activation_quantization_config(self):
        return TrainableQuantizerActivationConfig(activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                                  activation_n_bits=8,
                                                  activation_quantization_params={},
                                                  enable_activation_quantization=True,
                                                  min_threshold=0)
