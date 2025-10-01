# Copyright 2024 Sony Semiconductor Solutions, Inc. All rights reserved.
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
from typing import Union

import numpy as np
import tensorflow as tf

from mct_quantizers import mark_quantizer, QuantizationTarget, QuantizationMethod
from mct_quantizers.keras.quantizers import ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer
from model_compression_toolkit import constants as C
from model_compression_toolkit.trainable_infrastructure import TrainingMethod, TrainableQuantizerActivationConfig
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.common.constants import THRESHOLD_TENSOR
from model_compression_toolkit.trainable_infrastructure.keras.activation_quantizers import BaseKerasActivationTrainableQuantizer
from model_compression_toolkit.constants import SIGNED
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from model_compression_toolkit.trainable_infrastructure.common.constants import FQ_MIN, FQ_MAX


# moved (and renamed) from model_compression_toolkit/qat/keras/quantizer/ste_rounding/symmetric_ste.py
@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.STE)
class STESymmetricActivationTrainableQuantizer(BaseKerasActivationTrainableQuantizer):

    """
    Trainable constrained quantizer to quantize a layer outputs.
    """

    def __init__(self, quantization_config: TrainableQuantizerActivationConfig, freeze_quant_params: bool = False):
        """
        Initialize a STESymmetricActivationTrainableQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: trainable quantizer config class
            freeze_quant_params: whether to freeze learnable quantization parameters. This is unused here, since there is not any quantizaiton params that are learned.
        """
        super().__init__(quantization_config, freeze_quant_params)
        self.power_of_two = quantization_config.activation_quantization_method == QuantizationMethod.POWER_OF_TWO
        self.threshold_values = quantization_config.activation_quantization_params[C.THRESHOLD]
        self.threshold_shape = np.asarray(self.threshold_values).shape
        self.np_threshold_values = float(self.threshold_values)
        self.signed = quantization_config.activation_quantization_params[SIGNED]
        if self.power_of_two:
            self.np_threshold_values = np.power(2.0,
                                                np.ceil(
                                                    np.log2(np.maximum(self.np_threshold_values, C.MIN_THRESHOLD))))
        self.num_bits = quantization_config.activation_n_bits
        delta = self.np_threshold_values / np.power(2.0, self.num_bits - int(self.signed))
        min_int = -int(self.signed) * (2 ** (self.num_bits - int(self.signed)))
        max_int = (2 ** (self.num_bits - int(self.signed))) - 1
        self.min = delta * min_int
        self.max = delta * max_int

    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: KerasTrainableQuantizationWrapper):
        """
        Add quantizer parameters to the quantizer parameters dictionary

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.
        """
        ptq_threshold_tensor = layer.add_weight(
            name + THRESHOLD_TENSOR,
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        ptq_threshold_tensor.assign(self.np_threshold_values)

        fq_min = layer.add_weight(
            name + FQ_MIN,
            shape=(),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=False)
        fq_min.assign(self.min)

        fq_max = layer.add_weight(
            name + FQ_MAX,
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        fq_max.assign(self.max)

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(THRESHOLD_TENSOR, ptq_threshold_tensor, VariableGroup.QPARAMS)
        self.add_quantizer_variable(FQ_MIN, fq_min, VariableGroup.QPARAMS)
        self.add_quantizer_variable(FQ_MAX, fq_max, VariableGroup.QPARAMS)

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        """
        Quantize a tensor.
        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """

        _min = self.get_quantizer_variable(FQ_MIN)
        _max = self.get_quantizer_variable(FQ_MAX)
        q_tensor = tf.quantization.fake_quant_with_min_max_vars(inputs, _min, _max,
                                                                num_bits=self.num_bits)

        return q_tensor

    def convert2inferable(self) -> Union[ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer]:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            BaseKerasInferableQuantizer object.
        """

        if self.power_of_two:
            pot_threshold = 2 ** np.ceil(np.log2(self.get_quantizer_variable(THRESHOLD_TENSOR)))
            return ActivationPOTInferableQuantizer(num_bits=self.num_bits,
                                                   # In activation quantization is per-tensor only - thus we pass
                                                   # the threshold as a list with a len of 1
                                                   threshold=[pot_threshold],
                                                   signed=self.signed)
        else:
            return ActivationSymmetricInferableQuantizer(num_bits=self.num_bits,
                                                         # In activation quantization is per-tensor only - thus we
                                                         # pass the threshold as a list with a len of 1
                                                         threshold=[
                                                             self.get_quantizer_variable(THRESHOLD_TENSOR).numpy()],
                                                         signed=self.signed)
