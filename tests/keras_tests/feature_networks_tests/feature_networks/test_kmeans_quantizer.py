# Copyright 2021 Sony Semiconductor Solutions, Inc. All rights reserved.
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

import unittest

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common.network_editors.node_filters import NodeNameFilter
from model_compression_toolkit.core.common.network_editors.actions import EditRule, ChangeCandidatesWeightsQuantConfigAttr
from model_compression_toolkit.core.common.quantization.quantizers.uniform_quantizers import power_of_two_quantizer
from model_compression_toolkit.core.common.quantization.quantization_params_generation.power_of_two_selection import power_of_two_selection_tensor
import model_compression_toolkit as mct
import tensorflow as tf

from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np

from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


def get_uniform_weights(kernel, in_channels, out_channels):
    return np.array([i - np.round((in_channels * kernel * kernel * out_channels) / 2) for i in
                     range(in_channels * kernel * kernel * out_channels)]).reshape(
        [out_channels, kernel, kernel, in_channels]).transpose(1, 2, 3, 0)


def get_zero_as_weights(kernel, in_channels, out_channels):
    return np.zeros([kernel, kernel, in_channels, out_channels])


class KmeansQuantizerTestBase(BaseKerasFeatureNetworkTest):
    '''
    - Check name filter- that only the node with the name changed
    - Check that different quantization methods on the same weights give different results
    '''

    def __init__(self,
                 unit_test,
                 quantization_method: QuantizationMethod.LUT_POT_QUANTIZER,
                 weight_fn=get_uniform_weights,
                 weights_n_bits: int = 3):

        self.quantization_method = quantization_method
        self.weights_n_bits = weights_n_bits
        self.node_to_change_name = 'change'
        self.num_conv_channels = 4
        self.kernel = 3
        self.conv_w = weight_fn(self.kernel, self.num_conv_channels, self.num_conv_channels)
        super().__init__(unit_test, num_calibration_iter=5, val_batch_size=32)

    def get_tpc(self):
        tp = generate_test_tpc({'weights_quantization_method': self.quantization_method,
                                     'weights_n_bits': self.weights_n_bits,
                                     'activation_n_bits': 4})
        return generate_keras_tpc(name="kmean_quantizer_test", tpc=tp)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                           False, False)

    def get_input_shapes(self):
        return [[self.val_batch_size, 16, 16, self.num_conv_channels]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False, name=self.node_to_change_name)(inputs)
        x = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        outputs = layers.Conv2D(self.num_conv_channels, self.kernel, use_bias=False)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.layers[1].set_weights([self.conv_w])
        model.layers[2].set_weights([self.conv_w])
        model.layers[3].set_weights([self.conv_w])
        return model

    def get_debug_config(self):
        return mct.core.DebugConfig(network_editor=[EditRule(filter=NodeNameFilter(self.node_to_change_name),
                                                             action=ChangeCandidatesWeightsQuantConfigAttr(
                                                                 attr_name=KERNEL,
                                                                 weights_quantization_method=QuantizationMethod.POWER_OF_TWO)),
                                                    EditRule(filter=NodeNameFilter(self.node_to_change_name),
                                                             action=ChangeCandidatesWeightsQuantConfigAttr(
                                                                 attr_name=KERNEL,
                                                                 weights_quantization_fn=power_of_two_quantizer)),
                                                    EditRule(filter=NodeNameFilter(self.node_to_change_name),
                                                             action=ChangeCandidatesWeightsQuantConfigAttr(
                                                                 attr_name=KERNEL,
                                                                 weights_quantization_params_fn=power_of_two_selection_tensor)),
                                                    ])

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they were quantized
        # using different methods (but started as the same value)
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue(np.sum(
            np.abs(conv_layers[0].weights[0].numpy() - conv_layers[2].weights[0].numpy())) > 0)


class KmeansQuantizerTest(KmeansQuantizerTestBase):
    '''
    This test checks the chosen quantization method is different that symmetric uniform
    '''

    def __init__(self,
                 unit_test,
                 quantization_method: QuantizationMethod.LUT_POT_QUANTIZER,
                 weights_n_bits: int = 3):
        super().__init__(unit_test, quantization_method, get_uniform_weights, weights_n_bits)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they were quantized
        # using different methods (but started as the same value)
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue(np.sum(
            np.abs(conv_layers[0].get_quantized_weights()[KERNEL] - conv_layers[2].get_quantized_weights()[KERNEL])) > 0)



class KmeansQuantizerNotPerChannelTest(KmeansQuantizerTestBase):
    """
    This test checks the chosen quantization method is different that symmetric uniform when weights are not
    been quantized per channel
    """

    def __init__(self,
                 unit_test,
                 quantization_method: QuantizationMethod.LUT_POT_QUANTIZER,
                 weights_n_bits: int = 3):
        super().__init__(unit_test, quantization_method, get_uniform_weights, weights_n_bits)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(activation_error_method=mct.core.QuantizationErrorMethod.MSE,
                                           weights_error_method=mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=False, weights_bias_correction=False)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they where quantized
        # using different methods (but started as the same value)
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue(np.sum(
            np.abs(conv_layers[0].get_quantized_weights()[KERNEL] - conv_layers[2].get_quantized_weights()[KERNEL])) > 0)


class KmeansQuantizerTestManyClasses(KmeansQuantizerTestBase):
    '''
    This test checks the chosen quantization method is different that symmetric uniform
    '''

    def __init__(self, unit_test, quantization_method: QuantizationMethod.LUT_POT_QUANTIZER, weights_n_bits: int = 8):
        super().__init__(unit_test, quantization_method, get_uniform_weights, weights_n_bits)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they were quantized
        # using different methods (but started as the same value)
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue(
            np.all(np.isclose(float_model.layers[1].weights[0].numpy(), conv_layers[2].weights[0].numpy())))


class KmeansQuantizerTestZeroWeights(KmeansQuantizerTestBase):
    '''
    This test checks the case where all the weight values are zero
    '''

    def __init__(self, unit_test,
                 quantization_method: QuantizationMethod.LUT_POT_QUANTIZER,
                 weights_n_bits: int = 3):
        super().__init__(unit_test, quantization_method, get_zero_as_weights, weights_n_bits)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # check that the two conv's weights have different values since they where quantized
        # using different methods (but started as the same value)
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        self.unit_test.assertTrue(np.sum(np.abs(conv_layers[0].weights[0].numpy())) == 0)
        self.unit_test.assertTrue(np.sum(np.abs(conv_layers[1].weights[0].numpy())) == 0)
        self.unit_test.assertTrue(np.sum(np.abs(conv_layers[2].weights[0].numpy())) == 0)


if __name__ == '__main__':
    unittest.main()
