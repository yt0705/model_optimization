# Copyright 2022 Sony Semiconductor Solutions, Inc. All rights reserved.
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
import math
import typing

import abc

import numpy as np
import tensorflow as tf
from keras.activations import sigmoid, softmax

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from mct_quantizers import KerasActivationQuantizationHolder
from model_compression_toolkit import DefaultDict
from model_compression_toolkit.core import QuantizationConfig, CoreConfig
from model_compression_toolkit.core.keras.constants import SIGMOID, SOFTMAX, BIAS
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, KERAS_KERNEL
from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework import LayerFilterParams
from tests.common_tests.helpers.generate_test_tpc import generate_test_op_qc, generate_test_attr_configs
from tests.keras_tests.exporter_tests.tflite_int8.imx500_int8_tpc import get_op_quantization_configs
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from keras import backend as K

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    ResourceUtilization
from model_compression_toolkit.core.common.user_info import UserInformation
from tests.keras_tests.tpc_keras import get_tpc_with_activation_mp_keras
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


def get_base_mp_nbits_candidates():
    return [(4, 8), (4, 4), (4, 2),
            (8, 8), (8, 4), (8, 2),
            (2, 8), (2, 4), (2, 2)]


class MixedPrecisionActivationBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, activation_layers_idx, num_calibration_iter=1):
        super().__init__(unit_test, num_calibration_iter=num_calibration_iter)
        # for the model that is used here, the two last tensors compose the max cut
        self.max_cut = 10 * 10 * 32 + 13 * 13 * 32
        self.activation_layers_idx = activation_layers_idx

    def get_core_config(self):
        return CoreConfig(quantization_config=QuantizationConfig(
            custom_tpc_opset_to_layer={"Input": CustomOpsetLayers([layers.InputLayer])}))

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())

        # sets all combinations of 2, 4, 8 bits for weights and activations
        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_test")

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                           mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=False,
                                           weights_bias_correction=True,
                                           input_scaling=False,
                                           activation_channel_equalization=False,
                                           custom_tpc_opset_to_layer={"Input": CustomOpsetLayers([layers.InputLayer])})

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

    def get_input_shapes(self):
        return [[self.val_batch_size, 16, 16, 3]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 4)(x)
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    @typing.final
    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # call concrete validation of the specific test
        self._compare(quantized_model, float_model, input_x, quantization_info)
        # make sure the final utilization satisfies the target constraints
        self.unit_test.assertTrue(
            self.get_resource_utilization().is_satisfied_by(quantization_info.final_resource_utilization))

    @abc.abstractmethod
    def _compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # test-specific validation, to be implemented by each test
        raise NotImplementedError()

    def verify_quantization(self, quantized_model, input_x, weights_layers_idx, weights_layers_channels_size,
                            activation_layers_idx, unique_tensor_values):
        # verify weights quantization
        conv_layers = get_layers_from_model_by_type(quantized_model, layers.Conv2D)
        for conv_layer, num_channels in zip(conv_layers, weights_layers_channels_size):
            for j in range(num_channels):  # quantized per channel
                self.unit_test.assertTrue(
                    np.unique(conv_layer.get_quantized_weights()['kernel'][:, :, :, j]).flatten().shape[
                        0] <= unique_tensor_values)

        # verify activation quantization
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[
                        1:]  # skip the input layer
        inp = quantized_model.input  # input placeholder
        out = [layer.output for layer in holder_layers]  # all layer outputs
        get_outputs = K.function([inp], out)
        layer_outs = get_outputs([input_x])

        # verifying fake quant nodes output
        for layer_out in layer_outs:
            self.unit_test.assertTrue(np.unique(layer_out).flatten().shape[0] <= unique_tensor_values)
class MixedPrecisionActivationSearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 4])

    def get_resource_utilization(self):
        return ResourceUtilization(weights_memory=17919, activation_memory=self.max_cut-1)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        # Since the max cut is the last two tensors, one of them have to get 4 bits
        self.unit_test.assertIn(activation_bits, ([8, 4, 8], [8, 8, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationSearch4BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2, 4])

    def get_resource_utilization(self):
        # resource utilization is for 4 bits on average
        return ResourceUtilization(weights_memory=17920 * 4 / 8, activation_memory=math.ceil(self.max_cut*4/8))

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())
        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set only 8 and 4 bit candidates for test, to verify that all layers get exactly 4 bits
        mixed_precision_candidates_list = [(8, 8), (8, 4), (4, 8), (4, 4)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_4bit_test")

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is 4 bit average
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]

        # Note that since we're using default max aggregation for activation resource utilization,
        # then there is no guarantee that the activation bitwidth for each layer would be 4-bit,
        # this assertion tests the expected result for this specific
        # test with its current setup (therefore, we don't check the input layer's bitwidth)
        self.unit_test.assertTrue((activation_bits == [4, 4]))


class MixedPrecisionActivationSearch2BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2, 4])

    def get_resource_utilization(self):
        # resource utilization is for 2 bits on average
        return ResourceUtilization(weights_memory=17920.0 * 2 / 8, activation_memory=math.ceil(self.max_cut * 2 / 8))

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is minimal
        # Note that since we're using default max aggregation for activation resource utilization, then there is no guarantee that the
        # activation bitwidth for each layer would be 2-bit, this assertion tests the expected result for this specific
        # test with its current setup (therefore, we don't check the input layer's bitwidth)
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [2, 2]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=4)


class MixedPrecisionActivationDepthwiseTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3])

    def get_resource_utilization(self):
        # 638 = round_up((16*16*3+13*13*3)/2) -> so it must choose (4,4)
        return ResourceUtilization(47, 638)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 4]))


class MixedPrecisionActivationDepthwise4BitTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1])

    def get_resource_utilization(self):
        return ResourceUtilization(48.0 * 4 / 8, math.ceil((16*16*3+13*13*3) * 4 / 8))

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())
        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set only 8 and 4 bit candidates for test, to verify that all layers get exactly 4 bits
        mixed_precision_candidates_list = [(8, 8), (8, 4), (4, 8), (4, 4)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_depthwise_4bit_test")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is 4 bit average
        # Note that since we're using default max aggregation for activation resource utilization, then there is no guarantee that the
        # activation bitwidth for each layer would be 4-bit, this assertion tests the expected result for this specific
        # test with its current setup (therefore, we don't check the relu layer's bitwidth)
        holder_layer = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[0]
        self.unit_test.assertTrue(holder_layer.activation_holder_quantizer.get_config()['num_bits'] == 4)


class MixedPrecisionActivationSplitLayerTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3, 4])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.split(inputs, num_or_size_splits=2, axis=1)
        c0 = layers.Conv2D(32, 4)(x[0])
        c1 = layers.Conv2D(32, 4)(x[1])
        model = keras.Model(inputs=inputs, outputs=[c0, c1])
        return model

    def get_resource_utilization(self):
        return ResourceUtilization(3071, 2079)

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue(activation_bits in [[8, 4, 2], [8, 2, 4]])  # There are 2 options because the maxcut may choose either.

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[3, 4],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationOnlyTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 3, 4])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        outputs = layers.Conv2D(32, 4)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())
        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        mixed_precision_candidates_list = [(8, 8), (8, 4), (8, 2)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_weights_disabled_test")

    def get_resource_utilization(self):
        return ResourceUtilization(activation_memory=6507)

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 4, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.activation_memory + quantization_info.final_resource_utilization.weights_memory ==
            quantization_info.final_resource_utilization.total_memory,
            "Running activation mixed-precision with unconstrained weights and total resource utilization, "
            "final total memory should be equal to the sum of activation and weights memory.")


class MixedPrecisionActivationOnlyWeightsDisabledTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 3])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Conv2D(32, 4)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs(enable_kernel_weights_quantization=False))
        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})

        mixed_precision_candidates_list = [(8, 8), (8, 4), (8, 2)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_weights_disabled_test")

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 6407)

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 4, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationAddLayerTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 3])

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 5607)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.Add()([x, x])
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [h.activation_holder_quantizer.get_config()['num_bits'] for h in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 4, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2],
                                 weights_layers_channels_size=[32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionActivationMultipleInputsTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, num_calibration_iter=3, activation_layers_idx=[4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.num_of_inputs = 4
        self.val_batch_size = 2

    def get_resource_utilization(self):
        return ResourceUtilization(6143, 13.64e6)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, 3] for _ in range(self.num_of_inputs)]

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                           input_scaling=False, activation_channel_equalization=False)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=self.num_of_inputs)

    def create_networks(self):
        inputs_1 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_2 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_3 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_4 = layers.Input(shape=self.get_input_shapes()[0][1:])
        x1 = layers.Conv2D(32, 4)(inputs_1)
        x2 = layers.Conv2D(32, 4)(inputs_2)
        x3 = layers.Conv2D(32, 4)(inputs_3)
        x4 = layers.Conv2D(32, 4)(inputs_4)
        outputs = layers.Concatenate()([x1, x2, x3, x4])
        model = keras.Model(inputs=[inputs_1, inputs_2, inputs_3, inputs_4], outputs=outputs)
        return model

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        # resource utilization is infinity -> should give best model - 8bits
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [8, 8, 8, 8, 8, 8, 8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=256)


class MixedPrecisionTotalMemoryUtilizationSearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2, 4])

    def get_resource_utilization(self):
        # 17920: 8-bit weights, 6176: max cut of input+conv_bn
        return ResourceUtilization(np.inf, np.inf, total_memory=(17920 + self.max_cut) * 4 / 8)

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=16)


class MixedPrecisionMultipleResourcesTightUtilizationSearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2, 4])

    def get_resource_utilization(self):
        weights = 17920 * 4 / 8
        activation = math.ceil(self.max_cut * 4 / 8)
        return ResourceUtilization(weights, activation, total_memory=weights + activation)

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 4]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=16)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.total_memory ==
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionReducedTotalMemorySearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[2, 4])

    def get_resource_utilization(self):
        weights = 17920 * 4 / 8
        activation = math.ceil(self.max_cut * 4 / 8)
        return ResourceUtilization(weights, activation, total_memory=(weights + activation) / 2)

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1:]
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [2, 2]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 3],
                                 weights_layers_channels_size=[32, 32],
                                 activation_layers_idx=self.activation_layers_idx,
                                 unique_tensor_values=16)

        # Verify final ResourceUtilization
        self.unit_test.assertTrue(
            quantization_info.final_resource_utilization.total_memory ==
            quantization_info.final_resource_utilization.weights_memory + quantization_info.final_resource_utilization.activation_memory,
            "Running weights and activation mixed-precision, "
            "final total memory should be equal to sum of weights and activation memory.")


class MixedPrecisionDistanceSoftmaxTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 4])

    def get_resource_utilization(self):
        return ResourceUtilization(activation_memory=768)

    def get_core_config(self):
        return CoreConfig(quantization_config=QuantizationConfig(
            custom_tpc_opset_to_layer={"Softmax": CustomOpsetLayers([layers.Softmax, tf.nn.softmax, softmax,
                                                    LayerFilterParams(layers.Activation, activation=SOFTMAX)]),
                                       "Input": CustomOpsetLayers([layers.InputLayer])}))

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())

        # sets all combinations of 2, 4, 8 bits for weights and activations
        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_test",
                                                custom_opsets={'Softmax': mixed_precision_candidates_list})

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Softmax()(inputs)
        x = tf.nn.softmax(x)
        x = softmax(x)
        x = layers.Activation(SOFTMAX)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 4, 4, 4, 4]))


class MixedPrecisionDistanceSigmoidTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[1, 2, 4])

    def get_resource_utilization(self):
        return ResourceUtilization(activation_memory=768)

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())

        # sets all combinations of 2, 4, 8 bits for weights and activations
        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_activation_test",
                                                custom_opsets={schema.OperatorSetNames.SIGMOID:
                                                                   mixed_precision_candidates_list})

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = sigmoid(inputs)
        x = tf.nn.sigmoid(x)
        x = layers.Activation(SIGMOID)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # verify chosen activation bitwidth config
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in
                           holder_layers]
        self.unit_test.assertTrue((activation_bits == [4, 4, 4, 4]))


class MixedPrecisionActivationOnlyConfigurableWeightsTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, activation_layers_idx=[3, 4])

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.Add()([x, x])
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_core_config(self):
        return CoreConfig(quantization_config=QuantizationConfig(
            custom_tpc_opset_to_layer={"Weights": CustomOpsetLayers([layers.Conv2D],
                                                   {KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                                                    BIAS_ATTR: DefaultDict(default_value=BIAS)}),
                                       "Activations":CustomOpsetLayers([layers.ReLU, layers.Add])}))

    def get_tpc(self):
        cfg, mixed_precision_cfg_list, _ = get_op_quantization_configs()

        act_eight_bit_cfg = cfg.clone_and_edit(activation_n_bits=8,
                                               attr_weights_configs_mapping={})
        act_four_bit_cfg = cfg.clone_and_edit(activation_n_bits=4,
                                              attr_weights_configs_mapping={})
        act_two_bit_cfg = cfg.clone_and_edit(activation_n_bits=2,
                                             attr_weights_configs_mapping={})

        mixed_precision_cfg_list = \
            [c.clone_and_edit(enable_activation_quantization=False) for c in mixed_precision_cfg_list]
        cfg = mixed_precision_cfg_list[0]

        act_mixed_cfg = schema.QuantizationConfigOptions(quantization_configurations=tuple(
            [act_eight_bit_cfg, act_four_bit_cfg, act_two_bit_cfg]),
            base_config=act_eight_bit_cfg,
        )

        weight_mixed_cfg = schema.QuantizationConfigOptions(quantization_configurations=tuple(
            mixed_precision_cfg_list),
            base_config=cfg,
        )

        tpc = schema.TargetPlatformCapabilities(
            default_qco=schema.QuantizationConfigOptions(quantization_configurations=tuple([cfg]), base_config=cfg),
            tpc_minor_version=None,
            tpc_patch_version=None,
            tpc_platform_type=None,
            operator_set=tuple([schema.OperatorsSet(name="Activations", qc_options=act_mixed_cfg),
                          schema.OperatorsSet(name="Weights", qc_options=weight_mixed_cfg)]),
            add_metadata=False,
            name="mp_activation_conf_weights_test")

        return tpc

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 5410)

    def _compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)

        activation_bits = [layer.activation_holder_quantizer.get_config()['num_bits'] for layer in holder_layers]
        self.unit_test.assertTrue(activation_bits == [4, 4])
