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
import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
from mct_quantizers import KerasActivationQuantizationHolder, KerasQuantizationWrapper
from model_compression_toolkit.core.common.network_editors import NodeNameFilter, NodeTypeFilter
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames, \
    QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import \
    get_config_options_by_operators_set
from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from tests.common_tests.helpers.generate_test_tpc import generate_test_op_qc, generate_test_attr_configs, \
    generate_custom_test_tpc
from tests.common_tests.helpers.tpcs_for_tests.v3.tpc import get_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.tpc_keras import get_tpc_with_activation_mp_keras

keras = tf.keras
layers = keras.layers

get_op_set = lambda x, x_list: [op_set for op_set in x_list if op_set.name == x][0]


class ManualBitWidthSelectionTest(BaseKerasFeatureNetworkTest):
    """
    This test check the manual bit width configuration.
    Call it with a layer type filter or list of layer type filters, bit width or list of bit widths.
    Uses the manual bit width API in the "get_core_configs" method.
    """

    def __init__(self, unit_test, filters, bit_widths, **kwargs):
        self.filters = filters
        self.bit_widths = bit_widths
        self.layer_types = {}
        self.layer_names = {}
        self.functional_names = {}

        filters = [filters] if not isinstance(filters, list) else filters
        bit_widths = [bit_widths] if not isinstance(bit_widths, list) else bit_widths
        if len(bit_widths) < len(filters):
            bit_widths = [bit_widths[0] for f in filters]
        for filter, bit_width in zip(filters, bit_widths):
            if isinstance(filter, NodeNameFilter):
                self.layer_names.update({filter.node_name: bit_width})
            elif isinstance(filter, NodeTypeFilter):
                self.layer_types.update({filter.node_type: bit_width})
        super().__init__(unit_test, **kwargs)

    def create_networks(self):
        input_tensor = layers.Input(shape=self.get_input_shapes()[0][1:], name='input')
        x1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', name='conv1')(input_tensor)
        x1 = layers.Add(name='add1')([x1, np.ones((3,), dtype=np.float32)])

        # Second convolutional block
        x2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', name='conv2')(x1)
        x2 = layers.BatchNormalization(name='bn1')(x2)
        x2 = layers.ReLU(name='relu1')(x2)

        # Addition
        x = layers.Add(name='add2')([x1, x2])

        # Flatten and fully connected layer
        x = layers.Flatten()(x)
        output_tensor = layers.Dense(units=10, activation='softmax', name='fc')(x)

        return keras.Model(inputs=input_tensor, outputs=output_tensor)

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())
        default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
        # set only 8 and 4 bit candidates for test, to verify that all layers get exactly 4 bits
        mixed_precision_candidates_list = [(8, 8), (8, 4), (8, 2), (4, 8), (4, 4), (4, 2), (2, 8), (2, 4), (2, 2)]

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="mixed_precision_4bit_test")

    def get_mp_core_config(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False,
                                         custom_tpc_opset_to_layer={'Input': CustomOpsetLayers([layers.InputLayer])})
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

        core_config = mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)
        return core_config

    def get_core_config(self):
        # Configures the core settings including manual bit width adjustments.
        core_config = self.get_mp_core_config()
        core_config.bit_width_config.set_manual_activation_bit_width(self.filters, self.bit_widths)
        return core_config

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # in the compare we need bit_widths to be a list
        bit_widths = [self.bit_widths] if not isinstance(self.bit_widths, list) else self.bit_widths

        for layer in quantized_model.layers:
            # check if the layer is an activation quantizer
            if isinstance(layer, KerasActivationQuantizationHolder):
                # get the layer that's activation is being quantized
                layer_q = quantized_model.layers[quantized_model.layers.index(layer) - 1]
                if isinstance(layer_q, KerasQuantizationWrapper):
                    layer_q = layer_q.layer
                # check if this layer is in the layer types to change bit width and check that the correct bit width was applied.
                layer_q_bit_width = self.layer_names.get(layer_q.name) if self.layer_names.get(
                    layer_q.name) is not None else self.layer_types.get(type(layer_q))
                if layer_q_bit_width is not None:
                    self.unit_test.assertTrue(layer.activation_holder_quantizer.num_bits == layer_q_bit_width)
                else:
                    # make sure that the bit width of other layers was not changed.
                    self.unit_test.assertFalse(layer.activation_holder_quantizer.num_bits in bit_widths,
                                               msg=f"name {layer_q.name}, layer.activation_holder_quantizer.num_bits {layer.activation_holder_quantizer.num_bits}, {self.bit_widths}")


class Manual16BitWidthSelectionTest(ManualBitWidthSelectionTest):
    """
    This test check the manual bit width configuration for 16 bits.
    The network is built such that one multiply can be configured to 16 bit (mul1) and one cannot (mul2).
    Call it with a layer type filter or list of layer type filters, bit width or list of bit widths.
    Uses the manual bit width API in the "get_core_configs" method.
    """
    def get_tpc(self):
        tpc = get_tpc()
        base_cfg_16 = [c for c in get_config_options_by_operators_set(tpc,
                                                                      OperatorSetNames.MUL).quantization_configurations
                       if c.activation_n_bits == 16][0].clone_and_edit()
        qco_16 = QuantizationConfigOptions(base_config=base_cfg_16,
                                           quantization_configurations=(tpc.default_qco.base_config,
                                                                        base_cfg_16))
        tpc = generate_custom_test_tpc(
            name="custom_16_bit_tpc",
            base_cfg=tpc.default_qco.base_config,
            base_tpc=tpc,
            operator_sets_dict={
                OperatorSetNames.MUL: qco_16,
                OperatorSetNames.GELU: qco_16,
                OperatorSetNames.TANH: qco_16,
            })

        return tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:], name='input')
        x = layers.Multiply(name='mul1')([inputs, inputs])[:, :8, :8, :]
        x1 = layers.Add(name='add1')([x, x])
        x2 = layers.Subtract(name='sub1')([x1, x])
        x = layers.Multiply(name='mul2')([x, x2])
        x = layers.Conv2D(3, 1, name='conv1')(x)
        outputs = tf.divide(x, 2 * np.ones((3,), dtype=np.float32))
        return keras.Model(inputs=inputs, outputs=outputs)


class Manual16BitWidthSelectionMixedPrecisionTest(Manual16BitWidthSelectionTest):
    """
    This test check the manual bit width configuration for 16 bits with mixed precision.
    The network is built such that one multiply can be configured to 16 bit (mul1) and one cannot (mul2).
    Call it with a layer type filter or list of layer type filters, bit width or list of bit widths.
    Uses the manual bit width API in the "get_core_configs" method.
    """
    def get_tpc(self):
        tpc = get_tpc()

        mul_qco = get_config_options_by_operators_set(tpc, OperatorSetNames.MUL)
        base_cfg_16 = [l for l in mul_qco.quantization_configurations if l.activation_n_bits == 16][0]
        quantization_configurations = list(mul_qco.quantization_configurations)
        quantization_configurations.extend([
            base_cfg_16.clone_and_edit(activation_n_bits=4),
            base_cfg_16.clone_and_edit(activation_n_bits=2)])

        qco_16 = QuantizationConfigOptions(base_config=base_cfg_16,
                                           quantization_configurations=quantization_configurations)

        add_qco = get_config_options_by_operators_set(tpc, OperatorSetNames.ADD)
        base_cfg_8 = [l for l in add_qco.quantization_configurations if l.activation_n_bits == 8][0]
        add_qco_8 = QuantizationConfigOptions(base_config=base_cfg_8, quantization_configurations=[base_cfg_8])

        sub_qco = get_config_options_by_operators_set(tpc, OperatorSetNames.SUB)
        base_cfg_8 = [l for l in sub_qco.quantization_configurations if l.activation_n_bits == 8][0]
        sub_qco_8 = QuantizationConfigOptions(base_config=base_cfg_8, quantization_configurations=[base_cfg_8])

        tpc = generate_custom_test_tpc(
            name="custom_16_bit_tpc",
            base_cfg=tpc.default_qco.base_config,
            base_tpc=tpc,
            operator_sets_dict={
                OperatorSetNames.MUL: qco_16,
                OperatorSetNames.ADD: add_qco_8,
                OperatorSetNames.SUB: sub_qco_8,
            })

        return tpc

    def get_resource_utilization(self):
        return mct.core.ResourceUtilization(activation_memory=6000)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantization_info.mixed_precision_cfg) > 0, "Expected mixed-precision in test.")
        super().compare(quantized_model, float_model, input_x=input_x, quantization_info=quantization_info)
