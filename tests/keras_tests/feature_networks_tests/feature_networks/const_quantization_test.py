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
from functools import partial
import tensorflow as tf
import numpy as np

import model_compression_toolkit as mct
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.core import MixedPrecisionQuantizationConfig

from tests.common_tests.helpers.generate_test_tpc import generate_custom_test_tpc
from tests.common_tests.helpers.tpcs_for_tests.v3.tpc import get_tpc as get_tp_v3
from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import get_tpc as get_tp_v4
from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import generate_tpc, get_op_quantization_configs
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from tests.keras_tests.utils import get_layers_from_model_by_type
from mct_quantizers import KerasQuantizationWrapper, QuantizationMethod

keras = tf.keras
layers = keras.layers


def create_const_quant_tpc(qmethod):
    name = "const_quant_tpc"
    base_cfg, mp_op_cfg_list, default_cfg = get_op_quantization_configs()
    base_tpc = generate_tpc(default_config=default_cfg,
                                 base_config=base_cfg,
                                 mixed_precision_cfg_list=mp_op_cfg_list,
                                 name=name)

    const_config = default_cfg.clone_and_edit(
        default_weight_attr_config=default_cfg.default_weight_attr_config.clone_and_edit(
            enable_weights_quantization=True, weights_per_channel_threshold=True,
            weights_n_bits=16, weights_quantization_method=qmethod))
    const_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([const_config]))
    const_merge_config = default_cfg.clone_and_edit(
        default_weight_attr_config=default_cfg.default_weight_attr_config.clone_and_edit(
            weights_per_channel_threshold=False))
    const_merge_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([const_merge_config]))

    operator_sets_dict = {}
    operator_sets_dict[schema.OperatorSetNames.ADD] = const_configuration_options
    operator_sets_dict[schema.OperatorSetNames.SUB] = const_configuration_options
    operator_sets_dict[schema.OperatorSetNames.MUL] = const_configuration_options
    operator_sets_dict[schema.OperatorSetNames.DIV] = const_configuration_options
    operator_sets_dict[schema.OperatorSetNames.STACK] = const_merge_configuration_options
    operator_sets_dict[schema.OperatorSetNames.CONCATENATE] = const_merge_configuration_options

    tpc = generate_custom_test_tpc(name=name,
                                        base_cfg=base_cfg,
                                        base_tpc=base_tpc,
                                        operator_sets_dict=operator_sets_dict)

    return tpc


class ConstQuantizationTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, layer, const, is_list_input=False, input_reverse_order=False, use_kwargs=False,
                 error_method: mct.core.QuantizationErrorMethod = mct.core.QuantizationErrorMethod.MSE,
                 qmethod: QuantizationMethod = QuantizationMethod.POWER_OF_TWO,
                 input_shape=(32, 32, 16)):
        super(ConstQuantizationTest, self).__init__(unit_test=unit_test, input_shape=input_shape)
        self.layer = layer
        self.const = const
        self.is_list_input = is_list_input
        self.input_reverse_order = input_reverse_order
        self.use_kwargs = use_kwargs
        self.error_method = error_method
        self.qmethod = qmethod

    def generate_inputs(self):
        # need positive inputs so won't divide with zero or take root of negative number
        return [1 + np.random.random(in_shape) for in_shape in self.get_input_shapes()]

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(weights_error_method=self.error_method)

    def get_tpc(self):
        return create_const_quant_tpc(self.qmethod)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = inputs
        if self.is_list_input:
            if self.input_reverse_order:
                x = self.layer([self.const, x])
            else:
                x = self.layer([x, self.const])
        else:
            if self.input_reverse_order:
                if self.use_kwargs:
                    x = self.layer(x=self.const, y=x)
                else:
                    x = self.layer(self.const, x)
            else:
                if self.use_kwargs:
                    x = self.layer(x=x, y=self.const)
                else:
                    x = self.layer(x, self.const)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1, atol=0.001), msg=f'fail cosine similarity check:{cs}')
        self.unit_test.assertTrue(isinstance(quantized_model.layers[2], KerasQuantizationWrapper),
                                  msg='TFOpLambda should be quantized')
        const_index = 0 if self.input_reverse_order else 1
        self.unit_test.assertTrue((quantized_model.layers[2].weight_values[const_index] == self.const).all(),
                                  msg='Constant value should not change')


class AdvancedConstQuantizationTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, input_shape=(32, 32, 3)):
        super(AdvancedConstQuantizationTest, self).__init__(unit_test=unit_test, input_shape=input_shape,
                                                            num_calibration_iter=32)
        self.const = np.random.random((130,))

    def get_ptq_facade(self):
        gptq_config = mct.gptq.get_keras_gptq_config(30)
        return partial(mct.gptq.keras_gradient_post_training_quantization,
                       gptq_config=gptq_config)

    def get_resource_utilization(self):
        return mct.core.ResourceUtilization(9e3)

    def get_mixed_precision_config(self):
        return MixedPrecisionQuantizationConfig()

    def generate_inputs(self):
        # need positive inputs so won't divide with zero or take root of negative number
        return [1 + np.random.random(in_shape) for in_shape in self.get_input_shapes()]

    def get_tpc(self):
        return get_tp_v3()

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(130, 3)(inputs)
        x = layers.ReLU()(x)
        x = tf.add(x, self.const)
        x = layers.Conv2D(16, 3)(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        self.unit_test.assertTrue(isinstance(quantized_model.layers[5], KerasQuantizationWrapper),
                                  msg='TFOpLambda should be quantized')
        self.unit_test.assertTrue((quantized_model.layers[5].weight_values[1] == self.const).all(),
                                  msg='Constant value should not change')


class ConstQuantizationMultiInputTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, input_shape=(32, 32, 16)):
        super(ConstQuantizationMultiInputTest, self).__init__(unit_test=unit_test, input_shape=input_shape)

    def get_tpc(self):
        return get_tp_v4()

    def create_networks(self):
        as_const = lambda v: np.random.random(v.shape.as_list()).astype(np.float32)
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Concatenate()([inputs, np.random.random((1, 32, 32, 3)),
                                  inputs, np.random.random((1, 32, 32, 3))])
        x1 = layers.Add()([np.random.random((1, x.shape[-1])), x, np.random.random((1, x.shape[-1]))])
        x2 = layers.Multiply()([x, np.random.random((1, x.shape[-1])), x, np.random.random((1, x.shape[-1]))])
        x3 = tf.add_n([x1, as_const(x), x2])
        x1 = tf.reshape(tf.stack([as_const(x1), x1, as_const(x1)], axis=1),
                        (-1, 3 * x1.shape[1], x1.shape[2], x1.shape[3]))
        x = tf.concat([x1, x2, as_const(x3), x3], 1)
        ind_select_const = np.zeros((192 * 32, 38))
        ind_select_const[4, :] = 100
        x1 = tf.add(x, ind_select_const.reshape((192, 32, 38)))
        inds = tf.argmax(tf.reshape(x1, (-1, 192 * 32, 38)), axis=1)
        b = tf.reshape(tf.gather(np.random.random((192 * 32 * 38,)).astype(np.float32), inds), (-1, 1, 1, 38))
        x = tf.add(x, b)

        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(y, y_hat)
        # atol is rather high because there's a potential large difference between the float and quantized indices tensor
        # that goes to the tf.argmax.
        self.unit_test.assertTrue(np.isclose(cs, 1, atol=0.01), msg=f'fail cosine similarity check:{cs}')

        # check quantization layers:
        for op in [tf.concat, tf.stack, layers.Add, layers.Multiply, layers.Concatenate, tf.gather,
                   tf.compat.v1.gather]:
            for qlayer in get_layers_from_model_by_type(quantized_model, op):
                self.unit_test.assertTrue(isinstance(qlayer, KerasQuantizationWrapper),
                                          msg=f"{op} should be quantized.")
