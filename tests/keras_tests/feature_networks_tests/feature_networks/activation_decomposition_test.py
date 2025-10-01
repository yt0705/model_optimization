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


import tensorflow as tf

from model_compression_toolkit.core.keras.constants import ACTIVATION, LINEAR, AXIS, SOFTMAX, SOFTMAX_AXIS_DEFAULT
from tests.keras_tests.tpc_keras import get_quantization_disabled_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class ActivationDecompositionTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, activation_function: str):
        self.activation_function = activation_function
        super().__init__(unit_test)

    def get_tpc(self):
        return get_quantization_disabled_keras_tpc("activation_decomp_test")

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = layers.Conv2D(3, 4, activation=self.activation_function)(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv_layer = get_layers_from_model_by_type(quantized_model, layers.Conv2D)[0]
        self.unit_test.assertTrue(conv_layer.get_config().get(ACTIVATION) == LINEAR)

        if self.activation_function==SOFTMAX:
            activation_layer = get_layers_from_model_by_type(quantized_model, keras.layers.Softmax)[0]
            self.unit_test.assertTrue(activation_layer.get_config().get(AXIS) == SOFTMAX_AXIS_DEFAULT)

        else:
            activation_layer = get_layers_from_model_by_type(quantized_model, layers.Activation)[0]
            self.unit_test.assertTrue(activation_layer.get_config().get(ACTIVATION) == self.activation_function)


