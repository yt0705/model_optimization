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

from abc import ABC
from packaging import version
import model_compression_toolkit as mct
import tensorflow as tf
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda

from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class BaseConv2DCollapsingTest(BaseKerasFeatureNetworkTest, ABC):

    def __init__(self, unit_test):
        super(BaseConv2DCollapsingTest, self).__init__(unit_test=unit_test, input_shape=(32,32,16))

    def get_tpc(self):
        tp = generate_test_tpc({'weights_n_bits': 32,
                                     'activation_n_bits': 32,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="linear_collapsing_test", tpc=tp)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.NOCLIPPING,
                                           mct.core.QuantizationErrorMethod.NOCLIPPING, False, False)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        # FIXME this doesn't test anything, the number of quantized convs in the network is exactly 0. Even if it
        #  looked at correct layers it hardly checks anything.
        self.unit_test.assertTrue(len([l for l in quantized_model.layers if isinstance(l, KerasTrainableQuantizationWrapper) and isinstance(l.layer, layers.Conv2D)]) < len([l for l in float_model.layers if isinstance(l, layers.Conv2D)]), msg=f'fail number of layers should decrease!')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class TwoConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', bias_initializer='glorot_uniform')(inputs)
        y = layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_model, float_model, input_x, quantization_info)
        for layer in quantized_model.layers:
            if type(layer) == layers.Conv2D:
                self.unit_test.assertTrue(len(layer.weights) == 2, msg=f'fail Bias should appear in weights!!')


class ThreeConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        y = layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, activation='relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_model, float_model, input_x, quantization_info)
        for layer in quantized_model.layers:
            if type(layer) == layers.Conv2D:
                self.unit_test.assertTrue(len(layer.weights) == 1,msg=f'fail Bias should not appear in weights!!')


class FourConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', bias_initializer='glorot_uniform', activation='linear')(inputs)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform', activation='linear')(x)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', bias_initializer='glorot_uniform', activation='linear')(x)
        y = layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform', activation='relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_model, float_model, input_x, quantization_info)
        convs = [l for l in quantized_model.layers if isinstance(l, layers.Conv2D)]
        self.unit_test.assertTrue(len(convs) == 1)
        for layer in convs:
            self.unit_test.assertTrue(len(layer.weights) == 2,msg=f'fail Bias should appear in weights!!')


class FuncConvCollapsingTest(FourConv2DCollapsingTest):
    def create_networks(self):
        # Tests the combination of functional conv to Conv2D substitution with linear collapsing
        # (in case of default values, tf layer doesn't contain these attributes, and they must be added explicitly
        # to node's attributes dict, which is not covered by substitution test)
        h, w, c = self.get_input_shapes()[0][1:]
        inputs = layers.Input(shape=(h, w, c))
        x = tf.nn.conv2d(inputs, tf.random.uniform((3, 3, c, 16)), 1, 'SAME')
        x = tf.nn.convolution(x, tf.random.uniform((1, 1, 16, 8)))
        x = tf.nn.relu(x)
        x = tf.nn.convolution(x, tf.random.uniform((3, 3, 8, 32)))
        y = tf.nn.conv2d(x, tf.random.uniform((1, 1, 32, 4)), 1, 'VALID')
        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        convs = [l for l in quantized_model.layers if isinstance(l, layers.Conv2D)]
        self.unit_test.assertTrue(len(convs) == 2)

        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(y, y_hat)
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class SixConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform')(x)
        x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform',  activation='relu')(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform')(x)
        y = layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', bias_initializer='glorot_uniform', activation='relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_model, float_model, input_x, quantization_info)
        for layer in quantized_model.layers:
            if type(layer) == layers.Conv2D:
                self.unit_test.assertTrue(len(layer.weights) == 2,msg=f'fail Bias should appear in weights!!')


class Op2DAddConstCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        # ########
        # Cond2D #
        # ########
        # Collapse Conv2D with bias
        x = layers.Conv2D(filters=7, kernel_size=(5, 5), strides=(1, 1), padding='same',
                          use_bias=True, bias_initializer='glorot_uniform')(inputs)
        x = tf.math.add(x, tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype))
        x = layers.ReLU()(x)

        # Collapse Conv2D without bias, const first argument of tf.math.add
        x = layers.Conv2D(filters=5, kernel_size=(5, 5), strides=(1, 1), padding='same',
                          use_bias=False)(x)
        x = tf.math.add(tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype), x)
        x = layers.ReLU()(x)

        # Collapse + operator to Conv2D without bias
        # TODO: replace add with + (currently using tf.math.add because below TF 2.14 creates TFOpLambda which fails ths node matcher)
        x = layers.Conv2D(filters=9, kernel_size=(5, 5), strides=(1, 1), padding='same',
                          use_bias=False)(x)
        x = tf.math.add(x, tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype))

        # #################
        # DepthwiseConv2D #
        # #################
        # Collapse DepthwiseConv2D with bias
        x = layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same',
                                   use_bias=True, bias_initializer='glorot_uniform')(x)
        x = tf.math.add(x, tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype))
        x = layers.ReLU()(x)

        # Collapse DepthwiseConv2D without bias, const first argument of tf.math.add
        x = layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same',
                                   use_bias=False)(x)
        x = tf.math.add(tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype), x)
        x = layers.ReLU()(x)

        # Collapse + operator to DepthwiseConv2D without bias
        # TODO: replace add with + (currently using tf.math.add because below TF 2.14 creates TFOpLambda which fails ths node matcher)
        x = layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same',
                                   use_bias=False)(x)
        x = tf.math.add(x, tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype))

        # #################
        # Conv2DTranspose #
        # #################
        # Collapse Conv2DTranspose with bias
        x = layers.Conv2DTranspose(filters=9, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                   use_bias=True, bias_initializer='glorot_uniform')(x)
        x = tf.math.add(x, tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype))
        x = layers.ReLU()(x)

        # Collapse Conv2DTranspose without bias, const first argument of tf.math.add
        x = layers.Conv2DTranspose(filters=9, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                   use_bias=False)(x)
        x = tf.math.add(tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype), x)
        x = layers.ReLU()(x)

        # Collapse + operator to Conv2DTranspose without bias
        # TODO: replace add with + (currently using tf.math.add because below TF 2.14 creates TFOpLambda which fails ths node matcher)
        x = layers.Conv2DTranspose(filters=9, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                   use_bias=False)(x)
        x = tf.math.add(x, tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype))

        # #######
        # Dense #
        # #######
        x = layers.Reshape((-1,))(x)
        # Collapse Dense with bias
        x = layers.Dense(9, use_bias=True, bias_initializer='glorot_uniform')(x)
        x = tf.math.add(x, tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype))
        x = layers.ReLU()(x)

        # Collapse Dense without bias, const first argument of tf.math.add
        x = layers.Dense(9, use_bias=False)(x)
        x = tf.math.add(tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype), x)
        x = layers.ReLU()(x)

        # Collapse + operator to Conv2DTranspose without bias
        # TODO: replace add with + (currently using tf.math.add because below TF 2.14 creates TFOpLambda which fails ths node matcher)
        x = layers.Dense(9, use_bias=False)(x)
        x = tf.math.add(x, tf.constant(np.random.normal(size=x.shape[-1]), dtype=x.dtype))

        # Don't collapse
        x2 = layers.Dense(9, use_bias=True, bias_initializer='glorot_uniform')(x)
        x = tf.math.add(x2, x)
        y = layers.ReLU()(x)

        return tf.keras.models.Model(inputs=inputs, outputs=y)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_model, float_model, input_x, quantization_info)
        num_adds = 0
        for layer in quantized_model.layers:
            if type(layer) in [layers.Conv2D, layers.DepthwiseConv2D, layers.Conv2DTranspose, layers.Dense]:
                self.unit_test.assertTrue(len(layer.weights) == 2, msg=f'fail Bias should appear in weights!!')
            elif isinstance(layer, TFOpLambda) and (layer.function is tf.add or layer.symbol == TFOpLambda(tf.add).symbol):
                num_adds += 1

        # check all "add"s were folded except the one with 2 tensor inputs
        self.unit_test.assertTrue(num_adds == 1, msg=f'Only one add should remain in the quantized model')