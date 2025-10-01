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

from model_compression_toolkit.core import ResourceUtilization, MixedPrecisionQuantizationConfig, CoreConfig, \
    QuantizationConfig
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, BatchNormalization, ReLU, Input, Add

from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs
import tensorflow as tf

from tests.keras_tests.tpc_keras import get_tpc_with_activation_mp_keras

keras = tf.keras
layers = keras.layers


def get_base_mp_nbits_candidates():
    return [(4, 8), (4, 4), (4, 2),
            (8, 8), (8, 4), (8, 2),
            (2, 8), (2, 4), (2, 2)]


class BaseMixedPrecisionBopsTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, mixed_precision_candidates_list):
        super().__init__(unit_test)

        self.mixed_precision_candidates_list = mixed_precision_candidates_list

    def get_core_config(self):
        return CoreConfig(quantization_config=QuantizationConfig(
            custom_tpc_opset_to_layer={"Input": CustomOpsetLayers([layers.InputLayer])}))

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()

        return get_tpc_with_activation_mp_keras(base_config=base_config,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=self.mixed_precision_candidates_list,
                                                name="mp_bopts_test")

    def get_mixed_precision_config(self):
        return MixedPrecisionQuantizationConfig(num_of_images=1)

    def get_input_shapes(self):
        return [[self.val_batch_size, 16, 16, 3]]

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Verify that some layers got bit-width smaller than 8 bits (so checking candidate index is not 0)
        self.unit_test.assertTrue(any(i > 0 for i in quantization_info.mixed_precision_cfg))
        # Verify final BOPs utilization
        self.unit_test.assertTrue(quantization_info.final_resource_utilization.bops <= self.get_resource_utilization().bops)


class MixedPrecisionBopsBasicTest(BaseMixedPrecisionBopsTest):
    def __init__(self, unit_test):

        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        super().__init__(unit_test, mixed_precision_candidates_list)

    def create_networks(self):
        inputs = Input(shape=self.get_input_shapes()[0][1:])
        x = Conv2D(3, 4)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        outputs = Conv2D(3, 4)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_resource_utilization(self):
        return ResourceUtilization(bops=1000000)  # should require some quantization to all layers


class MixedPrecisionBopsAllWeightsLayersTest(BaseMixedPrecisionBopsTest):
    def __init__(self, unit_test, mixed_precision_candidates_list=None):

        if mixed_precision_candidates_list is None:
            mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        super().__init__(unit_test, mixed_precision_candidates_list)

    def create_networks(self):
        inputs = Input(shape=self.get_input_shapes()[0][1:])
        x = Conv2D(3, 4)(inputs)
        x = Conv2D(3, 4)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(3, 4)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = DepthwiseConv2D(3, depth_multiplier=5)(x)
        outputs = Dense(5)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_resource_utilization(self):
        return ResourceUtilization(bops=1252512)  # should require some quantization to all layers


class MixedPrecisionWeightsOnlyBopsTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, mixed_precision_candidates_list=[(8, 8), (4, 8), (2, 8)])

    def get_resource_utilization(self):
        return ResourceUtilization(bops=5010100)  # should require some quantization to all layers


class MixedPrecisionActivationOnlyBopsTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, mixed_precision_candidates_list=[(8, 8), (8, 4), (8, 2)])

    def get_resource_utilization(self):
        return ResourceUtilization(bops=5010100)  # should require some quantization to all layers

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Verify that some layers got bit-width smaller than 8 bits (so checking candidate index is not 0)
        self.unit_test.assertTrue(any(i > 0 for i in quantization_info.mixed_precision_cfg))
        # Verify final BOPs utilization
        self.unit_test.assertTrue(quantization_info.final_resource_utilization.bops <= self.get_resource_utilization().bops)


class MixedPrecisionBopsAndWeightsUtilizationTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(weights_memory=170, bops=1300000)  # should require some quantization to all layers


class MixedPrecisionBopsAndActivationUtilizationTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(activation_memory=1000, bops=1300000)  # should require some quantization to all layers


class MixedPrecisionBopsAndTotalUtilizationTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(total_memory=1000, bops=1300000)  # should require some quantization to all layers


class MixedPrecisionBopsWeightsActivationUtilizationTest(MixedPrecisionBopsAllWeightsLayersTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(weights_memory=200, activation_memory=1000, bops=1300000)  # should require some quantization to all layers


class MixedPrecisionBopsMultipleOutEdgesTest(BaseMixedPrecisionBopsTest):
    def __init__(self, unit_test):

        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        super().__init__(unit_test, mixed_precision_candidates_list)

    def create_networks(self):
        inputs = Input(shape=self.get_input_shapes()[0][1:])
        x = Conv2D(3, 4)(inputs)
        y = Conv2D(3, 4)(inputs)
        outputs = Add()([x, y])
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_resource_utilization(self):
        return ResourceUtilization(bops=(16*3*3*13*13)*8*8*2)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # Verify that all layers got 8 bits (so checking candidate index is 0)
        self.unit_test.assertTrue(all(i == 0 for i in quantization_info.mixed_precision_cfg))
