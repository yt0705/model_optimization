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
from typing import List, Tuple

import tensorflow as tf
from packaging import version

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, KERAS_KERNEL, BIAS_ATTR, BIAS, \
    KERAS_DEPTHWISE_KERNEL, WEIGHTS_N_BITS
from tests.common_tests.helpers.generate_test_tpc import generate_test_op_qc, generate_test_attr_configs

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Reshape, ZeroPadding2D, Dropout, \
        MaxPooling2D, Activation, ReLU, Add, Subtract, Multiply, PReLU, Flatten, Cropping2D, LeakyReLU, Permute, \
        Conv2DTranspose

import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities, OpQuantizationConfig
from tests.common_tests.helpers.tpcs_for_tests.v1.tpc import generate_tpc



def get_tpc(edit_weights_params_dict, edit_act_params_dict) -> TargetPlatformCapabilities:
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()

    updated_config = base_config.clone_and_edit(attr_to_edit={KERNEL_ATTR: edit_weights_params_dict},
                                                **edit_act_params_dict)
    op_cfg_list = [updated_config]

    return generate_tpc(default_config=updated_config,
                        base_config=updated_config,
                        mixed_precision_cfg_list=op_cfg_list,
                        name='int8_tpc')


def get_op_quantization_configs() -> Tuple[OpQuantizationConfig, List[OpQuantizationConfig], OpQuantizationConfig]:
    eight_bits = generate_test_op_qc(**generate_test_attr_configs())
    four_bits = eight_bits.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4}},
                                          simd_size=eight_bits.simd_size * 2)
    two_bits = eight_bits.clone_and_edit({KERNEL_ATTR: {WEIGHTS_N_BITS: 2}},
                                         simd_size=eight_bits.simd_size * 4)
    mixed_precision_cfg_list = [eight_bits, four_bits, two_bits]
    default_config = eight_bits.clone_and_edit(attr_weights_configs_mapping={})
    return eight_bits, mixed_precision_cfg_list, default_config


def get_int8_tpc(edit_weights_params_dict={}, edit_act_params_dict={}) -> TargetPlatformCapabilities:
    default_tpc = get_tpc(edit_weights_params_dict, edit_act_params_dict)
    return default_tpc


def generate_keras_tpc(name: str, tpc: schema.TargetPlatformCapabilities):
    keras_tpc = FrameworkQuantizationCapabilities(tpc)

    with keras_tpc:
        OperationsSetToLayers("NoQuantization", [Reshape,
                                                    tf.reshape,
                                                    Permute,
                                                    tf.transpose,
                                                    Flatten,
                                                    Cropping2D,
                                                    ZeroPadding2D,
                                                    Dropout,
                                                    MaxPooling2D,
                                                    tf.split,
                                                    tf.quantization.fake_quant_with_min_max_vars,
                                                    tf.math.argmax,
                                                    tf.shape,
                                                    tf.math.equal,
                                                    tf.gather,
                                                    tf.cast,
                                                    tf.compat.v1.gather,
                                                    tf.nn.top_k,
                                                    tf.__operators__.getitem,
                                                    tf.compat.v1.shape])
        OperationsSetToLayers("Conv",
                                 [Conv2D,
                                  DepthwiseConv2D,
                                  Conv2DTranspose,
                                  tf.nn.conv2d,
                                  tf.nn.depthwise_conv2d,
                                  tf.nn.conv2d_transpose],
                                 attr_mapping={
                                     KERNEL_ATTR: DefaultDict({
                                         DepthwiseConv2D: KERAS_DEPTHWISE_KERNEL,
                                         tf.nn.depthwise_conv2d: KERAS_DEPTHWISE_KERNEL}, default_value=KERAS_KERNEL),
                                     BIAS_ATTR: DefaultDict(default_value=BIAS)})
        OperationsSetToLayers("FullyConnected", [Dense],
                                 attr_mapping={KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                                               BIAS_ATTR: DefaultDict(default_value=BIAS)})
        OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                             tf.nn.relu6,
                                             tf.nn.leaky_relu,
                                             ReLU,
                                             LeakyReLU,
                                             LayerFilterParams(Activation, activation="relu"),
                                             LayerFilterParams(Activation, activation="leaky_relu")])
        OperationsSetToLayers("Add", [tf.add, Add])
        OperationsSetToLayers("Sub", [tf.subtract, Subtract])
        OperationsSetToLayers("Mul", [tf.math.multiply, Multiply])
        OperationsSetToLayers("Div", [tf.math.divide])
        OperationsSetToLayers("PReLU", [PReLU])
        OperationsSetToLayers("Swish", [tf.nn.swish, LayerFilterParams(Activation, activation="swish")])
        OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid, LayerFilterParams(Activation, activation="sigmoid")])
        OperationsSetToLayers("Tanh", [tf.nn.tanh, LayerFilterParams(Activation, activation="tanh")])
    return keras_tpc
