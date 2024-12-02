# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import model_compression_toolkit as mct
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, WEIGHTS_N_BITS, \
    WEIGHTS_QUANTIZATION_METHOD, IMX500_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformModel, Signedness, \
    AttributeQuantizationConfig, OpQuantizationConfig

tp = mct.target_platform


def get_tp_model() -> TargetPlatformModel:
    """
    A method that generates a default target platform model, with base 8-bit quantization configuration and 8, 4, 2
    bits configuration list for mixed-precision quantization.
    NOTE: in order to generate a target platform model with different configurations but with the same Operators Sets
    (for tests, experiments, etc.), use this method implementation as a test-case, i.e., override the
    'get_op_quantization_configs' method and use its output to call 'generate_tp_model' with your configurations.
    This version enables metadata by default.

    Returns: A TargetPlatformModel object.

    """
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
    return generate_tp_model(default_config=default_config,
                             base_config=base_config,
                             mixed_precision_cfg_list=mixed_precision_cfg_list,
                             name='imx500_lut_tp_model')


def get_op_quantization_configs() -> \
        Tuple[OpQuantizationConfig, List[OpQuantizationConfig], OpQuantizationConfig]:
    """
    Creates a default configuration object for 8-bit quantization, to be used to set a default TargetPlatformModel.
    In addition, creates a default configuration objects list (with 8, 4 and 2 bit quantization) to be used as
    default configuration for mixed-precision quantization with non-uniform quantizer for 2 and 4 bit candidates.

    Returns: An OpQuantizationConfig config object and a list of OpQuantizationConfig objects.

    """

    # We define a default quantization config for all non-specified weights attributes.
    default_weight_attr_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=8,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
    kernel_base_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.SYMMETRIC,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        lut_values_bitwidth=None)

    # We define a quantization config to quantize the bias (for layers where there is a bias attribute).
    bias_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=FLOAT_BITWIDTH,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    # Create a quantization config.
    # A quantization configuration defines how an operator
    # should be quantized on the modeled hardware:

    # We define a default config for operation without kernel attribute.
    # This is the default config that should be used for non-linear operations.
    eight_bits_default = schema.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={},
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO)

    # We define an 8-bit config for linear operations quantization, that include a kernel and bias attributes.
    linear_eight_bits = schema.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config, BIAS_ATTR: bias_config},
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO)

    # To quantize a model using mixed-precision, create
    # a list with more than one OpQuantizationConfig.
    # In this example, we quantize some operations' weights
    # using 2, 4 or 8 bits, and when using 2 or 4 bits, it's possible
    # to quantize the operations' activations using LUT.
    four_bits_lut = linear_eight_bits.clone_and_edit(
        attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4,
                                    WEIGHTS_QUANTIZATION_METHOD: tp.QuantizationMethod.LUT_SYM_QUANTIZER}},
        simd_size=linear_eight_bits.simd_size * 2)
    two_bits_lut = linear_eight_bits.clone_and_edit(
        attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 2,
                                    WEIGHTS_QUANTIZATION_METHOD: tp.QuantizationMethod.LUT_SYM_QUANTIZER}},
        simd_size=linear_eight_bits.simd_size * 4)
    mixed_precision_cfg_list = [linear_eight_bits, four_bits_lut, two_bits_lut]

    return linear_eight_bits, mixed_precision_cfg_list, eight_bits_default


def generate_tp_model(default_config: OpQuantizationConfig,
                      base_config: OpQuantizationConfig,
                      mixed_precision_cfg_list: List[OpQuantizationConfig],
                      name: str) -> TargetPlatformModel:
    """
    Generates TargetPlatformModel with default defined Operators Sets, based on the given base configuration and
    mixed-precision configurations options list.

    Args
        default_config: A default OpQuantizationConfig to set as the TP model default configuration.
        base_config: An OpQuantizationConfig to set as the TargetPlatformModel base configuration for mixed-precision purposes only.
        mixed_precision_cfg_list: A list of OpQuantizationConfig to be used as the TP model mixed-precision
            quantization configuration options.
        name: The name of the TargetPlatformModel.

    Returns: A TargetPlatformModel object.

    """
    # Create a QuantizationConfigOptions, which defines a set
    # of possible configurations to consider when quantizing a set of operations (in mixed-precision, for example).
    # If the QuantizationConfigOptions contains only one configuration,
    # this configuration will be used for the operation quantization:
    default_configuration_options = schema.QuantizationConfigOptions([default_config])

    # Create a QuantizationConfigOptions for quantizing constants in functional ops.
    # Constant configuration is similar to the default eight bit configuration except for PoT
    # quantization method for the constant.
    # Since the constants are not named attributes of the layer, we use the default_weight_attr_config to
    # define the desired quantization properties for them.
    const_config = default_config.clone_and_edit(
        default_weight_attr_config=default_config.default_weight_attr_config.clone_and_edit(
            enable_weights_quantization=True, weights_per_channel_threshold=True,
            weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO))
    const_configuration_options = schema.QuantizationConfigOptions([const_config])

    # Create a TargetPlatformModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise (see OperatorsSet, for example):
    generated_tpm = schema.TargetPlatformModel(
        default_configuration_options,
        tpc_minor_version=3,
        tpc_patch_version=0,
        tpc_platform_type=IMX500_TP_MODEL,
        add_metadata=True,
        name=name)

    # To start defining the model's components (such as operator sets, and fusing patterns),
    # use 'with' the TargetPlatformModel instance, and create them as below:
    with generated_tpm:
        # Create an OperatorsSet to represent a set of operations.
        # Each OperatorsSet has a unique label.
        # If a quantization configuration options is passed, these options will
        # be used for operations that will be attached to this set's label.
        # Otherwise, it will be a configure-less set (used in fusing):

        # May suit for operations like: Dropout, Reshape, etc.
        default_qco = tp.get_default_quantization_config_options()
        schema.OperatorsSet("NoQuantization",
                               default_qco.clone_and_edit(enable_activation_quantization=False)
                               .clone_and_edit_weight_attribute(enable_weights_quantization=False))

        # Create Mixed-Precision quantization configuration options from the given list of OpQuantizationConfig objects
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(mixed_precision_cfg_list,
                                                                                    base_config=base_config)

        # Define operator sets that use mixed_precision_configuration_options:
        conv = schema.OperatorsSet("Conv", mixed_precision_configuration_options)
        fc = schema.OperatorsSet("FullyConnected", mixed_precision_configuration_options)

        # Define operations sets without quantization configuration
        # options (useful for creating fusing patterns, for example):
        any_relu = schema.OperatorsSet("AnyReLU")
        add = schema.OperatorsSet("Add", const_configuration_options)
        sub = schema.OperatorsSet("Sub", const_configuration_options)
        mul = schema.OperatorsSet("Mul", const_configuration_options)
        div = schema.OperatorsSet("Div", const_configuration_options)
        prelu = schema.OperatorsSet("PReLU")
        swish = schema.OperatorsSet("Swish")
        sigmoid = schema.OperatorsSet("Sigmoid")
        tanh = schema.OperatorsSet("Tanh")

        # Combine multiple operators into a single operator to avoid quantization between
        # them. To do this we define fusing patterns using the OperatorsSets that were created.
        # To group multiple sets with regard to fusing, an OperatorSetConcat can be created
        activations_after_conv_to_fuse = schema.OperatorSetConcat(any_relu, swish, prelu, sigmoid, tanh)
        activations_after_fc_to_fuse = schema.OperatorSetConcat(any_relu, swish, sigmoid)
        any_binary = schema.OperatorSetConcat(add, sub, mul, div)

        # ------------------- #
        # Fusions
        # ------------------- #
        schema.Fusing([conv, activations_after_conv_to_fuse])
        schema.Fusing([fc, activations_after_fc_to_fuse])
        schema.Fusing([any_binary, any_relu])

    return generated_tpm
