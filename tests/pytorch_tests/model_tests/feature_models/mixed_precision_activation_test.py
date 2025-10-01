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
import torch
import numpy as np
from torch import softmax, sigmoid
from torch.nn import Softmax, Sigmoid

from model_compression_toolkit.core import MixedPrecisionQuantizationConfig, ResourceUtilization, CoreConfig, \
    QuantizationConfig
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder
from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tpc import generate_tpc_with_activation_mp
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit as mct
from tests.pytorch_tests.tpc_pytorch import get_mp_activation_pytorch_tpc_dict

"""
This test checks the Mixed Precision feature.
"""


class MixedPrecisionActivationBaseTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()
        return get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)]),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

    def get_core_configs(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False,
                                         custom_tpc_opset_to_layer={"Input": CustomOpsetLayers([DummyPlaceHolder])})
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

        return {"mixed_precision_activation_model": mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)}

    def create_feature_network(self, input_shape):
        raise NotImplementedError()

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a base test, so it does not check a thing. Only actual tests of mixed precision
        # compare things to test.
        raise NotImplementedError

    def verify_config(self, result_config, expected_config):
        self.unit_test.assertTrue(result_config == expected_config,
                                  f"Configuration mismatch: expected {expected_config} but got {result_config}.")


class MixedPrecisionActivationSearch4BitFunctional(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [1, 4, 5, 1]

    def get_resource_utilization(self):
        return ResourceUtilization(81, 3600)

    def create_feature_network(self, input_shape):
        return MixedPrecisionFunctionalNet(input_shape)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)


class MixedPrecisionActivationMultipleInputs(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [0, 0, 0, 0, 2, 1, 1, 1, 1]  # expected config for this test.
        self.num_calibration_iter = 3
        self.val_batch_size = 2

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 431)

    def get_core_configs(self):
        return {"mixed_precision_activation_model": CoreConfig(quantization_config=QuantizationConfig(
            custom_tpc_opset_to_layer={'Concat': CustomOpsetLayers([torch.concat]),
                                       "Input": CustomOpsetLayers([DummyPlaceHolder])}))}

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()
        return get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tpc_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)],
                custom_opsets=['Concat']),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

    def get_mixed_precision_config(self):
        return MixedPrecisionQuantizationConfig(num_of_images=4)

    def create_feature_network(self, input_shape):
        return MixedPrecisionMultipleInputsNet(input_shape)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 1, 8, 8],
                [self.val_batch_size, 1, 8, 8],
                [self.val_batch_size, 1, 8, 8],
                [self.val_batch_size, 1, 8, 8]]

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)


class MixedPrecisionFunctionalNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionFunctionalNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(in_channels, 3, kernel_size=(3, 3))

    def forward(self, inp):
        x1 = self.conv1(inp)
        x2 = self.conv2(inp)
        output = x1 + x2
        return output


class MixedPrecisionMultipleInputsNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionMultipleInputsNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(1, 3, kernel_size=(3, 3))
        self.conv3 = torch.nn.Conv2d(1, 3, kernel_size=(3, 3))
        self.conv4 = torch.nn.Conv2d(1, 3, kernel_size=(3, 3))

    def forward(self, x, y, z, w):
        x1 = self.conv1(x)
        x2 = self.conv2(y)
        x3 = self.conv3(z)
        x4 = self.conv4(w)
        return torch.concat([x1, x2, x3, x4], dim=1)


class MixedPrecisionDistanceFunctionsNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionDistanceFunctionsNet, self).__init__()
        self.softmax = Softmax(dim=-1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.softmax(x)
        x = self.sigmoid(x)
        x = softmax(x, dim=-1)
        x = sigmoid(x)

        return x


class MixedPrecisionDistanceFunctions(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [2, 1, 2, 1, 2]

    def get_resource_utilization(self):
        return ResourceUtilization(activation_memory=3071)

    def get_core_configs(self):
        return {"mixed_precision_activation_model": CoreConfig(quantization_config=QuantizationConfig(
            custom_tpc_opset_to_layer={'Softmax': CustomOpsetLayers([softmax, Softmax]),
                                       "Input": CustomOpsetLayers([DummyPlaceHolder])}))}

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()
        mp_list = [(8, 8), (8, 4), (8, 2),
                   (4, 8), (4, 4), (4, 2),
                   (2, 8), (2, 4), (2, 2)]

        tpc = generate_tpc_with_activation_mp(
            base_cfg=base_config,
            default_config=default_config,
            mp_bitwidth_candidates_list=mp_list,
            custom_opsets=['Softmax'])

        return get_mp_activation_pytorch_tpc_dict(tpc_model=tpc,
                                                  test_name='mixed_precision_activation_model',
                                                  tpc_name='mixed_precision_distance_fn_test')

    def create_feature_network(self, input_shape):
        return MixedPrecisionDistanceFunctionsNet(input_shape)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)
