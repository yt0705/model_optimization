# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import model_compression_toolkit as mct
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common.network_editors.node_filters import NodeNameFilter
from model_compression_toolkit.core.common.network_editors.actions import EditRule, \
    ChangeCandidatesWeightsQuantizationMethod
from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy, set_model
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest



def get_uniform_weights(out_channels, in_channels, kernel):
    return np.array([i - np.round((in_channels * kernel * kernel * out_channels) / 2) for i in
                     range(in_channels * kernel * kernel * out_channels)]).reshape(
        [out_channels, in_channels, kernel, kernel]).transpose(1, 2, 3, 0)


class LUTQuantizerNet(torch.nn.Module):
    def __init__(self, kernel=3, num_conv_channels=3):
        super(LUTQuantizerNet, self).__init__()
        self.conv_w = get_uniform_weights(num_conv_channels, num_conv_channels, kernel)
        self.conv1 = torch.nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=kernel, stride=1)
        self.conv2 = torch.nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=kernel, stride=1)
        self.conv3 = torch.nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=kernel, stride=1)
        self.conv1.weight = torch.nn.Parameter(to_torch_tensor(self.conv_w), requires_grad=False)
        self.conv2.weight = torch.nn.Parameter(to_torch_tensor(self.conv_w), requires_grad=False)
        self.conv3.weight = torch.nn.Parameter(to_torch_tensor(self.conv_w), requires_grad=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LUTActivationQuantizerNet(torch.nn.Module):
    def __init__(self, kernel=3, num_conv_channels=3):
        super(LUTActivationQuantizerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=kernel, stride=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(num_conv_channels, num_conv_channels, kernel_size=kernel, stride=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu2(x)
        return x


class LUTWeightsQuantizerTest(BasePytorchTest):
    """
    This test checks multiple features:
    1. That the LUT quantizer quantizes the weights differently than than the Power-of-two quantizer
    2. That the Network Editor works

    In this test we set the weights of 3 conv2d operator to be the same. We set the quantization method
    to "Power-of-two". With the Network Editor we change the quantization method of "conv1" to "LUT quantizer".
    We check that the weights have different values for conv1 and conv2, and that conv2 and conv3 have the same
    values.
    """
    def __init__(self, unit_test, weights_n_bits=4, quant_method=QuantizationMethod.LUT_POT_QUANTIZER):
        super().__init__(unit_test)
        self.weights_n_bits = weights_n_bits
        self.quant_method = quant_method
        self.node_to_change_name = 'conv1'
        self.num_conv_channels = 3
        self.kernel = 3

    def get_tpc(self):
        return get_pytorch_test_tpc_dict(
            tpc=generate_test_tpc({"weights_n_bits": self.weights_n_bits}),
            test_name='lut_quantizer_test',
            ftp_name='lut_quantizer_pytorch_test')

    def get_core_configs(self):
        network_editor = [EditRule(filter=NodeNameFilter(self.node_to_change_name),
                                   action=ChangeCandidatesWeightsQuantizationMethod(
                                       attr_name=KERNEL,
                                       weights_quantization_method=self.quant_method))]
        return {'lut_quantizer_test': mct.core.CoreConfig(quantization_config=mct.core.QuantizationConfig(
            mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE),
                                                          debug_config=mct.core.DebugConfig(network_editor=network_editor))}

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 16, 16], [self.val_batch_size, 3, 16, 16]]

    def create_feature_network(self, input_shape):
        return LUTQuantizerNet(self.kernel, self.num_conv_channels)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        quantized_model = quantized_models.get('lut_quantizer_test')
        q_weight1 = quantized_model.conv1.get_quantized_weights()['weight']
        q_weight2 = quantized_model.conv2.get_quantized_weights()['weight']
        self.unit_test.assertTrue(np.sum(np.abs(torch_tensor_to_numpy(q_weight1) -
                                                torch_tensor_to_numpy(q_weight2))) > 0)
        q_weight3 = quantized_model.conv3.get_quantized_weights()['weight']
        self.unit_test.assertFalse(np.sum(np.abs(torch_tensor_to_numpy(q_weight3) -
                                                 torch_tensor_to_numpy(q_weight3))) > 0)


class LUTActivationQuantizerTest(BasePytorchTest):
    """
    This test checks that activation are quantized correctly using LUT quantizer

    """
    def __init__(self, unit_test, activation_n_bits=4):
        super().__init__(unit_test)
        self.activation_n_bits = activation_n_bits
        self.kernel = 3
        self.num_conv_channels = 3

    def get_tpc(self):
        return get_pytorch_test_tpc_dict(
            tpc=generate_test_tpc({"activation_n_bits": self.activation_n_bits,
                                             "activation_quantization_method": QuantizationMethod.LUT_POT_QUANTIZER}),
            test_name='lut_quantizer_test',
            ftp_name='lut_quantizer_pytorch_test')

    def get_core_configs(self):
        return {'lut_quantizer_test': mct.core.CoreConfig()}

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 16, 16], [self.val_batch_size, 3, 16, 16]]

    def create_feature_network(self, input_shape):
        return LUTActivationQuantizerNet(self.kernel, self.num_conv_channels)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        quantized_model = quantized_models.get('lut_quantizer_test')

        # Check that quantization occurred and the number of quantization lut values
        set_model(float_model)
        self.unit_test.assertFalse(np.all(torch_tensor_to_numpy(float_model(input_x[0]) == quantized_model(input_x[0]))))
        self.unit_test.assertTrue(len(np.unique(torch_tensor_to_numpy(quantized_model(input_x[0])).flatten())) <=
                                  2 ** self.activation_n_bits)
