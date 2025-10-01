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
from torch import nn
import numpy as np
from model_compression_toolkit.core.pytorch.utils import set_model, to_torch_tensor, \
    torch_tensor_to_numpy
from tests.common_tests.helpers.tensors_compare import normalized_mse
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks the BatchNorm folding feature, plus adding a residual connection.
"""
class BNFoldingNet(nn.Module):
    def __init__(self, test_layer, functional, fold_applied, has_weight=True):
        super(BNFoldingNet, self).__init__()
        self.conv1 = test_layer
        self.fold_applied = fold_applied
        self.bn = nn.BatchNorm2d(test_layer.out_channels, eps=1e-2)
        self.functional = functional
        self.has_weight = has_weight

    def forward(self, inp):
        x1 = self.conv1(inp)
        if self.functional:
            if self.has_weight:
                x = nn.functional.batch_norm(x1, self.bn.running_mean, self.bn.running_var, eps=self.bn.eps,
                                             bias=self.bn.bias, weight=self.bn.weight, training=self.bn.training,
                                             momentum=self.bn.momentum)
            else:
                x = nn.functional.batch_norm(x1, running_var=self.bn.running_var, running_mean=self.bn.running_mean,
                                             eps=self.bn.eps, bias=self.bn.bias, weight=self.bn.weight,
                                             training=self.bn.training, momentum=self.bn.momentum)
        else:
            x = self.bn(x1)
        x = torch.relu(x)
        if not self.fold_applied:
            x = x + x1
        return x


class BNFoldingNetTest(BasePytorchTest):
    """
    This test checks the BatchNorm folding feature, plus adding a residual connection.
    """
    def __init__(self, unit_test, test_layer, functional, fold_applied=True,
                 has_weight=True, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)
        self.input_channels = test_layer.in_channels
        self.test_layer = test_layer
        self.fold_applied = fold_applied
        self.functional = functional
        self.has_weight = has_weight

    def create_inputs_shape(self):
        return [[self.val_batch_size, self.input_channels, 32, 32]]

    def create_feature_network(self, input_shape):
        return BNFoldingNet(self.test_layer, self.functional, self.fold_applied, self.has_weight)

    def get_tpc(self):
        return {'no_quantization': super().get_tpc()['no_quantization']}

    def get_core_configs(self):
        return {'no_quantization': super().get_core_configs()['no_quantization']}

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        quant_model = quantized_models['no_quantization']
        set_model(quant_model)
        out_float = torch_tensor_to_numpy(float_model(*input_x))
        out_quant = torch_tensor_to_numpy(quant_model(*input_x))

        is_bn_in_model = nn.BatchNorm2d in [type(module) for name, module in quant_model.named_modules()]
        self.unit_test.assertTrue(self.fold_applied is not is_bn_in_model)
        self.unit_test.assertTrue(np.isclose(out_quant, out_float, atol=1e-5, rtol=1e-4).all())


class BNForwardFoldingNet(nn.Module):
    def __init__(self, test_layer, add_bn=False, is_dw=False):
        super(BNForwardFoldingNet, self).__init__()
        if is_dw:
            self.bn = nn.Conv2d(3, 3, 1, groups=3)
        else:
            self.bn = nn.BatchNorm2d(3)
            nn.init.uniform_(self.bn.weight, 0.02, 1.05)
            nn.init.uniform_(self.bn.bias, -1.2, 1.05)
            nn.init.uniform_(self.bn.running_var, 0.02, 1.05)
            nn.init.uniform_(self.bn.running_mean, -1.2, 1.05)
        self.conv = test_layer
        if add_bn:
            self.bn2 = nn.BatchNorm2d(test_layer.out_channels)
            nn.init.uniform_(self.bn2.weight, 0.02, 1.05)
            nn.init.uniform_(self.bn2.bias, -1.2, 1.05)
            nn.init.uniform_(self.bn2.running_var, 0.02, 1.05)
            nn.init.uniform_(self.bn2.running_mean, -1.2, 1.05)
        else:
            self.bn2 = None

    def forward(self, inp):
        x = self.bn(inp)
        x = self.conv(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = torch.tanh(x)
        return x


class BNForwardFoldingNetTest(BasePytorchTest):
    """
    This test checks the BatchNorm forward folding feature. When fold_applied is False
    test that the BN isn't folded
    """
    def __init__(self, unit_test, test_layer, fold_applied=True, add_bn=False, is_dw=False):
        super().__init__(unit_test, float_reconstruction_error=1e-6, val_batch_size=2)
        self.test_layer = test_layer
        self.bn_layer = nn.BatchNorm2d
        self.fold_applied = fold_applied
        self.add_bn = add_bn
        self.is_dw = is_dw

    def create_feature_network(self, input_shape):
        return BNForwardFoldingNet(self.test_layer, self.add_bn, self.is_dw)

    def get_tpc(self):
        return {'no_quantization': super().get_tpc()['no_quantization']}

    def get_core_configs(self):
        return {'no_quantization': super().get_core_configs()['no_quantization']}

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        quant_model = quantized_models['no_quantization']
        set_model(quant_model)

        if self.is_dw:
            is_bn_in_model = (sum([type(module) is nn.Conv2d for name, module in float_model.named_modules()]) ==
                              sum([type(module) is nn.Conv2d for name, module in quant_model.named_modules()]))
        else:
            is_bn_in_model = nn.BatchNorm2d in [type(module) for name, module in quant_model.named_modules()]

        self.unit_test.assertTrue(self.fold_applied is not is_bn_in_model)

        # Checking on multiple inputs to reduce probability for numeric error that will randomly fail the test
        self.unit_test.assertEqual(input_x[0].shape[0], 2, "Expecting batch of size 2 for BN folding test.")

        out_float = torch_tensor_to_numpy(float_model(*input_x))
        out_quant = torch_tensor_to_numpy(quant_model(*input_x))

        norm_mse, _, max_error, _ = normalized_mse(out_float, out_quant)

        self.unit_test.assertTrue(np.isclose(norm_mse[0], 0, atol=1e-5) or np.isclose(norm_mse[1], 0, atol=1e-5))
        self.unit_test.assertTrue(np.isclose(max_error[0], 0, atol=1e-4) or np.isclose(max_error[1], 0, atol=1e-4))
