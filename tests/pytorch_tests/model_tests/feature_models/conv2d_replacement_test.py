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
from mct_quantizers import PytorchQuantizationWrapper

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.network_editors.node_filters import NodeTypeFilter
from model_compression_toolkit.core.common.network_editors.actions import EditRule, ReplaceLayer
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest



def get_new_weights_for_identity_dw_conv2d_layer(weights={}, activation_quantization_params={}, **kwargs):
    """
    return the weights of depthwise conv2d layers set to ones
    """

    new_weights = weights
    old_kernel_shape = weights['weight'].shape
    new_kernel = np.ones(old_kernel_shape)
    new_weights['weight'] = new_kernel
    return new_weights, kwargs


class OneLayerConv2dNet(torch.nn.Module):
    def __init__(self):
        super(OneLayerConv2dNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 3, 1, groups=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return x


class DwConv2dReplacementTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_debug_config(self):
        return mct.core.DebugConfig(network_editor=[EditRule(filter=NodeTypeFilter(torch.nn.Conv2d),
                                                        action=ReplaceLayer(torch.nn.Conv2d, get_new_weights_for_identity_dw_conv2d_layer))])

    def create_feature_network(self, input_shape):
        return OneLayerConv2dNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        quantized_model = quantized_models.get('no_quantization')
        self.unit_test.assertTrue(isinstance(quantized_model.conv1, torch.nn.Conv2d))
        self.unit_test.assertTrue(torch.all(quantized_model.conv1.weight == 1))
        self.unit_test.assertTrue(torch.all(torch.eq(quantized_model(input_x), input_x[0])))
