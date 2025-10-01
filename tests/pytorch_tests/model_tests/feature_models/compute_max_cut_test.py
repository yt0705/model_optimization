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

import torch.nn as nn
import model_compression_toolkit as mct
from model_compression_toolkit.core.common.fusion.fusing_info import FUSED_OP_ID_PREFIX
from tests.common_tests.helpers.tpcs_for_tests.v2.tpc import get_tpc
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from model_compression_toolkit.constants import PYTORCH
from mct_quantizers.pytorch.metadata import get_metadata



class MaxCutModel(nn.Module):
    def __init__(self):
        super(MaxCutModel, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(4)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(4)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        y = self.relu_1(self.batch_norm_1(self.conv2d_1(x)))
        x = self.relu_2(self.batch_norm_2(self.conv2d_2(y)))
        return x + y


class ComputeMaxCutTest(BasePytorchFeatureNetworkTest):

    def get_tpc(self):
        return get_tpc()

    def create_networks(self):
        return MaxCutModel()

    def get_debug_config(self):
        return mct.core.DebugConfig(simulate_scheduler=True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        _metadata = get_metadata(quantized_model)
        self.unit_test.assertEqual(_metadata['scheduling_info']['operators_scheduling'],
                                   ['DummyPlaceHolder:x',
                                    f'FusedLayerType:{FUSED_OP_ID_PREFIX}conv2d_1_bn_relu_1',
                                    f'FusedLayerType:{FUSED_OP_ID_PREFIX}conv2d_2_bn_relu_2',
                                    'add:add'])
        self.unit_test.assertEqual(_metadata['scheduling_info']['max_cut'], 256 * 3)

        expected_fused_nodes_mapping = {
            'conv2d_1_bn': f"{FUSED_OP_ID_PREFIX}conv2d_1_bn_relu_1",
            'relu_1': f"{FUSED_OP_ID_PREFIX}conv2d_1_bn_relu_1",
            'conv2d_2_bn': f"{FUSED_OP_ID_PREFIX}conv2d_2_bn_relu_2",
            'relu_2': f"{FUSED_OP_ID_PREFIX}conv2d_2_bn_relu_2"
        }
        self.unit_test.assertEqual(_metadata['scheduling_info']['fused_nodes_mapping'], expected_fused_nodes_mapping)
