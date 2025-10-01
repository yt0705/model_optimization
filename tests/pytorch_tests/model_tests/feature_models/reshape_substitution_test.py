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
import torch
import torch.nn as nn
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, set_model
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc


class BaseReshapeSubstitutionTest(BasePytorchFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test)

    def get_tpc(self):
        tp = generate_test_tpc({'weights_n_bits': 32,
                                     'activation_n_bits': 32,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_pytorch_tpc(name="permute_substitution_test", tpc=tp)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.NOCLIPPING,
                                           mct.core.QuantizationErrorMethod.NOCLIPPING, False, False)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        in_torch_tensor = to_torch_tensor(input_x[0])
        set_model(float_model)
        y = float_model(in_torch_tensor)
        y_hat = quantized_model(in_torch_tensor)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')

class ReshapeSubstitutionTest(BaseReshapeSubstitutionTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class ReshapeNet(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.gamma = nn.Parameter(1 * torch.ones((1, 3, 1, 1)))  
        def forward(self, x):
            x=x.mul(self.gamma.reshape(1,-1,1,1))
            return x
    def create_networks(self):
        return self.ReshapeNet()
