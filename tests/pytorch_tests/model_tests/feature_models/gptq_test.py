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

import numpy as np
import torch
import torch.nn as nn

import model_compression_toolkit as mct
from mct_quantizers import QuantizationMethod
from model_compression_toolkit import DefaultDict
from model_compression_toolkit.constants import GPTQ_HESSIAN_NUM_SAMPLES
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy, set_model
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig, RoundingType, \
    GPTQHessianScoresConfig, GradualActivationQuantizationConfig
from model_compression_toolkit.gptq.common.gptq_constants import QUANT_PARAM_LEARNING_STR, MAX_LSB_STR
from model_compression_toolkit.gptq.pytorch.gptq_loss import multiple_tensors_mse_loss
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.pytorch_tests.utils import extract_model_weights


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
        self.activation = nn.SiLU()
        self.linear = nn.Linear(14, 3)

    def forward(self, inp):
        x0 = self.conv1(inp)
        x1 = self.activation(x0)
        x2 = self.conv2(x1)
        y = self.activation(x2)
        y = self.linear(y)
        return y


class GPTQBaseTest(BasePytorchFeatureNetworkTest):
    def __init__(self, unit_test, weights_bits=8, weights_quant_method=QuantizationMethod.SYMMETRIC,
                 rounding_type=RoundingType.STE, per_channel=True,
                 hessian_weights=True, norm_scores=True, log_norm_weights=True, scaled_log_norm=False, params_learning=True,
                 num_calibration_iter=GPTQ_HESSIAN_NUM_SAMPLES, gradual_activation_quantization=False,
                 hessian_num_samples=GPTQ_HESSIAN_NUM_SAMPLES, sample_layer_attention=False,
                 loss=multiple_tensors_mse_loss, hessian_batch_size=1, reg_factor=1):
        super().__init__(unit_test, input_shape=(3, 16, 16), num_calibration_iter=num_calibration_iter)
        self.seed = 0
        self.rounding_type = rounding_type
        self.weights_bits = weights_bits
        self.weights_quant_method = weights_quant_method
        self.per_channel = per_channel
        if rounding_type == RoundingType.SoftQuantizer:
            self.override_params = {QUANT_PARAM_LEARNING_STR: params_learning}
        elif rounding_type == RoundingType.STE:
            self.override_params = {MAX_LSB_STR: DefaultDict(default_value=1)}
        else:
            raise ValueError('unknown rounding_type', rounding_type)
        self.gradual_activation_quantization = gradual_activation_quantization
        self.loss = loss
        self.reg_factor = reg_factor
        self.hessian_cfg = None
        if hessian_weights:
            self.hessian_cfg = GPTQHessianScoresConfig(per_sample=sample_layer_attention,
                                                       norm_scores=norm_scores,
                                                       log_norm=log_norm_weights,
                                                       scale_log_norm=scaled_log_norm,
                                                       hessians_num_samples=hessian_num_samples,
                                                       hessian_batch_size=hessian_batch_size)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.NOCLIPPING,
                                           mct.core.QuantizationErrorMethod.NOCLIPPING)

    def create_networks(self):
        return TestModel()

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="gptq_test",
            tpc=generate_test_tpc({'weights_n_bits': self.weights_bits,
                                             'weights_quantization_method': self.weights_quant_method}))

    def gptq_compare(self, ptq_model, gptq_model, input_x=None):
        pass

    def get_representative_data_gen_experimental_fixed_images(self):
        # data generator that generates same images in each epoch (in different order)
        dataset = []
        for _ in range(self.num_calibration_iter):
            dataset.append(self.generate_inputs())
        dataset = [np.concatenate(d) for d in zip(*dataset)]
        batch_size = int(np.ceil(dataset[0].shape[0] / self.num_calibration_iter))

        def gen():
            indices = np.random.permutation(range(dataset[0].shape[0]))
            shuffled_dataset = [d[indices] for d in dataset]
            for i in range(self.num_calibration_iter):
                yield [d[batch_size*i: batch_size*(i+1)] for d in shuffled_dataset]
        return gen

    def run_test(self):
        # Create model
        self.float_model = self.create_networks()
        set_model(self.float_model)

        # Run MCT with PTQ
        np.random.seed(self.seed)
        data_generator = self.get_representative_data_gen_experimental_fixed_images()
        ptq_model, _ = mct.ptq.pytorch_post_training_quantization(self.float_model,
                                                                  data_generator,
                                                                  core_config=self.get_core_config(),
                                                                  target_platform_capabilities=self.get_tpc())

        # Run MCT with GPTQ
        np.random.seed(self.seed)
        gptq_model, quantization_info = mct.gptq.pytorch_gradient_post_training_quantization(
            self.float_model,
            data_generator,
            core_config=self.get_core_config(),
            target_platform_capabilities=self.get_tpc(),
            gptq_config=self.get_gptq_config())

        # Generate inputs
        x = to_torch_tensor(self.representative_data_gen())

        # Compare
        self.gptq_compare(ptq_model, gptq_model, input_x=x)
        return gptq_model


class GPTQAccuracyTest(GPTQBaseTest):

    def get_gptq_config(self):
        gradual_act_cfg = GradualActivationQuantizationConfig() if self.gradual_activation_quantization else None
        return GradientPTQConfig(5, optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 loss=self.loss, train_bias=True, rounding_type=self.rounding_type,
                                 optimizer_bias=torch.optim.Adam([torch.Tensor([])], lr=0.4),
                                 hessian_weights_config=self.hessian_cfg,
                                 gptq_quantizer_params_override=self.override_params,
                                 gradual_activation_quantization_config=gradual_act_cfg,
                                 regularization_factor=self.reg_factor)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None):
        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')


class GPTQWeightsUpdateTest(GPTQBaseTest):

    def get_gptq_config(self):
        gradual_act_cfg = GradualActivationQuantizationConfig() if self.gradual_activation_quantization else None
        return GradientPTQConfig(50, optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 loss=multiple_tensors_mse_loss, train_bias=True, rounding_type=self.rounding_type,
                                 gradual_activation_quantization_config=gradual_act_cfg,
                                 gptq_quantizer_params_override=self.override_params,
                                 regularization_factor=self.reg_factor,
                                 hessian_weights_config=self.hessian_cfg)

    def compare(self, ptq_model, gptq_model, input_x=None, max_change=None):
        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))

        # check number of weights are equal
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')

        # check all weights were updated in gptq model compared to ptq model
        w_diff = [np.any(w_ptq != w_gptq) for w_ptq, w_gptq in zip(ptq_weights, gptq_weights)]
        self.unit_test.assertTrue(all(w_diff), msg="GPTQ: some weights weren't updated")


class GPTQLearnRateZeroTest(GPTQBaseTest):

    def get_gptq_config(self):
        gradual_act_cfg = GradualActivationQuantizationConfig() if self.gradual_activation_quantization else None
        return GradientPTQConfig(5, optimizer=torch.optim.Adam([torch.Tensor([])], lr=0),
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0),
                                 loss=multiple_tensors_mse_loss, train_bias=False, rounding_type=self.rounding_type,
                                 gradual_activation_quantization_config=gradual_act_cfg,
                                 gptq_quantizer_params_override=self.override_params,
                                 regularization_factor=self.reg_factor,
                                 hessian_weights_config=self.hessian_cfg)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None):
        ptq_out = torch_tensor_to_numpy(ptq_model(input_x))
        gptq_out = torch_tensor_to_numpy(gptq_model(input_x))
        float_output = torch_tensor_to_numpy(self.float_model(torch.Tensor(input_x[0])))
        self.unit_test.assertTrue(np.isclose(np.linalg.norm(ptq_out - float_output),
                                             np.linalg.norm(gptq_out - float_output), atol=1e-3))

        ptq_weights = extract_model_weights(ptq_model)
        ordered_weights = [ptq_weights[key] for key in sorted(ptq_weights.keys())]
        ptq_weights = torch_tensor_to_numpy(ordered_weights)

        gptq_weights = extract_model_weights(gptq_model)
        ordered_weights = [gptq_weights[key] for key in sorted(gptq_weights.keys())]
        gptq_weights = torch_tensor_to_numpy(ordered_weights)

        # check number of weights are equal
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')

        # check all weights were not updated in gptq model compared to ptq model
        w_diffs = [np.isclose(np.max(np.abs(w_ptq - w_gptq)), 0) for w_ptq, w_gptq in zip(ptq_weights, gptq_weights)]
        self.unit_test.assertTrue(np.all(w_diffs), msg="GPTQ: some weights were updated in zero learning rate test")
