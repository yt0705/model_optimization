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
from typing import List, Any

import torch

from mct_quantizers import PytorchActivationQuantizationHolder
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup, VAR
from model_compression_toolkit.trainable_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig
from model_compression_toolkit.trainable_infrastructure.pytorch.base_pytorch_quantizer import \
    BasePytorchTrainableQuantizer
from tests.pytorch_tests.trainable_infrastructure_tests.base_pytorch_trainable_infra_test import \
    BasePytorchInfrastructureTest, ZeroWeightsQuantizer, ZeroActivationsQuantizer
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.trainable_infrastructure.pytorch.activation_quantizers import (
    STESymmetricActivationTrainableQuantizer, STEUniformActivationTrainableQuantizer)


class TestPytorchBaseWeightsQuantizer(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_weights_quantization_config(self):
        return TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.UNIFORM,
                                               weights_n_bits=8,
                                               weights_quantization_params={},
                                               enable_weights_quantization=True,
                                               weights_channels_axis=0,
                                               weights_per_channel_threshold=True,
                                               min_threshold=0)

    def run_test(self):

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Quantization method mismatch. Expected methods: [<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 2>], received: QuantizationMethod.UNIFORM.', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_activation_quantization_config())
        self.unit_test.assertEqual(f'Expected weight quantization configuration; received activation quantization instead.', str(e.exception))

        weight_quantization_config = super(TestPytorchBaseWeightsQuantizer, self).get_weights_quantization_config()
        quantizer = ZeroWeightsQuantizer(weight_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == weight_quantization_config)
        # unless implemented explicitly, by default quant params should not be frozen
        self.unit_test.assertTrue(quantizer.freeze_quant_params is False)


class TestPytorchBaseActivationQuantizer(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_activation_quantization_config(self):
        return TrainableQuantizerActivationConfig(activation_quantization_method=QuantizationMethod.UNIFORM,
                                                  activation_n_bits=8,
                                                  activation_quantization_params={},
                                                  enable_activation_quantization=True,
                                                  min_threshold=0)

    def run_test(self):

        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(self.get_activation_quantization_config())
        self.unit_test.assertEqual(f'Quantization method mismatch. Expected methods: [<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 2>], received: QuantizationMethod.UNIFORM.', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Expected activation quantization configuration; received weight quantization instead.', str(e.exception))

        activation_quantization_config = super(TestPytorchBaseActivationQuantizer, self).get_activation_quantization_config()
        quantizer = ZeroActivationsQuantizer(activation_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == activation_quantization_config)
        # unless implemented explicitly, by default quant params should not be frozen
        self.unit_test.assertTrue(quantizer.freeze_quant_params is False)


class TestPytorchSTEActivationQuantizerQParamFreeze(BasePytorchInfrastructureTest):
    def run_test(self):
        sym_qparams = {'is_signed': True, 'threshold': [1]}
        self._run_test(STESymmetricActivationTrainableQuantizer, False, QuantizationMethod.POWER_OF_TWO, sym_qparams)
        self._run_test(STESymmetricActivationTrainableQuantizer, True, QuantizationMethod.SYMMETRIC, sym_qparams)

        uniform_qparams = {'range_min': 0, 'range_max': 5}
        self._run_test(STEUniformActivationTrainableQuantizer, False, QuantizationMethod.UNIFORM, uniform_qparams)
        self._run_test(STEUniformActivationTrainableQuantizer, True, QuantizationMethod.UNIFORM, uniform_qparams)

    def _run_test(self, activation_quantizer_cls, freeze, quant_method, activation_quant_params):
        quant_config = self.get_activation_quantization_config(quant_method=quant_method,
                                                               activation_quant_params=activation_quant_params)
        quantizer = activation_quantizer_cls(quant_config, freeze_quant_params=freeze)
        holder = PytorchActivationQuantizationHolder(quantizer)
        quantizer.initialize_quantization(torch.Size((5,)), 'foo', holder)
        self.unit_test.assertTrue(quantizer.freeze_quant_params is freeze)
        self.unit_test.assertTrue(quantizer.quantizer_parameters)
        for p in quantizer.quantizer_parameters.values():
            self.unit_test.assertTrue(p[VAR].requires_grad is not freeze)


class _TestQuantizer(BasePytorchTrainableQuantizer):

    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        super().__init__(quantization_config)

    def get_trainable_variables(self, group: VariableGroup) -> List[Any]:
        pass

    def initialize_quantization(self, tensor_shape, name: str, layer):
        pass

    def __call__(self, input2quantize, training: bool):
        pass


class TestPytorchQuantizerWithoutMarkDecorator(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):
        # create instance of dummy _TestQuantizer. Should throw exception because it is not marked with @mark_quantizer.
        with self.unit_test.assertRaises(Exception) as e:
            test_quantizer = _TestQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(
            "Quantizer class inheriting from 'BaseTrainableQuantizer' is improperly defined. "
            "Ensure it includes the '@mark_quantizer' decorator and is correctly applied.",
            str(e.exception))
