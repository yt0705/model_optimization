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

import unittest

import numpy as np

import model_compression_toolkit as mct
from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.exporter.model_exporter.pytorch.pytorch_export_facade import DEFAULT_ONNX_OPSET_VERSION

from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc


class BasePytorchExportTest(unittest.TestCase):

    def get_model(self):
        raise NotImplemented

    def get_input_shapes(self):
        return [(1, 3, 8, 8)]

    def get_dataset(self):
        yield [to_torch_tensor(np.random.rand(*shape)).to(get_working_device()) for shape in self.get_input_shapes()]

    def get_tpc(self):
        return mct.get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)

    def get_serialization_format(self):
        raise NotImplemented

    def get_tmp_filepath(self):
        raise NotImplemented

    def load_exported_model(self, filepath):
        raise NotImplemented

    def get_core_config(self):
        return mct.core.CoreConfig()

    def compare(self, loaded_model, quantized_model, quantization_info):
        raise NotImplemented

    def run_test(self):
        quantized_model, quantization_info = self.run_mct()
        self.run_export(quantized_model)
        loaded_model = self.load_exported_model(self.filepath)
        self.compare(loaded_model, quantized_model, quantization_info)

    def infer(self, model, images):
        raise NotImplemented


    def get_quantization_format(self):
        return mct.exporter.QuantizationFormat.MCTQ

    def run_mct(self):
        model = self.get_model()
        return mct.ptq.pytorch_post_training_quantization(model,
                                                          self.get_dataset,
                                                          core_config=self.get_core_config(),
                                                          target_platform_capabilities=self.get_tpc())

    def run_export(self, quantized_model, onnx_opset_version=DEFAULT_ONNX_OPSET_VERSION):
        self.filepath = self.get_tmp_filepath()
        mct.exporter.pytorch_export_model(quantized_model,
                                          self.filepath,
                                          self.get_dataset,
                                          serialization_format=self.get_serialization_format(),
                                          quantization_format=self.get_quantization_format(),
                                          onnx_opset_version=onnx_opset_version)
