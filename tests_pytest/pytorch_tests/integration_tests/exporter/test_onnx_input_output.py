# Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
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
import onnx
import pytest
import torch
import torch.nn as nn

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.pytorch.back2framework.float_model_builder import FloatPyTorchModel
from model_compression_toolkit.core.pytorch.utils import set_model
from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_onnx_pytorch_exporter import \
    FakelyQuantONNXPyTorchExporter
from model_compression_toolkit.exporter.model_exporter.pytorch.pytorch_export_facade import DEFAULT_ONNX_OPSET_VERSION
from model_compression_toolkit.exporter.model_wrapper import is_pytorch_layer_exportable
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import BaseTorchIntegrationTest


class SingleOutputModel(nn.Module):
    def __init__(self):
        super(SingleOutputModel, self).__init__()
        self.linear = nn.Linear(8, 5)

    def forward(self, x):
        return self.linear(x)


class MultipleOutputModel(nn.Module):
    def __init__(self):
        super(MultipleOutputModel, self).__init__()
        self.linear = nn.Linear(8, 5)

    def forward(self, x):
        return self.linear(x), x, x + 2


class MultipleInputsModel(nn.Module):
    def __init__(self):
        super(MultipleInputsModel, self).__init__()
        self.linear = nn.Linear(8, 5)

    def forward(self, input1, input2):
        return self.linear(input1) + self.linear(input2)


class TestONNXExporter(BaseTorchIntegrationTest):
    test_input_1 = None
    test_expected_1 = ['output']

    test_input_2 = ['output_2']
    test_expected_2 = ['output_2']

    test_input_3 = None
    test_expected_3 = ['output_0', 'output_1', 'output_2']

    test_input_4 = ['out', 'out_11', 'out_22']
    test_expected_4 = ['out', 'out_11', 'out_22']

    test_input_5 = ['out', 'out_11', 'out_22', 'out_33']
    test_expected_5 = ("Mismatch between number of requested output names (['out', 'out_11', 'out_22', 'out_33']) and "
                       "model output count (3):\n")

    def representative_data_gen(self, num_inputs=1):
        batch_size, num_iter, shape = 2, 1, (3, 8, 8)

        def data_gen():
            for _ in range(num_iter):
                yield [torch.randn(batch_size, *shape)] * num_inputs

        return data_gen

    def get_pytorch_model(self, model, data_generator, minimal_tpc):
        qc = QuantizationConfig()
        graph = self.run_graph_preparation(model=model, datagen=data_generator, tpc=minimal_tpc,
                                           quant_config=qc)
        pytorch_model = FloatPyTorchModel(graph=graph)
        return pytorch_model

    def export_model(self, model, save_model_path, data_generator, output_names=None):
        exporter = FakelyQuantONNXPyTorchExporter(model,
                                                  is_pytorch_layer_exportable,
                                                  save_model_path,
                                                  data_generator,
                                                  onnx_opset_version=DEFAULT_ONNX_OPSET_VERSION)

        exporter.export(output_names)

        assert save_model_path.exists(), "ONNX file was not created"
        assert save_model_path.stat().st_size > 0, "ONNX file is empty"

        onnx_model = onnx.load(str(save_model_path))
        return onnx_model

    def validate_outputs(self, onnx_model, expected_output_names):
        outputs = onnx_model.graph.output

        # Check number of outputs
        assert len(outputs) == len(
            expected_output_names), f"Expected {len(expected_output_names)} output, but found {len(outputs)}"

        found_output_names = [output.name for output in outputs]
        assert found_output_names == expected_output_names, (
            f"Expected output name '{expected_output_names}' found {found_output_names}"
        )

    @pytest.mark.parametrize(
        ("model", "output_names", "expected_output_names"), [
            (SingleOutputModel(), test_input_1, test_expected_1),
            (SingleOutputModel(), test_input_2, test_expected_2),
            (MultipleOutputModel(), test_input_3, test_expected_3),
            (MultipleOutputModel(), test_input_4, test_expected_4),
        ])
    def test_output_model_name(self, tmp_path, model, output_names, expected_output_names, minimal_tpc):
        save_model_path = tmp_path / "model.onnx"
        data_generator = self.representative_data_gen(num_inputs=1)
        pytorch_model = self.get_pytorch_model(model, data_generator, minimal_tpc)
        onnx_model = self.export_model(pytorch_model, save_model_path, data_generator, output_names=output_names)
        self.validate_outputs(onnx_model, expected_output_names)

    @pytest.mark.parametrize(
        ("model", "output_names", "expected_output_names"), [
            (MultipleOutputModel(), test_input_5, test_expected_5),
        ])
    def test_wrong_number_output_model_name(self, tmp_path, model, output_names, expected_output_names, minimal_tpc):
        save_model_path = tmp_path / "model.onnx"
        data_generator = self.representative_data_gen(num_inputs=1)
        pytorch_model = self.get_pytorch_model(model, data_generator, minimal_tpc)
        try:
            onnx_model = self.export_model(pytorch_model, save_model_path, data_generator, output_names=output_names)
            self.validate_outputs(onnx_model, expected_output_names)
        except Exception as e:
            assert expected_output_names == str(e)

    def test_multiple_inputs(self, minimal_tpc, tmp_path):
        """
        Test that model with multiple inputs is exported to onnx file properly and that the exported onnx model
        has all expected inputs.
        """
        save_model_path = tmp_path / "model.onnx"
        model = MultipleInputsModel()
        data_generator = self.representative_data_gen(num_inputs=2)
        pytorch_model = self.get_pytorch_model(model, data_generator, minimal_tpc)
        onnx_model = self.export_model(pytorch_model, save_model_path, data_generator)
        assert [_input.name for _input in onnx_model.graph.input] == ["input_0", "input_1"]
