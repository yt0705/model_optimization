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

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.pytorch.back2framework.float_model_builder import FloatPyTorchModelBuilder
from model_compression_toolkit.core.pytorch.back2framework.mixed_precision_model_builder import \
    MixedPrecisionPyTorchModelBuilder
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
from model_compression_toolkit.core.pytorch.back2framework.quantized_model_builder import QuantizedPyTorchModelBuilder

pytorch_model_builders = {ModelBuilderMode.QUANTIZED: QuantizedPyTorchModelBuilder,
                          ModelBuilderMode.FLOAT: FloatPyTorchModelBuilder,
                          ModelBuilderMode.MIXEDPRECISION: MixedPrecisionPyTorchModelBuilder}


def get_pytorch_model_builder(mode: ModelBuilderMode) -> type:
    """
    Return a PyTorch model builder given a ModelBuilderMode.

    Args:
        mode: Mode of the PyTorch model builder.

    Returns:
        PyTorch model builder for the given mode.
    """

    if not isinstance(mode, ModelBuilderMode):  # pragma: no cover
        Logger.critical(f"Expected a ModelBuilderMode type for 'mode' parameter; received {type(mode)} instead.")
    if mode is None:  # pragma: no cover
        Logger.critical(f"Received 'mode' parameter is None.")
    if mode not in pytorch_model_builders.keys():  # pragma: no cover
        Logger.critical(f"'mode' parameter {mode} is not supported by the PyTorch model builders factory.")
    return pytorch_model_builders.get(mode)
