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
from enum import Enum


class PytorchExportSerializationFormat(Enum):
    """
    Specify which serialization format to use for exporting a quantized Pytorch model.

    TORCHSCRIPT - torchscript format

    ONNX - onnx format

    """

    TORCHSCRIPT = 0
    ONNX = 1
