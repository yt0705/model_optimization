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

from model_compression_toolkit.gptq.common.gptq_config import (
    GradientPTQConfig,
    RoundingType,
    GPTQHessianScoresConfig,
    GradualActivationQuantizationConfig,
    QFractionLinearAnnealingConfig
)

from model_compression_toolkit.verify_packages import FOUND_TF, FOUND_TORCH

if FOUND_TF:
    from model_compression_toolkit.gptq.keras.quantization_facade import keras_gradient_post_training_quantization
    from model_compression_toolkit.gptq.keras.quantization_facade import get_keras_gptq_config

if FOUND_TORCH:
    from model_compression_toolkit.gptq.pytorch.quantization_facade import pytorch_gradient_post_training_quantization
    from model_compression_toolkit.gptq.pytorch.quantization_facade import get_pytorch_gptq_config