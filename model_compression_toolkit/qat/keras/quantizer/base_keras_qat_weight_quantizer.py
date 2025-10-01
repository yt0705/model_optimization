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
from typing import Union

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.verify_packages import FOUND_TF

from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig, \
    TrainableQuantizerActivationConfig, BaseKerasTrainableQuantizer

if FOUND_TF:

    class BaseKerasQATWeightTrainableQuantizer(BaseKerasTrainableQuantizer):
        """
        A base class for trainable Keras quantizer for QAT.
        """
        pass

else:  # pragma: no cover
    class BaseKerasQATWeightTrainableQuantizer(BaseKerasTrainableQuantizer):
        def __init__(self,
                     quantization_config: Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]):

            super().__init__(quantization_config)
            Logger.critical("Tensorflow must be installed with a version of 2.15 or lower to use "
                            "BaseKerasQATTrainableQuantizer. The 'tensorflow' package is missing "
                            "or is installed with a version higher than 2.15.")  # pragma: no cover
