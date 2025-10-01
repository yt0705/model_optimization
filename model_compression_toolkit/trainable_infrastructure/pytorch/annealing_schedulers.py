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
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.trainable_infrastructure.common.annealing_schedulers import BaseLinearAnnealingScheduler


class PytorchLinearAnnealingScheduler(BaseLinearAnnealingScheduler):
    def _compute_factor(self, t: int) -> float:
        """
        Computes the annealing factor for torch models.

        Args:
            t: Current time step.

        Returns:
            float: Clipped annealing factor between 0 and 1.
        """
        factor = to_torch_tensor((t - self.t_start) / (self.t_end - self.t_start))
        return factor.clip(0, 1)
