# Copyright 2021 Sony Semiconductor Solutions, Inc. All rights reserved.
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

from model_compression_toolkit.core.common.collectors.base_collector import BaseCollector
from model_compression_toolkit.constants import LAST_AXIS


class MeanCollector(BaseCollector):
    """
        Class to collect observed per channel mean values of tensors that goes through it (passed to update).
        The mean is calculated using a exponential moving average with bias correction.
    """

    def __init__(self,
                 axis: int):
        """
        Instantiate a per channel mean collector using a exponential moving average with bias correction.

        Args:
            axis: Compute the mean with regard to this axis.
        """
        super().__init__()
        self.axis = axis
        self.current_mean = 0
        self.current_sum = 0
        self.i = 0

    def scale(self, scale_factor: np.ndarray):
        """
        Scale all statistics in collector by some factor.
        Since mean was collected per-channel, it can be scaled either by a single factor or a factor
        per-channel.
        The scaling is done using the corrected mean.

        Args:
            scale_factor: Factor to scale all collector's statistics by.

        """

        self.current_mean *= scale_factor

    def shift(self, shift_value: np.ndarray):
        """
        Shift all statistics in collector by some value.
        Since mean was collected per-channel, it can be shifted either by a single value or a
        shifting value per-channel.
        The shifting is done using the corrected mean.

        Args:
            shift_value: Value to shift all collector's statistics by.

        """

        self.current_mean += shift_value

    @property
    def state(self):
        """
        The mean is kept internal and corrected when accessed from outside the collector.

        Returns: Mean of the collector after bias correction.
        """
        self.validate_data_correctness()
        return self.current_mean

    def update(self,
               x: np.ndarray):
        """
        Update the mean using a new tensor x to consider.

        Args:
            x: Tensor that goes through the mean collector and needs to be considered in the mean computation.
        """
        self.i += 1  # Update the iteration index
        axis = (len(x.shape) - 1) if self.axis == LAST_AXIS else self.axis
        n = x.shape[axis]
        transpose_index = [axis, *[i for i in range(len(x.shape)) if i != axis]]
        mu = np.mean(np.reshape(np.transpose(x, transpose_index), [n, -1]), axis=-1) # mean per channel for a batch
        self.current_sum += mu # sum of all batches
        self.current_mean = self.current_sum / self.i # mean of all batches


