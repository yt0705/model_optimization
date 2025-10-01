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

import math
import unittest
import numpy as np

from model_compression_toolkit.core import QuantizationErrorMethod
from model_compression_toolkit.constants import LUT_VALUES, MIN_THRESHOLD, THRESHOLD
from model_compression_toolkit.core.common.quantization.quantization_params_generation.lut_kmeans_params import \
    lut_kmeans_histogram


class TestLUTActivationsQuantizerParams(unittest.TestCase):

    def test_signed_lut_activation_quantization_params(self):
        data = np.random.randn(3, 4, 5, 6)
        counts, bins = np.histogram(data, bins=20)
        n_bits = 4

        quantization_params = lut_kmeans_histogram(bins=bins,
                                                   counts=counts,
                                                   p=2,  # dummy
                                                   n_bits=n_bits,
                                                   min_value=1,  # dummy
                                                   max_value=1,  # dummy
                                                   constrained=True,  # dummy
                                                   n_iter=20,  # dummy
                                                   min_threshold=MIN_THRESHOLD,
                                                   quant_error_method=QuantizationErrorMethod.MSE  # dummy
                                                   )

        lut_values = quantization_params[LUT_VALUES]
        threshold = quantization_params[THRESHOLD]
        # check threshold is power-of-two
        self.assertTrue(math.log2(threshold).is_integer(), "LUT quantization threshold must be a power of two")

        # check number of lut values
        self.assertTrue(lut_values.shape[0] <= 2 ** n_bits,
                        f"Number of lut values is {lut_values.shape[0]} but should not exceed {2 ** n_bits}"),
        # check lut values are rounded
        self.assertTrue(np.all(np.mod(lut_values, 1) == 0), "lut values are supposed to be rounded")

    def test_unsigned_lut_activation_quantization_params(self):
        data = np.random.randn(3, 4, 5, 6)
        data[data < 0] = data[data < 0] * -1
        counts, bins = np.histogram(data, bins=20)
        n_bits = 4

        quantization_params = lut_kmeans_histogram(bins=bins,
                                                   counts=counts,
                                                   p=2,  # dummy
                                                   n_bits=n_bits,
                                                   min_value=1,  # dummy
                                                   max_value=1,  # dummy
                                                   constrained=True,  # dummy
                                                   n_iter=20,  # dummy
                                                   min_threshold=MIN_THRESHOLD,
                                                   quant_error_method=QuantizationErrorMethod.MSE  # dummy
                                                   )

        lut_values = quantization_params[LUT_VALUES]
        threshold = quantization_params[THRESHOLD]
        # check threshold is power-of-two
        self.assertTrue(math.log2(threshold).is_integer(), "LUT quantization threshold must be a power of two")

        # check number of lut values
        self.assertTrue(lut_values.shape[0] <= 2 ** n_bits,
                        f"Number of lut values is {lut_values.shape[0]} but should not exceed {2 ** n_bits}"),
        # check lut values are rounded
        self.assertTrue(np.all(np.mod(lut_values, 1) == 0), "lut values are supposed to be rounded")

    def test_lut_activation_quantization_params_with_fewer_data(self):
        # check when the number of values of the data is lower than 2**n_bits
        data = np.random.randn(3, 4, 5)
        counts, bins = np.histogram(data, bins=20)
        n_bits = 7

        quantization_params = lut_kmeans_histogram(bins=bins,
                                                   counts=counts,
                                                   p=2,  # dummy
                                                   n_bits=n_bits,
                                                   min_value=1,  # dummy
                                                   max_value=1,  # dummy
                                                   constrained=True,  # dummy
                                                   n_iter=20,  # dummy
                                                   min_threshold=MIN_THRESHOLD,
                                                   quant_error_method=QuantizationErrorMethod.MSE  # dummy
                                                   )

        lut_values = quantization_params[LUT_VALUES]
        threshold = quantization_params[THRESHOLD]
        # check threshold is power-of-two
        self.assertTrue(math.log2(threshold).is_integer(), "LUT quantization threshold must be a power of two")

        # check number of lut values
        self.assertTrue(lut_values.shape[0] <= bins.shape[0],
                        f"Number of lut values is {lut_values.shape[0]} but should not exceed {bins.shape[0]}"),
        # check lut values are rounded
        self.assertTrue(np.all(np.mod(lut_values, 1) == 0), "lut values are supposed to be rounded")


if __name__ == '__main__':
    unittest.main()
