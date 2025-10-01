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

from model_compression_toolkit.constants import LUT_VALUES, THRESHOLD, SIGNED, \
    LUT_VALUES_BITWIDTH
from model_compression_toolkit.core.keras.quantizer.lut_fake_quant import activation_lut_kmean_quantizer


class TestLUTQuantizerFakeQuant(unittest.TestCase):

    def test_signed_lut_activation_fake_quant(self):
        threshold = 16
        lut_values = np.array([-8.0, 0.0, 4.0])
        tensor = np.linspace(-1*threshold, threshold, num=2*threshold+1)
        quantization_params = {SIGNED: True,
                               LUT_VALUES: lut_values,
                               THRESHOLD: threshold}

        # We divide the centers in 2^(8-1) (the minus 1 because of the signed quantization)
        div_val_output = (2 ** (LUT_VALUES_BITWIDTH - 1))

        # Construct the FakeQuant
        model = activation_lut_kmean_quantizer(activation_n_bits=8,  # dummy, not used in this function
                                               quantization_params=quantization_params)
        output = model(tensor)

        expected_unique_values = (lut_values / div_val_output) * threshold

        # Check expected unique values of the output
        self.assertTrue((np.unique(output) == expected_unique_values).all())

        # We expected each negative value in the input to be in the first center, each zero to be in the second
        # center and each positive value to be in the last center
        input_sign = np.sign(tensor)
        self.assertTrue((output[input_sign == -1] == expected_unique_values[0]).numpy().all())
        self.assertTrue((output[input_sign == 0] == expected_unique_values[1]).numpy().all())
        self.assertTrue((output[input_sign == 1] == expected_unique_values[2]).numpy().all())

    def test_unsigned_lut_activation_fake_quant(self):
        threshold = 8
        lut_values = np.array([0.0, 256.0])
        tensor = np.linspace(0, threshold, num=threshold+1)
        quantization_params = {SIGNED: False,
                               LUT_VALUES: lut_values,
                               THRESHOLD: threshold}

        # We divide the centers in 2^8
        div_val_output = (2 ** LUT_VALUES_BITWIDTH)

        # Construct the FakeQuant
        model = activation_lut_kmean_quantizer(activation_n_bits=8,  # dummy, not used in this function
                                               quantization_params=quantization_params)
        output = model(tensor)

        expected_unique_values = (lut_values / div_val_output) * threshold

        # Check expected unique values of the output
        self.assertTrue((np.unique(output) == expected_unique_values).all())

        # We expected each value that is lower or equal to 4 in the input to be in the first center and each value
        # bigger than that to be in the last center
        self.assertTrue((output[tensor <= 4] == expected_unique_values[0]).numpy().all())
        self.assertTrue((output[tensor > 4] == expected_unique_values[1]).numpy().all())


if __name__ == '__main__':
    unittest.main()
