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

import unittest
from unittest.mock import patch

from model_compression_toolkit.core.pytorch.pytorch_device_config import DeviceManager
import torch


class TestDeviceManager(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=False)
    def test_is_valid_device_cpu(self, mock_cuda_available):
        device_name = "cpu"
        is_valid, message = DeviceManager.is_valid_device(device_name)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Valid device")

    @patch('torch.cuda.is_available', return_value=True)
    def test_is_valid_device_cuda_available(self, mock_cuda_available):
        device_name = "cuda"
        is_valid, message = DeviceManager.is_valid_device(device_name)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Valid device")

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    def test_is_valid_device_cuda_invalid_index(self, mock_device_count, mock_cuda_available):
        device_name = "cuda:100"
        is_valid, message = DeviceManager.is_valid_device(device_name)
        self.assertFalse(is_valid)
        self.assertEqual(message, "CUDA device index 100 out of range. Number of valid devices: 1")

    @patch('torch.cuda.is_available', return_value=True)
    def test_is_valid_device_cuda_invalid_format(self, mock_cuda_available):
        device_name = "cuda:invalid"
        is_valid, message = DeviceManager.is_valid_device(device_name)
        self.assertFalse(is_valid)
        self.assertEqual(message, "Invalid CUDA device format. Use 'cuda' or 'cuda:x' where x is the device index.")

    def test_is_valid_device_invalid_device(self):
        device_name = "invalid_device"
        is_valid, message = DeviceManager.is_valid_device(device_name)
        self.assertFalse(is_valid)
        self.assertEqual(message, "Invalid device")


if __name__ == "__main__":
    unittest.main()
