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

from tests.keras_tests.pruning_tests.feature_networks.networks_tests.conv2d_conv2dtranspose_pruning_test import \
    Conv2DtoConv2DTransposePruningTest
from tests.keras_tests.pruning_tests.feature_networks.networks_tests.conv2d_pruning_test import Conv2DPruningTest

from tests.keras_tests.pruning_tests.feature_networks.networks_tests.conv2dtranspose_conv2d_pruning_test import \
    Conv2DTransposetoConv2DPruningTest
from tests.keras_tests.pruning_tests.feature_networks.networks_tests.conv2dtranspose_pruning_test import \
    Conv2DTransposePruningTest
from tests.keras_tests.pruning_tests.feature_networks.networks_tests.dense_pruning_test import DensePruningTest
import keras

layers = keras.layers


class PruningFeatureNetworksTest(unittest.TestCase):

    def test_conv2d_pruning(self):
        Conv2DPruningTest(self).run_test()
        Conv2DPruningTest(self, use_bn=True).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=layers.ReLU()).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=layers.Softmax()).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=layers.PReLU()).run_test()
        Conv2DPruningTest(self, simd=2).run_test()
        Conv2DPruningTest(self, use_bn=True, simd=2).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=layers.ReLU(), simd=2).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=layers.Softmax(), simd=2).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=layers.PReLU(), simd=2).run_test()

        # Use dummy LFH
        Conv2DPruningTest(self, use_constant_importance_metric=False).run_test()
        Conv2DPruningTest(self, simd=2, use_constant_importance_metric=False).run_test()

    def test_dense_pruning(self):
        DensePruningTest(self).run_test()
        DensePruningTest(self, use_bn=True).run_test()
        DensePruningTest(self, use_bn=True, activation_layer=layers.ReLU()).run_test()
        DensePruningTest(self, use_bn=True, activation_layer=layers.Softmax()).run_test()
        DensePruningTest(self, use_bn=True, activation_layer=layers.PReLU()).run_test()
        DensePruningTest(self, simd=2).run_test()
        DensePruningTest(self, use_bn=True, simd=2).run_test()
        DensePruningTest(self, use_bn=True, activation_layer=layers.ReLU(), simd=2).run_test()
        DensePruningTest(self, use_bn=True, activation_layer=layers.Softmax(), simd=2).run_test()
        DensePruningTest(self, use_bn=True, activation_layer=layers.PReLU(), simd=2).run_test()
        # Use dummy LFH
        DensePruningTest(self, use_constant_importance_metric=False).run_test()
        DensePruningTest(self, simd=2, use_constant_importance_metric=False).run_test()

    def test_conv2dtranspose_pruning(self):
        Conv2DTransposePruningTest(self, ).run_test()
        Conv2DTransposePruningTest(self, use_bn=True).run_test()
        Conv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.ReLU()).run_test()
        Conv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.Softmax()).run_test()
        Conv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.PReLU()).run_test()
        Conv2DTransposePruningTest(self, simd=2).run_test()
        Conv2DTransposePruningTest(self, use_bn=True, simd=2).run_test()
        Conv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.ReLU(), simd=2).run_test()
        Conv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.Softmax(), simd=2).run_test()
        Conv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.PReLU(), simd=2).run_test()
        # Use dummy LFH
        Conv2DTransposePruningTest(self, use_constant_importance_metric=False).run_test()
        Conv2DTransposePruningTest(self, simd=2, use_constant_importance_metric=False).run_test()

    def test_conv2d_conv2dtranspose_pruning(self):
        Conv2DtoConv2DTransposePruningTest(self).run_test()
        Conv2DtoConv2DTransposePruningTest(self, use_bn=True).run_test()
        Conv2DtoConv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.ReLU()).run_test()
        Conv2DtoConv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.Softmax()).run_test()
        Conv2DtoConv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.PReLU()).run_test()
        Conv2DtoConv2DTransposePruningTest(self, simd=2).run_test()
        Conv2DtoConv2DTransposePruningTest(self, use_bn=True, simd=2).run_test()
        Conv2DtoConv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.ReLU(), simd=2).run_test()
        Conv2DtoConv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.Softmax(), simd=2).run_test()
        Conv2DtoConv2DTransposePruningTest(self, use_bn=True, activation_layer=layers.PReLU(), simd=2).run_test()
        # Use dummy LFH
        Conv2DtoConv2DTransposePruningTest(self, use_constant_importance_metric=False).run_test()
        Conv2DtoConv2DTransposePruningTest(self, simd=2, use_constant_importance_metric=False).run_test()

    def test_conv2dtranspose_conv2d_pruning(self):
        Conv2DTransposetoConv2DPruningTest(self).run_test()
        Conv2DTransposetoConv2DPruningTest(self, use_bn=True).run_test()
        Conv2DTransposetoConv2DPruningTest(self, use_bn=True, activation_layer=layers.ReLU()).run_test()
        Conv2DTransposetoConv2DPruningTest(self, use_bn=True, activation_layer=layers.Softmax()).run_test()
        Conv2DTransposetoConv2DPruningTest(self, use_bn=True, activation_layer=layers.PReLU()).run_test()
        Conv2DTransposetoConv2DPruningTest(self, simd=2).run_test()
        Conv2DTransposetoConv2DPruningTest(self, use_bn=True, simd=2).run_test()
        Conv2DTransposetoConv2DPruningTest(self, use_bn=True, activation_layer=layers.ReLU(), simd=2).run_test()
        Conv2DTransposetoConv2DPruningTest(self, use_bn=True, activation_layer=layers.Softmax(), simd=2).run_test()
        Conv2DTransposetoConv2DPruningTest(self, use_bn=True, activation_layer=layers.PReLU(), simd=2).run_test()
        # Use dummy LFH
        Conv2DTransposetoConv2DPruningTest(self, use_constant_importance_metric=False).run_test()
        Conv2DTransposetoConv2DPruningTest(self, simd=2, use_constant_importance_metric=False).run_test()

if __name__ == '__main__':
    unittest.main()
