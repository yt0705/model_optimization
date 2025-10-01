# Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
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
import re

import pytest
import torch
from torch import nn
import torch.testing as ptt

from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PytorchModel
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import BaseTorchIntegrationTest


def get_data_generator():
    def data_gen():
        yield [torch.rand(1, 3, 5, 5)]
    return data_gen


def get_model_with_reused_weights():
    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.shared_conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
            self.another_conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
            self.fc1 = nn.Linear(8 * 3 * 3, 10)
            self.fc2 = nn.Linear(8 * 3 * 3, 10)

        def forward(self, x):
            x1 = self.shared_conv(x)  # shape: (batch_size, 8, height-2, width-2) = (1, 8, 3, 3)
            x2 = self.shared_conv(x)  # shape: (batch_size, 8, height-2, width-2) = (1, 8, 3, 3)
            x = x1 + x2
            return x

    return Model()


class TestWeightsReuse(BaseTorchIntegrationTest):
    """
    Test that reused nodes are always initiated after their group node was initiated.
    We test it by validating that building pytorch model back from the graph succeed with no errors
    (specifically _add_all_modules method), independently to the graph nodes topological order.
    """

    def test_weights_reuse_toposort(self, minimal_tpc):
        """
        Test that reused nodes are successfully initiated after their group node was initiated, and that they use the
        same weights as their group node.
        Test it with nodes sorted in topological order.
        """
        model = get_model_with_reused_weights()
        data_generator = get_data_generator()
        graph = self.run_graph_preparation(model=model, datagen=data_generator, tpc=minimal_tpc)
        pytorch_model = PytorchModel(graph=graph)
        assert len(pytorch_model._reused_nodes) == 1

    def test_weights_reuse_reversed_toposort(self, minimal_tpc):
        """
        Test that reused nodes are successfully initiated after their group node was initiated, and that they use the
        same weights as their group node.
        Test it with nodes sorted in reversed topological order.
        """
        model = get_model_with_reused_weights()
        data_generator = get_data_generator()
        graph = self.run_graph_preparation(model=model, datagen=data_generator, tpc=minimal_tpc)
        pytorch_model = PytorchModel(graph=graph)

        pytorch_model.node_sort.reverse()
        pytorch_model.reuse_groups = {}
        pytorch_model._reused_nodes = []
        pytorch_model._add_all_modules()
        assert len(pytorch_model._reused_nodes) == 1

    def test_reused_only_initialization(self, minimal_tpc):
        """
        Test that in case reused nodes are initiated before none-reused nodes, exception is raised.
        """
        model = get_model_with_reused_weights()
        data_generator = get_data_generator()
        graph = self.run_graph_preparation(model=model, datagen=data_generator, tpc=minimal_tpc)
        pytorch_model = PytorchModel(graph=graph)
        pytorch_model.reuse_groups = {}
        reused_node = pytorch_model._reused_nodes[0]
        with pytest.raises(Exception, match=re.escape(f"Reuse group {reused_node.reuse_group} not found for node "
                                                      f"{reused_node.name}. Make sure you first call the method with "
                                                      f"reused_nodes_only=False")):
            pytorch_model._add_modules(reused_nodes_only=True)
