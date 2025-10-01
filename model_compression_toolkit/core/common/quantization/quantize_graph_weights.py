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

import copy

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.quantization.quantize_node import get_quantized_weights_attr_by_qc
from model_compression_toolkit.logger import Logger


def quantize_graph_weights(graph_to_quantize: Graph) -> Graph:
    """
    Get a graph representing a model, and quantize its nodes' weights.
    Each node is quantized according to the passed framework info and quantization configuration.
    If weights bias correction is enabled in the quantization configuration, a bias correction term
    is calculated and subtracted from the original node's bias. The graph is quantized in-place.

    Args:
        graph_to_quantize: Graph to quantize its nodes.

    """
    _quantized_graph = copy.deepcopy(graph_to_quantize)
    # Iterate over nodes in the graph and quantize each node's weights and activations
    # (according to operators groups in framework info).
    for n in _quantized_graph.nodes():
        for attr in n.get_node_weights_attributes():
            if n.is_weights_quantization_enabled(attr):
                quantized_attr, io_channels_axes = \
                    get_quantized_weights_attr_by_qc(attr,
                                                     n,
                                                     n.final_weights_quantization_cfg.get_attr_config(attr))

                Logger.debug(
                    f'Weights attribute: {attr} of node name: {n.name} has the following quantization params: '
                    f'{str(n.final_weights_quantization_cfg.get_attr_config(attr).weights_quantization_params)}')

                # Set the attribute to be the quantized attribute.
                n.set_weights_by_keys(attr, quantized_attr)

    return _quantized_graph
