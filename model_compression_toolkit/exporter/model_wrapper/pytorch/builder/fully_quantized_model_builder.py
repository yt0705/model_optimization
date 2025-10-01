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

from typing import Union, Callable
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.verify_packages import FOUND_TORCH
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common import BaseNode
import model_compression_toolkit.core as C

if FOUND_TORCH:
    import torch
    from mct_quantizers import PytorchQuantizationWrapper, PytorchActivationQuantizationHolder, PytorchPreservingActivationQuantizationHolder
    from mct_quantizers.common.constants import OP_CALL_ARGS, OP_CALL_KWARGS
    from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
    from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode


    def fully_quantized_wrapper(node: common.BaseNode,
                                module: torch.nn.Module,
                                fw_impl) -> Union[torch.nn.Module, PytorchQuantizationWrapper]:
        """
        A function which takes a computational graph node and a pytorch module and
        perform the quantization wrapping

        Args:
            node: A node of mct graph.
            module: A Pytorch module
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        Returns: Wrapped layer

        """
        weight_quantizers, _ = fw_impl.get_inferable_quantizers(node)
        if len(weight_quantizers) > 0:
            # Set reuse for weight quantizers if node is reused
            for _, quantizer in weight_quantizers.items():
                if node.reuse_group:
                    quantizer.reuse = True
            # for positional weights we need to extract the weight's value.
            weights_values = {attr: fw_impl.to_tensor(node.get_weights_by_keys(attr))
                              for attr in weight_quantizers if isinstance(attr, int)}
            # When wrapping functional nodes, need to set call args\kwargs in wrapper, because they
            # are used during wrapper call method.
            # Temporary patch: for torch.gather this is not the case, so args & kwargs shouldn't be
            # saved in the warpper.
            func_node_kwargs = {OP_CALL_ARGS: node.op_call_args,
                                OP_CALL_KWARGS: node.op_call_kwargs
                                } if isinstance(node, FunctionalNode) and not node.functional_op is torch.gather else {}
            return PytorchQuantizationWrapper(module, weight_quantizers, weights_values,
                                              is_inputs_as_list=node.inputs_as_list,
                                              **func_node_kwargs)
        return module


    def get_activation_quantizer_holder(node: BaseNode, holder_type: PytorchActivationQuantizationHolder,
                                        fw_impl, **kwargs) -> Callable:
        """
        Retrieve a PytorchActivationQuantizationHolder layer to use for activation quantization of a node.
        If the layer is not supposed to be wrapped with an activation quantizer - return None.

        Args:
            node: Node to attach a PytorchActivationQuantizationHolder to its output.
            holder_type: The type of the activation quantization holder to use.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            **kwargs: Key-arguments to be passed to the quantization holder initialization to set specific arguments
                based on the holder's type.

        Returns:
            A PytorchActivationQuantizationHolder module for the node's activation quantization.
        """
        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node with no activation
        # quantization, which in this case has an empty list).
        _, activation_quantizers = fw_impl.get_inferable_quantizers(node)
        if len(activation_quantizers) == 1:
            return holder_type(activation_quantizers[0], **kwargs)

        Logger.critical(
            f'PytorchActivationQuantizationHolder supports a single quantizer but {len(activation_quantizers)} quantizers '
            f'were found for node {node}')


    def get_exportable_pytorch_model(graph: Graph):
        """
        Convert graph to fully quantized PyTorch model.

        Args:
            graph: Graph to convert to a PyTorch model.

        Returns:
            Fully quantized PyTorch model.
        """
        fw_impl = C.pytorch.pytorch_implementation.PytorchImplementation()
        exportable_model, user_info = PyTorchModelBuilder(graph=graph,
                                                          wrapper=lambda n, m:
                                                          fully_quantized_wrapper(n, m,
                                                                                  fw_impl=fw_impl),
                                                          get_activation_quantizer_holder_fn=lambda n, holder_type, **kwargs:
                                                          get_activation_quantizer_holder(n, holder_type,
                                                                                          fw_impl=fw_impl, **kwargs)).build_model()

        Logger.info("\nPlease run your accuracy evaluation on the exported quantized model to verify it's accuracy.\n"
                    "Checkout the FAQ and Troubleshooting pages for resolving common issues and improving the quantized model accuracy:\n"
                    "FAQ: https://github.com/sony/model_optimization/tree/main/FAQ.md\n"
                    "Quantization Troubleshooting: https://github.com/sony/model_optimization/tree/main/quantization_troubleshooting.md")

        return exportable_model, user_info


else:
    def get_exportable_pytorch_model(*args, **kwargs):
        Logger.critical("PyTorch must be installed to use 'get_exportable_pytorch_model'. "
                        "The 'torch' package is missing.")  # pragma: no cover
