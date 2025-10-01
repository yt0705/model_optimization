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
from typing import Dict, Callable

import torch

from model_compression_toolkit.data_generation.common.enums import BNLayerWeightingType
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import OriginalBNStatsHolder, \
    ActivationExtractor, PytorchActivationExtractor


def average_bn_layer_weighting_fn(orig_bn_stats_holder: OriginalBNStatsHolder,
                                  activation_extractor: PytorchActivationExtractor,
                                  i_iter: int,
                                  n_iter: int) -> Dict[str, float]:
    """
    Calculate average weighting for each batch normalization layer.

    Args:
        orig_bn_stats_holder (OriginalBNStatsHolder): Holder for original batch normalization statistics.
        activation_extractor (PytorchActivationExtractor): The activation extractor for the model.
        i_iter (int): Current optimization iteration.
        n_iter (int): Total number of optimization iterations.

    Returns:
        Dict[str, float]: A dictionary containing layer names as keys and average weightings as values.
    """
    num_bn_layers = orig_bn_stats_holder.get_num_bn_layers()
    return {bn_layer_name: 1 / num_bn_layers for bn_layer_name in orig_bn_stats_holder.get_bn_layer_names()}

def first_bn_multiplier_weighting_fn(orig_bn_stats_holder: OriginalBNStatsHolder,
                                  activation_extractor: PytorchActivationExtractor,
                                  i_iter: int,
                                  n_iter: int) -> Dict[str, float]:
    """
    Calculate layer weightings with a higher multiplier for the first batch normalization layer.

    Args:
        orig_bn_stats_holder (OriginalBNStatsHolder): Holder for original batch normalization statistics.
        activation_extractor (PytorchActivationExtractor): The activation extractor for the model.
        i_iter (int): Current optimization iteration.
        n_iter (int): Total number of optimization iterations.

    Returns:
        Dict[str, float]: A dictionary containing layer names as keys and weightings as values.
    """
    layer_weighting_dict = {orig_bn_stats_holder.get_bn_layer_names()[0]: 10}
    layer_weighting_dict.update({bn_layer_name: 1 for bn_layer_name in orig_bn_stats_holder.get_bn_layer_names()[1:]})
    return layer_weighting_dict


# Dictionary of layer weighting functions
bn_layer_weighting_function_dict: Dict[BNLayerWeightingType, Callable] = {
    BNLayerWeightingType.AVERAGE: average_bn_layer_weighting_fn,
    BNLayerWeightingType.FIRST_LAYER_MULTIPLIER: first_bn_multiplier_weighting_fn,
}
