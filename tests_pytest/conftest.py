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
from unittest.mock import Mock

from pytest import fixture

from model_compression_toolkit.core import FrameworkInfo, QuantizationConfig, QuantizationErrorMethod
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from tests_pytest._test_util import tpc_util


@fixture
def minimal_tpc():
    """ Minimal TPC as a fixture. """
    return tpc_util.minimal_tpc()


@fixture
def graph_mock():
    """ Basic Graph mock. """
    return Mock(spec_set=Graph, nodes=[])


@fixture
def fw_impl_mock():
    """ Basic FrameworkImplementation mock. """
    return Mock(spec_set=FrameworkImplementation)


@fixture
def fw_info_mock():
    """ Basic FrameworkInfo mock. """
    return Mock(spec_set=FrameworkInfo)


@fixture
def quant_config_mock():
    """ Basic QuantizationConfig mock. """
    return Mock(spec=QuantizationConfig, weights_error_method=QuantizationErrorMethod.NOCLIPPING,
                activation_error_method=QuantizationErrorMethod.NOCLIPPING)
