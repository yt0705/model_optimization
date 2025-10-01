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

import model_compression_toolkit as mct
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc


def get_pytorch_test_tpc_dict(tpc, test_name, ftp_name):
    return {
        test_name: tpc
    }

def get_activation_quantization_disabled_pytorch_tpc(name):
    tp = generate_test_tpc({'enable_activation_quantization': False})
    return get_pytorch_test_tpc_dict(tp, name, name)

def get_weights_quantization_disabled_pytorch_tpc(name):
    tp = generate_test_tpc({'enable_weights_quantization': False})
    return get_pytorch_test_tpc_dict(tp, name, name)


def get_mp_activation_pytorch_tpc_dict(tpc_model, test_name, tpc_name):
    # This is a legacy helper function that is kept for maintaining tests usability
    return {test_name: tpc_model}