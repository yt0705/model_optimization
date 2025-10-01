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

from tests.common_tests.base_test import BaseTest


class BaseFeatureNetworkTest(BaseTest):
    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3)):

        super().__init__(unit_test=unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape)

    def get_gptq_config(self):
        return None

    def get_resource_utilization(self):
        return None

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            core_config = self.get_core_config()
            ptq_model, quantization_info = self.get_ptq_facade()(model_float,
                                                                 self.representative_data_gen_experimental,
                                                                 target_resource_utilization=self.get_resource_utilization(),
                                                                 core_config=core_config,
                                                                 target_platform_capabilities=self.get_tpc()
                                                                 )

            self.compare(ptq_model,
                         model_float,
                         input_x=self.representative_data_gen(),
                         quantization_info=quantization_info)


