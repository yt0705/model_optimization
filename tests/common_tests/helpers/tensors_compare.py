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


import numpy as np


def check_power_of_two(x):
    return np.log2(np.abs(x)) == int(np.log2(np.abs(x)))


def check_quantizer_min_max_are_power_of_two(min_q, max_q, nbits=8):
    is_min_pow_of_two = check_power_of_two(min_q)
    lsb = (max_q - min_q) / (2 ** nbits - 1)
    is_max_pow_of_two = check_power_of_two(max_q + lsb)
    return is_min_pow_of_two, is_max_pow_of_two


def cosine_similarity(a, b, eps=1e-8):
    if np.all(b == 0) and np.all(a == 0):
        return 1.0
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_norm = tensor_norm(a)
    b_norm = tensor_norm(b)

    return np.sum(a_flat * b_flat) / ((a_norm * b_norm) + eps)


def norm_similarity(a, b):
    return tensor_norm(a) / tensor_norm(b)


def normalized_mse(a, b, norm_factor=None):
    batch_size = a.shape[0]

    a = np.reshape(a, [batch_size, -1])
    b = np.reshape(b, [batch_size, -1])

    if norm_factor is None:
        norm_factor = np.square(np.abs(a)).mean(axis=-1)
        norm_factor = np.reshape(norm_factor, [batch_size, 1])

    lsb_error = (np.abs(a - b)**2 / norm_factor)
    return np.mean(lsb_error, axis=-1), np.std(lsb_error, axis=-1), np.max(lsb_error, axis=-1), np.min(lsb_error, axis=-1)


def tensor_norm(a):
    return np.sqrt(np.power(a.flatten(), 2.0).sum())


def tensor_compare(a, b):
    cs_value = cosine_similarity(a, b)
    norm_rate = tensor_norm(a) / tensor_norm(b)
    return cs_value, norm_rate


def calc_db(x, eps=1e-10):
    return 10*np.log10(x+eps)