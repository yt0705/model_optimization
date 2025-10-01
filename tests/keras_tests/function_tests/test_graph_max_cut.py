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

import keras
import unittest

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Activation, Add
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input, SeparableConv2D, Reshape
import tensorflow as tf

from model_compression_toolkit.core.common.graph.memory_graph.compute_graph_max_cut import compute_graph_max_cut
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph
from model_compression_toolkit.core.keras.reader.reader import model_reader

import model_compression_toolkit as mct


def simple_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    x_bn = BatchNormalization()(x)
    outputs = ReLU()(x_bn)
    return keras.Model(inputs=inputs, outputs=outputs)


def residual_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    x_bn = BatchNormalization()(x)
    outputs = Add()([ReLU()(x_bn), x])
    return keras.Model(inputs=inputs, outputs=outputs)

def complex_model(input_shape):
    """
    This is a model that has all the different situations that define different structures for the memory graph
    which is used to run astar.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    x_bn = BatchNormalization()(x)
    x_relu = ReLU()(x_bn)
    y = tf.split(x_relu, num_or_size_splits=2, axis=0)
    x1 = Conv2D(2, 3)(y[0])
    x2 = Conv2D(2, 3)(y[1])
    concat = keras.layers.Concatenate()([x1, x2])
    x_bn2 = BatchNormalization()(concat)
    x_relu2 = Activation('relu')(x_bn2)
    outputs = Add()([x_relu2, concat])
    return keras.Model(inputs=inputs, outputs=outputs)


def expanding_model(input_shape):
    """
    This is a model has a split which afterwards increases the size of the output tensor in one of the split paths.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(2, 3)(inputs)
    y = tf.split(x, num_or_size_splits=2, axis=0)
    x1 = Conv2D(2, 3)(y[0])
    x2 = Conv2D(2, 3)(y[1])
    x_expand = Conv2D(20, 1)(x2)
    x_relu = Activation('relu')(x_expand)
    x_shrink = Conv2D(2, 1)(x_relu)
    concat = keras.layers.Concatenate()([x1, x_shrink])
    return keras.Model(inputs=inputs, outputs=concat)


class TestGraphMaxCut(unittest.TestCase):

    def test_graph_max_cut_plain_graph_simple(self):
        input_shape = (8, 8, 3)
        model = simple_model(input_shape)
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        schedule, max_cut_size, cuts = compute_graph_max_cut(memory_graph)
        self.assertIsNotNone(schedule)
        self.assertIsNotNone(cuts)
        self.assertTrue(len(cuts) > 0)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)


    def test_graph_max_cut_residual_graph(self):
        input_shape = (8, 8, 3)
        model = residual_model(input_shape)
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        schedule, max_cut_size, cuts = compute_graph_max_cut(memory_graph)
        self.assertIsNotNone(schedule)
        self.assertIsNotNone(cuts)
        self.assertTrue(len(cuts) > 0)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)

    def test_graph_max_cut_plain_graph_complex(self):
        input_shape = (8, 8, 3)
        model = complex_model(input_shape)
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        schedule, max_cut_size, cuts = compute_graph_max_cut(memory_graph)
        self.assertIsNotNone(schedule)
        self.assertIsNotNone(cuts)
        self.assertTrue(len(cuts) > 0)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)

    def test_graph_max_cut_plain_graph_expanding(self):
        input_shape = (8, 8, 3)
        model = expanding_model(input_shape)
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        schedule, max_cut_size, cuts = compute_graph_max_cut(memory_graph)
        self.assertIsNotNone(schedule)
        self.assertIsNotNone(cuts)
        self.assertTrue(len(cuts) > 0)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)

    def test_graph_max_cut_plain_graph_real_model(self):
        model = MobileNetV2()
        graph = model_reader(model)
        memory_graph = MemoryGraph(graph)

        schedule, max_cut_size, cuts = compute_graph_max_cut(memory_graph, n_iter=50, astar_n_iter=500)
        self.assertIsNotNone(schedule)
        self.assertIsNotNone(cuts)
        self.assertTrue(len(cuts) > 0)
        self.assertTrue(max_cut_size >= memory_graph.memory_lbound_single_op)

    def test_graph_max_cut_deterministic_order(self):
        input_shape = (8, 8, 3)
        model = complex_model(input_shape)
        graph = model_reader(model)

        solutions = [compute_graph_max_cut(MemoryGraph(graph)) for _ in range(10)]

        schedules, max_cut_sizes, cuts_solutions = zip(*solutions)
        assert len(set(max_cut_sizes)) == 1
        # nodes within each cut can be in different order, and cuts can be in different order inside cuts list,
        # but overall the cuts should be identical between different runs
        sorted_cuts_solutions = [sorted(cut.sorted_elements_signature for cut in cuts) for cuts in cuts_solutions]
        assert all(cuts == sorted_cuts_solutions[0] for cuts in sorted_cuts_solutions[1:])
