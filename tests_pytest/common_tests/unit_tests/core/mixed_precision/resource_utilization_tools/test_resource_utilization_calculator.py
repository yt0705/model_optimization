# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from types import MethodType
from unittest.mock import Mock

import numpy as np
import pytest

from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.graph.memory_graph.compute_graph_max_cut import compute_graph_max_cut
from model_compression_toolkit.core.common.graph.memory_graph.cut import Cut
from model_compression_toolkit.core.common.graph.memory_graph.memory_element import MemoryElements, \
    ActivationMemoryTensor
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitWeightsNode, VirtualSplitActivationNode
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    Utilization, ResourceUtilizationCalculator, TargetInclusionCriterion, BitwidthMode
from tests_pytest._test_util.graph_builder_utils import build_node, build_qc, full_attr_name

BM = BitwidthMode
TIC = TargetInclusionCriterion


class TestUtilization:
    def test_operations(self):
        u = [Utilization(10, 15), Utilization(25, 10), Utilization(35, 5)]
        assert u[0] + u[1] == Utilization(35, 25)
        assert sum(u) == Utilization(70, 30)
        # min/max is by bytes, not size
        assert max(u) == u[0]
        assert min(u) == u[2]

    def test_invalid_radd(self):
        with pytest.raises(ValueError, match='radd is only supported with 0'):
            1 + Utilization(5, 5)


class TestComputeResourceUtilization:
    """ Test compute_resource_utilization public api.

        compute_resource_utilization on a virtual graph is tested in TestBOPSAndVirtualGraph
    """
    @pytest.fixture(autouse=True)
    def setup(self, graph_mock, fw_impl_mock, fw_info_mock):
        n1 = build_node('n1', qcs=[build_qc()], output_shape=(None, 5, 10))
        n2 = build_node('n2', output_shape=(None, 10, 20, 3),
                        canonical_weights={'foo': np.zeros((3, 14))},
                        qcs=[build_qc(w_attr={'foo': (4, True)})])
        n3 = build_node('n3', qcs=[build_qc(4)], output_shape=(None, 2, 71))
        graph = Graph('g', input_nodes=[n1], nodes=[n2], output_nodes=[n3],
                      edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0)])

        fw_info_mock.get_kernel_op_attributes = Mock(return_value=['foo'])    # for bops
        fw_impl_mock.get_node_mac_operations = lambda n, fw_info: 42 if n == n2 else 0    # for bops

        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        # wrap real methods
        ru_calc.compute_activations_utilization = Mock(wraps=ru_calc.compute_activations_utilization)
        ru_calc.compute_weights_utilization = Mock(wraps=ru_calc.compute_weights_utilization)
        ru_calc.compute_bops = Mock(wraps=ru_calc.compute_bops)

        self.ru_calc = ru_calc
        self.nodes = [n1, n2, n3]

    @pytest.mark.parametrize('detailed', [True, False])
    def test_compute_ru_all_targets(self, detailed):
        ru_calc = self.ru_calc
        # default targets
        ret = self.ru_calc.compute_resource_utilization(TIC.AnyQuantized, BM.QDefaultSP, return_detailed=detailed)

        ru_calc.compute_activations_utilization.assert_called_once_with(TIC.AnyQuantized, BM.QDefaultSP, None)
        ru_calc.compute_weights_utilization.assert_called_once_with(TIC.AnyQuantized, BM.QDefaultSP, None)
        ru_calc.compute_bops.assert_called_once_with(TIC.AnyQuantized, BM.QDefaultSP, act_qcs=None, w_qcs=None)

        ru = ret[0] if detailed else ret
        assert ru == ResourceUtilization(weights_memory=21,
                                         activation_memory=671,
                                         total_memory=692,
                                         bops=42*4*8)
        if detailed:
            detailed_res = ret[1]
            assert set(detailed_res.keys()) == set(RUTarget)
            assert sorted(list(detailed_res[RUTarget.ACTIVATION].values())) == [50, 71, 650, 671]
            assert detailed_res[RUTarget.WEIGHTS] == {'n2': 21}
            assert sorted(list(detailed_res[RUTarget.TOTAL].values())) == [71, 92, 671, 692]
            assert detailed_res[RUTarget.BOPS] == {'n2': 42*4*8}

        # explicit targets
        ret2 = ru_calc.compute_resource_utilization(TIC.AnyQuantized, BM.QDefaultSP, ru_targets=list(RUTarget),
                                                    return_detailed=detailed)
        assert ret2 == ret

    @pytest.mark.parametrize('detailed', [True, False])
    def test_compute_ru_w(self, detailed):
        ru_calc = self.ru_calc
        ret = ru_calc.compute_resource_utilization(TIC.Any, BM.Q8Bit, ru_targets=[RUTarget.WEIGHTS],
                                                   return_detailed=detailed)
        ru_calc.compute_weights_utilization.assert_called_once_with(TIC.Any, BM.Q8Bit, None)
        ru_calc.compute_activations_utilization.assert_not_called()
        ru_calc.compute_bops.assert_not_called()
        self._validate(ret, detailed, ResourceUtilization(weights_memory=42))

    @pytest.mark.parametrize('detailed', [True, False])
    def test_compute_ru_act(self, detailed):
        ru_calc = self.ru_calc
        ret = self.ru_calc.compute_resource_utilization(TIC.Any, BM.Q8Bit, ru_targets=[RUTarget.ACTIVATION],
                                                        return_detailed=detailed)

        ru_calc.compute_activations_utilization.assert_called_once_with(TIC.Any, BM.Q8Bit, None)
        ru_calc.compute_weights_utilization.assert_not_called()
        ru_calc.compute_bops.assert_not_called()
        self._validate(ret, detailed, ResourceUtilization(activation_memory=742))

    @pytest.mark.parametrize('detailed', [True, False])
    def test_compute_ru_total(self, detailed):
        ru_calc = self.ru_calc
        ret = ru_calc.compute_resource_utilization(TIC.Any, BM.Q8Bit, ru_targets=[RUTarget.TOTAL],
                                                   return_detailed=detailed)

        ru_calc.compute_activations_utilization.assert_called_once_with(TIC.Any, BM.Q8Bit, None)
        ru_calc.compute_weights_utilization.assert_called_once_with(TIC.Any, BM.Q8Bit, None)
        ru_calc.compute_bops.assert_not_called()
        self._validate(ret, detailed, ResourceUtilization(total_memory=742+42))

    @pytest.mark.parametrize('detailed', [True, False])
    def test_compute_ru_bops(self, detailed):
        ru_calc = self.ru_calc
        ret = ru_calc.compute_resource_utilization(TIC.AnyQuantized, BM.Q8Bit, ru_targets=[RUTarget.BOPS],
                                                   return_detailed=detailed)

        ru_calc.compute_bops.assert_called_once_with(TIC.AnyQuantized, BM.Q8Bit, act_qcs=None, w_qcs=None)
        ru_calc.compute_activations_utilization.assert_not_called()
        ru_calc.compute_weights_utilization.assert_not_called()
        self._validate(ret, detailed, ResourceUtilization(bops=42*8*8))

    def test_compute_ru_custom_w_qcs(self):
        ru_calc = self.ru_calc
        w_qcs = {'n2': build_qc(w_attr={'foo': (16, True)}).weights_quantization_cfg}

        ru_calc.compute_resource_utilization(TIC.AnyQuantized, BM.QCustom, w_qcs=w_qcs)

        ru_calc.compute_activations_utilization.assert_called_once_with(TIC.AnyQuantized, BM.QCustom, None)
        ru_calc.compute_weights_utilization.assert_called_once_with(TIC.AnyQuantized, BM.QCustom, w_qcs)
        ru_calc.compute_bops.assert_called_once_with(TIC.AnyQuantized, BM.QCustom, act_qcs=None, w_qcs=w_qcs)

    def test_compute_ru_custom_a_qcs(self):
        ru_calc = self.ru_calc
        a_qcs = {'n2': build_qc(w_attr={'foo': (16, True)}).activation_quantization_cfg}

        ru_calc.compute_resource_utilization(TIC.AnyQuantized, BM.QCustom, act_qcs=a_qcs)

        ru_calc.compute_activations_utilization.assert_called_once_with(TIC.AnyQuantized, BM.QCustom, a_qcs)
        ru_calc.compute_weights_utilization.assert_called_once_with(TIC.AnyQuantized, BM.QCustom, None)
        ru_calc.compute_bops.assert_called_once_with(TIC.AnyQuantized, BM.QCustom, act_qcs=a_qcs, w_qcs=None)

    @pytest.mark.parametrize('bm', set(BM)-{BM.QCustom})
    def test_unexpected_custom_qcs_for_bitmode(self, bm):
        with pytest.raises(ValueError, match=self.ru_calc.unexpected_qc_error):
            self.ru_calc.compute_resource_utilization(TIC.Any, bm, act_qcs={'n': Mock()})

        with pytest.raises(ValueError, match=self.ru_calc.unexpected_qc_error):
            self.ru_calc.compute_resource_utilization(TIC.Any, bm, w_qcs={'n': Mock()})

    def test_unexpected_custom_qcs_for_targets(self):
        with pytest.raises(ValueError, match='Activation configuration passed but no relevant ru_targets requested.'):
            self.ru_calc.compute_resource_utilization(TIC.Any, BM.QCustom, act_qcs={'n1': Mock()},
                                                      ru_targets=[RUTarget.WEIGHTS])

        with pytest.raises(ValueError, match='Weight configuration passed but no relevant ru_targets requested.'):
            self.ru_calc.compute_resource_utilization(TIC.Any, BM.QCustom, w_qcs={'n1': Mock()},
                                                      ru_targets=[RUTarget.ACTIVATION])

    def test_allowed_unexpected_custom_qcs_for_targets(self):
        ru_calc = self.ru_calc
        ru_calc.compute_resource_utilization(TIC.Any, BM.QCustom, act_qcs={'n1': Mock()}, ru_targets=[RUTarget.WEIGHTS],
                                             allow_unused_qcs=True)
        # unexpected config is converted to None
        ru_calc.compute_weights_utilization.assert_called_once_with(TIC.Any, BM.QCustom, None)

        ru_calc.compute_resource_utilization(TIC.Any, BM.QCustom, w_qcs={'n1': Mock()}, ru_targets=[RUTarget.ACTIVATION],
                                             allow_unused_qcs=True)
        # unexpected config is converted to None
        ru_calc.compute_activations_utilization.assert_called_once_with(TIC.Any, BM.QCustom, None)

    def test_invalid_custom_qcs(self):
        with pytest.raises(ValueError, match=self.ru_calc.unexpected_qc_nodes_error):
            self.ru_calc.compute_resource_utilization(TIC.Any, BM.QCustom, act_qcs={'unknown': Mock()})

        with pytest.raises(ValueError, match=self.ru_calc.unexpected_qc_nodes_error):
            self.ru_calc.compute_resource_utilization(TIC.Any, BM.QCustom, w_qcs={'unknown': Mock()})

    def _validate(self, ret, detailed, exp_ru: ResourceUtilization):
        ru = ret[0] if detailed else ret
        assert ru == exp_ru
        if detailed:
            assert len(ret) == 2
            assert set(ret[1].keys()) == exp_ru.get_restricted_targets()


class TestActivationUtilizationMethods:
    """ Tests for non-public activation utilization api. """
    def test_get_a_nbits_configurable(self, graph_mock, fw_impl_mock, fw_info_mock):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        node = build_node(qcs=[build_qc(7), build_qc(4), build_qc(2)])
        assert ru_calc._get_activation_nbits(node, BM.Float, None) == FLOAT_BITWIDTH
        assert ru_calc._get_activation_nbits(node, BM.QMinBit, None) == 2
        assert ru_calc._get_activation_nbits(node, BM.QMaxBit, None) == 7
        assert ru_calc._get_activation_nbits(node, BM.Q8Bit, None) == 8

    @pytest.mark.parametrize('node', [
        build_node(qcs=[build_qc(42)]),
        build_node(qcs=[build_qc(42, w_attr={'foo': (4, True)}), build_qc(42, w_attr={'foo': (2, False)})])
    ])
    def test_get_a_nbits_nonconfigurable(self, graph_mock, fw_impl_mock, fw_info_mock, node):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        for bm in set(BitwidthMode) - {BM.Float, BM.Q8Bit}:
            assert ru_calc._get_activation_nbits(node, bm, None) == 42
        assert ru_calc._get_activation_nbits(node, BM.Float, None) == FLOAT_BITWIDTH
        assert ru_calc._get_activation_nbits(node, BM.Q8Bit, None) == 8

    @pytest.mark.parametrize('node, qc, exp_nbit', [
        (build_node(qcs=[build_qc(4)]), build_qc(17), 17),
        (build_node(qcs=[build_qc(4)]), build_qc(17, False), 32),
        (build_node(qcs=[build_qc(4, False)]), build_qc(17, True), 17)
    ])
    def test_get_a_nbits_custom(self, graph_mock, fw_impl_mock, fw_info_mock, node, qc, exp_nbit):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        assert ru_calc._get_activation_nbits(node, BM.QCustom, qc.activation_quantization_cfg) == exp_nbit

    @pytest.mark.parametrize('bm', list(BM))
    def test_get_a_nbits_non_q(self, graph_mock, fw_impl_mock, fw_info_mock, bm):
        node = build_node(qcs=[build_qc(a_enable=False)])
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        assert ru_calc._get_activation_nbits(node, bm, None) == FLOAT_BITWIDTH

    def test_get_a_nbits_errors(self, graph_mock, fw_impl_mock, fw_info_mock):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        node = build_node(qcs=[build_qc(8), build_qc(4)])

        with pytest.raises(ValueError, match=f'Could not retrieve the activation quantization candidate for node {node}'):
            ru_calc._get_activation_nbits(node, BM.QCustom, act_qc=None)

        with pytest.raises(ValueError, match='Could not retrieve the activation quantization candidate'):
            ru_calc._get_activation_nbits(node, BM.QDefaultSP, act_qc=None)

    def test_get_target_activation_nodes(self, graph_mock, fw_impl_mock, fw_info_mock):
        sp1 = build_node('n1', qcs=[build_qc(8), build_qc(4)])
        sp2 = build_node('n2', qcs=[build_qc(4, w_attr={'foo': (8, True)}),
                                    build_qc(4, w_attr={'foo': (4, True)})])
        sp3 = build_node('n3', qcs=[build_qc(4)], reuse=True)
        mp = build_node('n4', qcs=[build_qc(4), build_qc(2)], reuse=True)
        noq = build_node('noq', qcs=[build_qc(4, False, w_attr={'foo': (8, True)}),
                                     build_qc(4, False, w_attr={'foo': (4, True)})])

        graph_mock.nodes = [sp1, sp2, sp3, mp, noq]
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)

        assert len(TIC) == 4, 'enum changed, update tests'
        assert ru_calc._get_target_activation_nodes(TIC.QConfigurable, include_reused=True) == [sp1, mp]
        assert ru_calc._get_target_activation_nodes(TIC.QConfigurable, include_reused=False) == [sp1]

        assert ru_calc._get_target_activation_nodes(TIC.QNonConfigurable, include_reused=True) == [sp2, sp3]
        assert ru_calc._get_target_activation_nodes(TIC.QNonConfigurable, include_reused=False) == [sp2]

        assert ru_calc._get_target_activation_nodes(TIC.AnyQuantized, include_reused=True) == [sp1, sp2, sp3, mp]
        assert ru_calc._get_target_activation_nodes(TIC.AnyQuantized, include_reused=False) == [sp1, sp2]

        assert ru_calc._get_target_activation_nodes(TIC.Any, include_reused=True) == [sp1, sp2, sp3, mp, noq]
        assert ru_calc._get_target_activation_nodes(TIC.Any, include_reused=False) == [sp1, sp2, noq]
        # explicit nodes list
        assert ru_calc._get_target_activation_nodes(TIC.QNonConfigurable,
                                                    include_reused=True, nodes=[sp1, sp2, sp3]) == [sp2, sp3]
        # no nodes found
        assert ru_calc._get_target_activation_nodes(TIC.AnyQuantized,
                                                    include_reused=False, nodes=[sp3, mp, noq]) == []


class TestComputeActivationTensorsUtilization:
    """ Tests for activation tensors utilization public apis. """
    def test_compute_node_activation_tensor_utilization(self, graph_mock, fw_impl_mock, fw_info_mock):
        mp_reuse = build_node('mp_reuse', output_shape=(None, 3, 14), qcs=[build_qc(4), build_qc(16)], reuse=True)
        noq = build_node('noq', output_shape=(None, 15, 9), qcs=[build_qc(a_enable=False)])
        graph_mock.nodes = [mp_reuse, noq]

        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        # _get_activation_nbits is already fully checked, just make sure we use it, and use correctly
        ru_calc._get_activation_nbits = Mock(wraps=ru_calc._get_activation_nbits)

        custom_qc = build_qc(16, True).activation_quantization_cfg
        res = ru_calc.compute_node_activation_tensor_utilization(noq, TIC.Any, BM.QCustom, custom_qc)
        ru_calc._get_activation_nbits.assert_called_once_with(noq, BM.QCustom, custom_qc)
        assert res == Utilization(135, 270.)
        # reused is not ignored
        res = ru_calc.compute_node_activation_tensor_utilization(mp_reuse, TIC.QConfigurable, BM.QMinBit)
        assert res == Utilization(42, 21.)
        # not a target node
        res = ru_calc.compute_node_activation_tensor_utilization(noq, TIC.AnyQuantized, BM.QCustom, custom_qc)
        assert res == Utilization(0, 0)
        # no target
        res = ru_calc.compute_node_activation_tensor_utilization(noq, None, BM.Q8Bit)
        assert res == Utilization(135, 540.)

    @pytest.mark.parametrize('bitmode', set(BM)-{BM.QCustom})
    def test_compute_node_activation_tensor_utilization_errors(self, graph_mock, fw_impl_mock, fw_info_mock, bitmode):
        node = build_node(qcs=[build_qc()])
        graph_mock.nodes = [node]
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match=ResourceUtilizationCalculator.unexpected_qc_error):
            ru_calc.compute_node_activation_tensor_utilization(node, TIC.Any, bitmode, qc=build_qc())

    def test_compute_act_tensors_utilization(self, fw_impl_mock, fw_info_mock):
        mp = build_node('mp', output_shape=(None, 3, 14), qcs=[build_qc(4), build_qc(2)])
        noq = build_node('noq', output_shape=(None, 2, 71), qcs=[build_qc(a_enable=False)])
        sp = build_node('sp', output_shape=(None, 59), qcs=[build_qc()], reuse=True)

        g = Graph('g', input_nodes=[mp], nodes=[noq], output_nodes=[sp],
                  edge_list=[Edge(mp, noq, 0, 0), Edge(noq, sp, 0, 0)])
        ru_calc = ResourceUtilizationCalculator(g, fw_impl_mock, fw_info_mock)
        ru_calc._topo_sort = Mock(wraps=ru_calc._topo_sort)
        # wrap the methods that were fully tested separately to verify we use them and use correctly
        ru_calc._get_target_activation_nodes = Mock(wraps=ru_calc._get_target_activation_nodes)
        ru_calc.compute_node_activation_tensor_utilization = Mock(wraps=ru_calc.compute_node_activation_tensor_utilization)

        qcs = {
            'mp': build_qc(a_enable=False).activation_quantization_cfg,
            'noq': build_qc(4, True).activation_quantization_cfg
        }
        # include reuse + custom qc
        total, per_node = ru_calc.compute_activation_tensors_utilization(TIC.Any, BM.QCustom, act_qcs=qcs,
                                                                         include_reused=True)
        assert per_node == {'mp': Utilization(42, 168.), 'noq': Utilization(142, 71.), 'sp': Utilization(59, 59.)}
        assert total == 168.
        ru_calc._get_target_activation_nodes.assert_called_once_with(TIC.Any, include_reused=True)

        ru_calc._topo_sort.assert_called_once()
        assert sorted(ru_calc._topo_sort.call_args.args[0], key=lambda n: n.name) == [mp, noq, sp]

        calls = sorted(ru_calc.compute_node_activation_tensor_utilization.call_args_list,
                       key=lambda call: call.args[0].name)
        assert len(calls) == 3
        assert calls[0].args == (mp, None, BM.QCustom, qcs['mp'])
        assert calls[1].args == (noq, None, BM.QCustom, qcs['noq'])
        assert calls[2].args == (sp, None, BM.QCustom, None)

        # no reused + no custom
        total, per_node = ru_calc.compute_activation_tensors_utilization(TIC.AnyQuantized, BM.QMinBit,
                                                                         include_reused=False)
        ru_calc._get_target_activation_nodes.assert_called_with(TIC.AnyQuantized, include_reused=False)
        assert per_node == {'mp': Utilization(42, 10.5)}
        assert total == 10.5

        # no target nodes
        total, per_node = ru_calc.compute_activation_tensors_utilization(TIC.QNonConfigurable, BM.QMinBit,
                                                                         include_reused=False)
        assert total == 0
        assert per_node == {}

    @pytest.mark.parametrize('bitmode', set(BM) - {BM.QCustom})
    def test_compute_act_tensors_util_unexpected_custom_qcs(self, graph_mock, fw_impl_mock, fw_info_mock, bitmode):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match=ResourceUtilizationCalculator.unexpected_qc_error):
            ru_calc.compute_activation_tensors_utilization(TIC.Any, bitmode, act_qcs={'n': Mock()})

    def test_compute_act_tensors_util_invalid_custom_qcs(self, graph_mock, fw_impl_mock, fw_info_mock):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match=ResourceUtilizationCalculator.unexpected_qc_nodes_error):
            ru_calc.compute_activation_tensors_utilization(TIC.Any, BitwidthMode.QCustom,
                                                           act_qcs={'unknown': Mock()})


class TestActivationMaxCutUtilization:
    """ Tests for activation max cut utilization. """
    def test_compute_cuts_integration(self, graph_mock, fw_impl_mock, fw_info_mock, mocker):
        """ Test integration with max cut computation. """
        # Test a simple linear dummy graph with the real max cut computation.
        n1 = build_node('n1', qcs=[build_qc()], input_shape=(None, 10, 20, 3), output_shape=(None, 10, 20, 3))
        n2 = build_node('n2', qcs=[build_qc()], input_shape=(None, 10, 20, 3), output_shape=(None, 5, 10))
        n3 = build_node('n3', qcs=[build_qc()], input_shape=(None, 5, 10), output_shape=(None, 5, 10))
        n4 = build_node('n4', qcs=[build_qc()], input_shape=(None, 5, 10, 32), output_shape=(None, 5, 10, 32))
        edges = [Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0), Edge(n3, n4, 0, 0)]
        graph = Graph('g', input_nodes=[n1], nodes=[n2, n3], output_nodes=[n4], edge_list=edges)
        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        # wrap the real implementation
        maxcut_spy = mocker.patch('model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.'
                                  'resource_utilization_calculator.compute_graph_max_cut', wraps=compute_graph_max_cut)

        # trigger cuts cache computation
        cuts_cache = ru_calc.cuts

        # verify the cache
        assert len(cuts_cache) == 5
        assert all(isinstance(k, Cut) for k in cuts_cache.keys())
        # for each cut we save a list of its nodes
        cuts_nodes = {tuple(sorted(n.name for n in nodes)) for nodes in cuts_cache.values()}
        assert cuts_nodes == {('n1',), ('n4',), ('n1', 'n2'), ('n2', 'n3'), ('n3', 'n4')}

        # verify cuts computation only happens the first time
        cuts_cache2 = ru_calc.cuts
        maxcut_spy.assert_called_once()
        assert cuts_cache2 == cuts_cache

        # map from node names to cuts to retrieve the cuts
        nodes_to_cuts = {tuple(sorted(elem.node_name for elem in cut.mem_elements.elements)): cut
                         for cut in cuts_cache.keys()}
        cut1 = nodes_to_cuts[('n1',)]
        cut12 = nodes_to_cuts[('n1', 'n2')]
        cut23 = nodes_to_cuts[('n2', 'n3')]
        cut34 = nodes_to_cuts[('n3', 'n4')]
        cut4 = nodes_to_cuts[('n4',)]

        # compute utilization to check everything works together with real maxcut
        total, per_cut, per_cut_per_node = ru_calc.compute_activation_utilization_by_cut(target_criterion=TIC.AnyQuantized,
                                                                                         bitwidth_mode=BM.QDefaultSP)

        assert per_cut_per_node == {cut1: {'n1': Utilization(10 * 20 * 3, 600)},
                                    cut12: {'n1': Utilization(10 * 20 * 3, 600),
                                            'n2': Utilization(5 * 10, 50)},
                                    cut23: {'n2': Utilization(5*10, 50),
                                            'n3': Utilization(5*10, 50)},
                                    cut34: {'n3': Utilization(5*10, 50),
                                            'n4': Utilization(5*10*32, 1600)},
                                    cut4: {'n4': Utilization(5 * 10 * 32, 1600)}}
        assert per_cut == {
            nodes_to_cuts[('n1',)]: Utilization(600, 600),
            nodes_to_cuts[('n1', 'n2')]: Utilization(650, 650),
            nodes_to_cuts[('n2', 'n3')]: Utilization(100, 100),
            nodes_to_cuts[('n3', 'n4')]: Utilization(1650, 1650),
            nodes_to_cuts[('n4',)]: Utilization(1600, 1600)
        }
        assert total == 1650

    @pytest.fixture
    def prepare_compute_cuts(self, graph_mock, fw_impl_mock, fw_info_mock, mocker):
        # reused nodes should be always included
        mp_reuse = build_node('mp_reuse', qcs=[build_qc(5), build_qc(2)], output_shape=(None, 24), reuse=True)
        mp = build_node('mp', qcs=[build_qc(4), build_qc(2)], output_shape=(None, 5, 10))
        noq = build_node('noq', qcs=[build_qc(6, False)], output_shape=(None, 300))
        sp = build_node('sp', qcs=[build_qc(3)], output_shape=(None, 20, 10))
        mp2 = build_node('mp2', qcs=[build_qc(2), build_qc(4)], output_shape=(None, 150))

        nodes = [mp_reuse, mp, noq, sp, mp2]
        graph_mock.nodes = nodes
        # use the Graph original method (need to bind it to graph_mock instance)
        graph_mock.find_node_by_name = MethodType(Graph.find_node_by_name, graph_mock)

        # we should not use total size, setting it to bad number
        cut_elems1 = MemoryElements(elements={ActivationMemoryTensor(mp_reuse.output_shape, 'mp_reuse', 0)}, total_size=-1)
        cut_elems2 = MemoryElements(elements={ActivationMemoryTensor(mp_reuse.output_shape, 'mp_reuse', 0),
                                              ActivationMemoryTensor(mp.output_shape, 'mp', 0),
                                              ActivationMemoryTensor(noq.output_shape, 'noq', 0),
                                              ActivationMemoryTensor(sp.output_shape, 'sp', 0)}, total_size=-1)
        cut_elems3 = MemoryElements(elements={ActivationMemoryTensor(sp.output_shape, 'sp', 0),
                                              ActivationMemoryTensor(noq.output_shape, 'noq', 0)}, total_size=-1)
        cut_elems4 = MemoryElements(elements={ActivationMemoryTensor(mp2.output_shape, 'mp2', 0)}, total_size=-1)

        cuts = [Cut([], set(), mem_elements=cut_elems)
                for cut_elems in [cut_elems1, cut_elems2, cut_elems3, cut_elems4]]
        mocker.patch.object(ResourceUtilizationCalculator, '_compute_cuts', Mock(return_value=cuts))
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        return ru_calc, cuts, nodes

    def test_get_cut_target_nodes(self, prepare_compute_cuts):
        ru_calc, (cut1, cut2, cut3, cut4), (mp_reuse, mp, noq, sp, mp2) = prepare_compute_cuts
        assert len(TIC) == 4
        sorted_res = lambda res: sorted(res, key=lambda n: n.name)
        assert sorted_res(ru_calc._get_cut_target_nodes(cut2, TIC.Any)) == [mp, mp_reuse, noq, sp]
        assert sorted_res(ru_calc._get_cut_target_nodes(cut2, TIC.AnyQuantized)) == [mp, mp_reuse, sp]
        assert sorted_res(ru_calc._get_cut_target_nodes(cut2, TIC.QConfigurable)) == [mp, mp_reuse]
        assert sorted_res(ru_calc._get_cut_target_nodes(cut2, TIC.QNonConfigurable)) == [sp]

    def test_compute_act_utilization_by_cut(self, prepare_compute_cuts):
        ru_calc, (cut1, cut2, cut3, cut4), (mp_reuse, mp, noq, sp, mp2) = prepare_compute_cuts

        ru_calc.compute_node_activation_tensor_utilization = Mock(wraps=ru_calc.compute_node_activation_tensor_utilization)
        ru_calc._get_cut_target_nodes = Mock(wraps=ru_calc._get_cut_target_nodes)

        qcs = {'mp_reuse': build_qc(7), 'mp': build_qc(10), 'noq': build_qc(4, True), 'mp2': build_qc(4, False)}
        qcs = {k: v.activation_quantization_cfg for k, v in qcs.items()}
        total, per_cut, per_cut_node = ru_calc.compute_activation_utilization_by_cut(TIC.AnyQuantized, BM.QCustom, qcs)

        cut_nodes_calls = ru_calc._get_cut_target_nodes.call_args_list
        assert len(cut_nodes_calls ) == 4
        assert {call.args[0] for call in cut_nodes_calls} == {cut1, cut2, cut3, cut4}
        assert {call.args[1] for call in cut_nodes_calls } == {TIC.AnyQuantized}

        compute_tensor_calls = sorted(ru_calc.compute_node_activation_tensor_utilization.call_args_list,
                                      key=lambda call: call.args[0].name)
        assert len(compute_tensor_calls) == 6
        assert compute_tensor_calls[0].args == (mp, TIC.AnyQuantized, BM.QCustom, qcs['mp'])
        assert compute_tensor_calls[-1].args == (sp, TIC.AnyQuantized, BM.QCustom, None)

        assert len(per_cut_node) == 4
        assert per_cut_node[cut1] == {'mp_reuse': Utilization(24, 21.)}
        assert per_cut_node[cut2] == {'mp_reuse': Utilization(24, 21.),
                                      'mp': Utilization(50, 62.5),
                                      'sp': Utilization(200, 75.)}
        assert per_cut_node[cut3] == {'sp': Utilization(200, 75.)}
        assert per_cut_node[cut4] == {'mp2': Utilization(150, 600.)}

        assert per_cut == {cut1: Utilization(24, 21.),
                           cut2: Utilization(274, 158.5),
                           cut3: Utilization(200, 75.),
                           cut4: Utilization(150, 600.),
                           }
        assert total == 600.

    def test_compute_act_utilization_by_cut_no_cut_nodes(self, prepare_compute_cuts):
        ru_calc, (cut1, cut2, cut3, cut4), (mp_reuse, mp, noq, sp, mp2) = prepare_compute_cuts

        total, per_cut, per_cut_node = ru_calc.compute_activation_utilization_by_cut(TIC.QNonConfigurable, BM.QDefaultSP)
        assert len(per_cut_node) == 2
        assert per_cut_node[cut2] == {'sp': Utilization(200, 75.)}
        assert per_cut_node[cut3] == {'sp': Utilization(200, 75.)}
        assert per_cut == {cut2: Utilization(200, 75.),
                           cut3: Utilization(200, 75.)}
        assert total == 75.

    def test_compute_act_utilization_by_cut_no_target_nodes(self, graph_mock, fw_impl_mock, fw_info_mock):
        node = build_node(qcs=[build_qc(a_enable=False)])
        graph_mock.nodes = [node]
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        from unittest.mock import MagicMock

        ru_calc._compute_cuts = MagicMock()
        ru_calc._get_target_activation_nodes = Mock(wraps=ru_calc._get_target_activation_nodes)

        assert ru_calc.compute_activation_utilization_by_cut(TIC.AnyQuantized, BM.Float) == (0, {}, {})
        ru_calc._compute_cuts.assert_not_called()
        ru_calc._get_target_activation_nodes.assert_called_with(TIC.AnyQuantized, include_reused=True)

        # make sure _compute_cuts is supposed to be called when cuts are accessed, otherwise the test is meaningless
        ru_calc.cuts
        ru_calc._compute_cuts.assert_called()

    @pytest.mark.parametrize('bitmode', set(BM)-{BM.QCustom})
    def test_compute_act_utilization_by_cut_unexpected_custom_qcs(self, graph_mock, fw_impl_mock, fw_info_mock, bitmode):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match=ResourceUtilizationCalculator.unexpected_qc_error):
            ru_calc.compute_activation_utilization_by_cut(TIC.Any, bitmode, act_qcs={'n': Mock()})

    def test_compute_act_utilization_by_cut_invalid_custom_qcs(self, graph_mock, fw_impl_mock, fw_info_mock):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match=ResourceUtilizationCalculator.unexpected_qc_nodes_error):
            ru_calc.compute_activation_utilization_by_cut(TIC.Any, BM.QCustom, act_qcs={'unknown': Mock()})


class TestWeightUtilizationMethods:
    """ Tests for weights utilization non-public api. """

    def test_get_w_nbits(self, graph_mock, fw_impl_mock, fw_info_mock):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        node = build_node(canonical_weights={'mp': 1, 'sp': 2, 'noq': 3})
        node.candidates_quantization_cfg = [
            build_qc(1, w_attr={'mp': (2, True), 'sp': (5, True), 'noq': (12, False)}, pos_attr=(6, True, [2])),
            build_qc(10, w_attr={'mp': (4, True), 'sp': (5, True), 'noq': (1, False)}, pos_attr=(6, True, [2])),
            build_qc(8, False, w_attr={'mp': (7, True), 'sp': (5, True), 'noq': (2, False)}, pos_attr=(6, True, [2]))
        ]

        # configurable attr
        assert ru_calc._get_weight_nbits(node, 'mp', BM.Float, w_qc=None) == FLOAT_BITWIDTH
        assert ru_calc._get_weight_nbits(node, 'mp', BM.QMinBit, w_qc=None) == 2
        assert ru_calc._get_weight_nbits(node, 'mp', BM.QMaxBit, w_qc=None) == 7
        assert ru_calc._get_weight_nbits(node, 'mp', BM.Q8Bit, w_qc=None) == 8

        # non-configurable attr with multiple qcs with same w precision
        for bm in set(BitwidthMode) - {BM.Float, BM.Q8Bit}:
            assert ru_calc._get_weight_nbits(node, 'sp', bm, w_qc=None) == 5
        assert ru_calc._get_weight_nbits(node, 'sp', BM.Float, w_qc=None) == FLOAT_BITWIDTH
        assert ru_calc._get_weight_nbits(node, 'sp', BM.Q8Bit, w_qc=None) == 8
        # positional
        assert ru_calc._get_weight_nbits(node, 2, BM.QMaxBit, w_qc=None) == 6

        # for un-quantized, all modes return float
        for bm in set(BitwidthMode):
            assert ru_calc._get_weight_nbits(node, 'noq', bm, w_qc=None) == FLOAT_BITWIDTH

        # qc is passed but doesn't contain the weight, retrieve from the node
        qc = build_qc(w_attr={'foo': (10, True)})
        assert ru_calc._get_weight_nbits(node, 'sp', BM.QCustom, w_qc=qc.weights_quantization_cfg) == 5
        assert ru_calc._get_weight_nbits(node, 2, BM.QCustom, w_qc=qc.weights_quantization_cfg) == 6

        # custom qc
        custom_qc = build_qc(w_attr={'foo': (42, True), 'sp': (43, False), 'noq': (44, True)}, pos_attr=(11, True, [2]))
        wqc = custom_qc.weights_quantization_cfg
        assert ru_calc._get_weight_nbits(node, 'foo', BM.QCustom, w_qc=wqc) == 42
        assert ru_calc._get_weight_nbits(node, 2, BM.QCustom, w_qc=wqc) == 11
        # non-quantized qc for quantized weight
        assert ru_calc._get_weight_nbits(node, 'sp', BM.QCustom, w_qc=wqc) == FLOAT_BITWIDTH
        # quantized qc for non-quantized weight
        assert ru_calc._get_weight_nbits(node, 'noq', BM.QCustom, w_qc=wqc) == 44

    def test_get_w_nbits_errors(self, graph_mock, fw_impl_mock, fw_info_mock):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        node = build_node(canonical_weights={'foo': 1, 1: 2},
                          qcs=[build_qc(w_attr={'foo': (4, True)}, pos_attr=(4, True, [1])),
                               build_qc(w_attr={'foo': (8, True)}, pos_attr=(8, True, [1]))])
        # qc not passed for configurable attr
        with pytest.raises(ValueError, match='Could not retrieve the quantization candidate for attr foo'):
            ru_calc._get_weight_nbits(node, 'foo', BM.QCustom, w_qc=None)

        # qc passed but doesnt contain all configurable attrs
        qc = build_qc(w_attr={'foo': (8, True)}).weights_quantization_cfg
        with pytest.raises(ValueError, match='Could not retrieve the quantization candidate for attr 1'):
            ru_calc._get_weight_nbits(node, 1, BM.QCustom, w_qc=qc)

        # default bit mode cannot be requested for configurable attrs.
        with pytest.raises(ValueError, match='Could not retrieve the quantization candidate for attr foo'):
            ru_calc._get_weight_nbits(node, 'foo', BM.QDefaultSP, w_qc=None)

    def test_get_target_weight_attrs(self, graph_mock, fw_impl_mock, fw_info_mock):
        weights = {
            'foo': np.array(1.),
            'bar': np.array(2.),
            'baz': np.array(3.),
            1: np.array(4.),
            2: np.array(5.)
        }
        # default weight cfg is used for positional weights
        qcs = [
            build_qc(w_attr={'foo': (8, True), 'bar': (2, True), 'baz': (4, False)}, pos_attr=(10, True, [1, 2])),
            build_qc(w_attr={'foo': (4, True), 'bar': (2, True), 'baz': (2, False)}, pos_attr=(10, True, [2, 1]))
        ]
        node = build_node(canonical_weights=weights, qcs=qcs)
        assert node.has_positional_weights

        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        assert len(TIC) == 4, 'enum changed, update the test'
        assert ru_calc._get_target_weight_attrs(node, TIC.QConfigurable) == full_attr_name(['foo'])
        assert ru_calc._get_target_weight_attrs(node, TIC.QNonConfigurable) == full_attr_name(['bar', 1, 2])
        assert ru_calc._get_target_weight_attrs(node, TIC.AnyQuantized) == full_attr_name(['foo', 'bar', 1, 2])
        assert ru_calc._get_target_weight_attrs(node, TIC.Any) == full_attr_name(['foo', 'bar', 'baz', 1, 2])

    def test_collect_target_nodes_w_attrs(self, graph_mock, fw_impl_mock, fw_info_mock):
        node = build_node('mixed', canonical_weights={'foo': np.array(1.), 'bar': np.array(2.), 3: np.array(3.)},
                          qcs=[build_qc(w_attr={'foo': (8, True), 'bar': (2, True)}, pos_attr=(4, False, [3])),
                               build_qc(w_attr={'foo': (4, True), 'bar': (2, True)}, pos_attr=(2, False, [3]))])

        # should never be selected
        node_no_weights = build_node('no_w', qcs=[build_qc()])

        node_reuse = build_node('reuse', canonical_weights={'foo': np.array(1.), 1: np.array(2.)},
                                qcs=[build_qc(w_attr={'foo': (8, True)}, pos_attr=(4, True, [1]))], reuse=True)

        graph_mock.nodes = [node, node_no_weights, node_reuse]
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)

        # we only cover options relevant to this level, as test_get_target_weight_attrs fully covers node's attrs
        assert (ru_calc._collect_target_nodes_w_attrs(TIC.Any, include_reused=True) ==
                {node: full_attr_name(['foo', 'bar', 3]),
                 node_reuse: [full_attr_name('foo'), 1]})

        assert (ru_calc._collect_target_nodes_w_attrs(TIC.Any, include_reused=False) ==
                {node: full_attr_name(['foo', 'bar', 3])})

        assert (ru_calc._collect_target_nodes_w_attrs(TIC.QConfigurable, include_reused=True) ==
                {node: [full_attr_name('foo')]})


class TestComputeNodeWeightsUtilization:
    """ Tests for compute_node_weight_utilization public method. """

    @pytest.fixture
    def setup_node_w_test(self, graph_mock, fw_impl_mock, fw_info_mock):
        weights = {
            'mp': np.ones((3, 4, 5, 6)),
            'sp': np.full((10, 20), 2),
            'noq': np.full((15,), 3),
            1: np.full((2, 3, 5), 4),
            2: np.full((2, 3), 5)
        }
        qcs = [
            build_qc(w_attr={'mp': (16, True), 'sp': (4, True), 'noq': (5, False)}, pos_attr=(4, True, [1, 2])),
            build_qc(w_attr={'mp': (4, True), 'sp': (4, True), 'noq': (6, False)}, pos_attr=(8, True, [1, 2]))
        ]
        node = build_node(canonical_weights=weights, qcs=qcs)
        graph_mock.nodes = [node]
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        # wrap the original methods to verify integration
        ru_calc._get_weight_nbits = Mock(wraps=ru_calc._get_weight_nbits)
        ru_calc._get_target_weight_attrs = Mock(wraps=ru_calc._get_target_weight_attrs)
        return node, ru_calc

    def test_compute_node_w_utilization_custom_qc(self, setup_node_w_test):
        node, ru_calc = setup_node_w_test
        # _get_weight_nbits and _get_target_weight_attrs are fully tested separately, we wrap the implementation
        # to verify integration. No need to test all cases again.
        custom_qc = build_qc(w_attr={'mp': (3, True), 'noq': (4, True)},
                             pos_attr=(2, True, [1, 2])).weights_quantization_cfg
        total, per_attr = ru_calc.compute_node_weights_utilization(node, TIC.QConfigurable, BM.QCustom, qc=custom_qc)

        ru_calc._get_target_weight_attrs.assert_called_once_with(node, TIC.QConfigurable)
        call_args = [call.args for call in ru_calc._get_weight_nbits.call_args_list]
        assert len(call_args) == 3
        assert set(call_args) == {(node, full_attr_name('mp'), BM.QCustom, custom_qc),
                                  (node, 1, BM.QCustom, custom_qc),
                                  (node, 2, BM.QCustom, custom_qc)}
        assert per_attr == {full_attr_name('mp'): Utilization(360, 135.),
                            1: Utilization(30, 7.5),
                            2: Utilization(6, 1.5)}
        assert total == Utilization(396, 144.)

    def test_compute_node_w_utilization_explicit_attrs_non_custom(self, setup_node_w_test):
        node, ru_calc = setup_node_w_test
        # explicit attrs list, no custom qc
        total, per_attr = ru_calc.compute_node_weights_utilization(node, full_attr_name(['mp', 'noq', 2]),
                                                                   BM.QMinBit)
        ru_calc._get_target_weight_attrs.assert_not_called()
        call_args = [call.args for call in ru_calc._get_weight_nbits.call_args_list]
        assert len(call_args) == 3
        assert set(call_args) == {(node, full_attr_name('mp'), BM.QMinBit, None),
                                  (node, full_attr_name('noq'), BM.QMinBit, None),
                                  (node, 2, BM.QMinBit, None)}
        assert per_attr == {full_attr_name('mp'): Utilization(360, 180.),
                            full_attr_name('noq'): Utilization(15, 60.),
                            2: Utilization(6, 3.)}

    @pytest.mark.parametrize('node', [
        build_node(qcs=[build_qc()]),
        build_node(qcs=[build_qc(w_attr={'foo': (4, False)})])
    ])
    def test_compute_node_w_utilization_no_weights(self, graph_mock, fw_impl_mock, fw_info_mock, node):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)

        total, detailed = ru_calc.compute_node_weights_utilization(node, TIC.AnyQuantized, BM.Float)
        assert total == Utilization(0, 0) and detailed == {}

    def test_compute_node_w_utilization_errors(self, graph_mock, fw_impl_mock, fw_info_mock, setup_node_w_test):
        node = build_node(canonical_weights={'foo': 1, 1: 2}, qcs=[build_qc(w_attr={'foo': (4, True)}),
                                                                   build_qc(w_attr={'foo': (8, True)})])
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)

        # qc for non-custom mode
        qc = build_qc().weights_quantization_cfg
        with pytest.raises(ValueError, match=ResourceUtilizationCalculator.unexpected_qc_error):
            ru_calc.compute_node_weights_utilization(node, TIC.AnyQuantized, BM.QMaxBit, qc)

        qc = build_qc(w_attr={'whoisit': (4, True)}, pos_attr=(4, True, [77])).weights_quantization_cfg
        with pytest.raises(ValueError, match='Custom configuration contains unexpected weight attr'):
            ru_calc.compute_node_weights_utilization(node, TIC.Any, BM.QCustom, qc=qc)

        with pytest.raises(ValueError, match='Explicit list of attributes to compute cannot be empty'):
            ru_calc.compute_node_weights_utilization(node, target_criterion=[], bitwidth_mode=BM.QMaxBit, qc=None)


class TestComputeWeightUtilization:
    """ Tests for compute_weight_utilization public method. """
    @pytest.fixture
    def prepare_compute_w_util(self, fw_impl_mock, fw_info_mock):
        n1 = build_node('n1',
                        canonical_weights={'mp': np.ones((5, 10)), 'sp': np.zeros((42,)), 'noq': np.ones((3, 1, 4))},
                        qcs=[build_qc(w_attr={'mp': (6, True), 'sp': (4, True), 'noq': (8, False)}),
                             build_qc(w_attr={'mp': (2, True), 'sp': (4, True), 'noq': (4, False)})])
        n2 = build_node('n2', canonical_weights={1: np.ones((2, 3, 4, 5, 6)), 'mp': np.ones((31, 4))},
                        qcs=[build_qc(a_enable=False, w_attr={'mp': (4, True)}, pos_attr=(4, True, [1])),
                             build_qc(a_enable=False, w_attr={'mp': (16, True)}, pos_attr=(8, True, [1]))])
        n3 = build_node('n3', canonical_weights={'sp': np.ones((123,))}, qcs=[build_qc(w_attr={'sp': (2, True)})])
        # reused - should never be collected
        n_reuse = build_node('reused', canonical_weights={'sp': np.ones((31, 4))},
                             qcs=[build_qc(w_attr={'sp': (4, True)})], reuse=True)
        # no weights - should never be collected
        n_no_w = build_node('no_w', qcs=[build_qc()])

        g = Graph('g', nodes=[n_reuse, n_no_w], input_nodes=[n1], output_nodes=[n3],
                  edge_list=[Edge(*ns, 0, 0) for ns in [(n1, n_reuse), (n_reuse, n_no_w), (n_no_w, n2), (n2, n3)]])

        ru_calc = ResourceUtilizationCalculator(g, fw_impl_mock, fw_info_mock)
        # wrap original methods for api checks
        ru_calc._topo_sort = Mock(wraps=ru_calc._topo_sort)
        ru_calc._collect_target_nodes_w_attrs = Mock(wraps=ru_calc._collect_target_nodes_w_attrs)
        ru_calc.compute_node_weights_utilization = Mock(wraps=ru_calc.compute_node_weights_utilization)
        return ru_calc, {n.name: n for n in [n1, n2, n3, n_reuse, n_no_w]}

    def test_compute_weights_utilization_custom(self, prepare_compute_w_util):
        ru_calc, nodes = prepare_compute_w_util
        n1, n2, n3 = nodes['n1'], nodes['n2'], nodes['n3']
        # n3 - not in qc (but should be considered)
        custom_qc = {n1.name: build_qc(w_attr={'mp': (5, False), 'noq': (16, True)}).weights_quantization_cfg,
                     n2.name: build_qc(w_attr={'mp': (6, True)}, pos_attr=(2, True, [1])).weights_quantization_cfg,
                     nodes['no_w'].name: build_qc().weights_quantization_cfg,
                     nodes['reused'].name: build_qc(w_attr={'sp': (8, True)})}

        total, per_node, per_weight = ru_calc.compute_weights_utilization(TIC.Any, BM.QCustom, custom_qc)

        ru_calc._collect_target_nodes_w_attrs.assert_called_once_with(TIC.Any, include_reused=False)

        ru_calc._topo_sort.assert_called_once()
        assert sorted(ru_calc._topo_sort.call_args.args[0], key=lambda n: n.name) == [n1, n2, n3]

        calls = [call for call in ru_calc.compute_node_weights_utilization.call_args_list]
        assert len(calls) == 3
        calls = sorted(calls, key=lambda call: call.args[0].name)
        # first call
        assert (calls[0].args[0], *calls[0].args[2:]) == (n1, BitwidthMode.QCustom, custom_qc['n1'])
        assert sorted(calls[0].args[1]) == full_attr_name(['mp', 'noq', 'sp'])
        # second call
        assert (calls[1].args[0], *calls[1].args[2:]) == (n2, BitwidthMode.QCustom, custom_qc['n2'])
        assert calls[1].args[1] in (full_attr_name(['mp', 1]), full_attr_name([1, 'mp']))
        # third call
        assert (calls[2].args[0], *calls[2].args[2:]) == (n3, BitwidthMode.QCustom, None)
        assert calls[2].args[1] == [full_attr_name('sp')]

        # check the actual results
        assert len(per_weight) == len(per_node) == 3
        assert per_weight[n1.name] == {full_attr_name('mp'): Utilization(50, 200.),
                                       full_attr_name('sp'): Utilization(42, 21),
                                       full_attr_name('noq'): Utilization(12, 24.)}
        assert per_weight[n2.name] == {full_attr_name('mp'): Utilization(124, 93.),
                                       1: Utilization(720, 180.)}
        assert per_weight[n3.name] == {full_attr_name('sp'): Utilization(123, 30.75)}

        assert per_node == {n1.name: Utilization(104, 245.),
                            n2.name: Utilization(844, 273.),
                            n3.name: Utilization(123, 30.75)}
        assert total == 245+273+30.75

    def test_compute_w_utilization_non_custom(self, prepare_compute_w_util):
        ru_calc, nodes = prepare_compute_w_util
        n1, n2 = nodes['n1'], nodes['n2']
        ru_calc.compute_weights_utilization(TIC.QConfigurable, BM.QMaxBit)

        ru_calc._collect_target_nodes_w_attrs.assert_called_once_with(TIC.QConfigurable, include_reused=False)
        calls = [call for call in ru_calc.compute_node_weights_utilization.call_args_list]
        assert len(calls) == 2
        calls = sorted(calls, key=lambda call: call.args[0].name)
        assert calls[0].args == (n1, [full_attr_name('mp')], BM.QMaxBit, None)
        assert calls[1].args in [(n2, full_attr_name(attrs), BM.QMaxBit, None) for attrs in (['mp', 1], [1, 'mp'])]

    def test_compute_w_utilization_no_targets(self, graph_mock, fw_impl_mock, fw_info_mock):
        graph_mock.nodes = [
            build_node('n1', qcs=build_qc()),
            build_node('n2', canonical_weights={'foo': np.ones((5,))}, qcs=[build_qc(w_attr={'foo': (8, True)})])
        ]
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        total, per_node, per_weight = ru_calc.compute_weights_utilization(TIC.QConfigurable, BM.Float)
        assert total == 0
        assert per_node == {}
        assert per_weight == {}

    @pytest.mark.parametrize('bm', set(BM)-{BM.QCustom})
    def test_compute_w_utilization_unexpected_custom_qcs(self, graph_mock, fw_impl_mock, fw_info_mock, bm):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match=ResourceUtilizationCalculator.unexpected_qc_error):
            ru_calc.compute_weights_utilization(TIC.Any, BM.QMaxBit, w_qcs={'n': Mock()})

    def test_compute_w_utilization_invalid_custom_qcs(self, graph_mock, fw_impl_mock, fw_info_mock):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match=ResourceUtilizationCalculator.unexpected_qc_nodes_error):
            ru_calc.compute_weights_utilization(TIC.Any, BM.QCustom, w_qcs={'unknown': Mock()})


class TestCalculatorMisc:
    """ Calculator tests that don't belong to other test classes """
    def test_calculator_init(self, fw_impl_mock, fw_info_mock):
        n1 = build_node('n1', qcs=[build_qc(a_enable=False)], output_shape=(None, 5, 10))
        n2 = build_node('n2', output_shape=(None, 2, 111, 3),
                        canonical_weights={'foo': np.zeros((3, 14)),
                                           'bar': np.zeros((15, 9, 2, 6)),
                                           2: np.zeros((2, 71))},
                        qcs=[build_qc(w_attr={'foo': (8, False), 'bar': (8, True)}, pos_attr=(8, True, [2]))])
        n3 = build_node('n3', qcs=[build_qc(4)], output_shape=(None, 17))
        graph = Graph('g', input_nodes=[n1], nodes=[n2], output_nodes=[n3],
                      edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0)])

        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        assert ru_calc._act_tensors_size == {'n1': 50, 'n2': 666, 'n3': 17}
        assert ru_calc._params_cnt == {'n2': {full_attr_name('foo'): 42,
                                              full_attr_name('bar'): 1620,
                                              2: 142}}

    def test_topo_sort(self, graph_mock, fw_impl_mock, fw_info_mock):
        n1, n2, n3, n4, n5 = [build_node(f'n{i}') for i in range(5)]
        graph_mock.get_topo_sorted_nodes.return_value = [n3, n4, n2, n5, n1]
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)

        assert ru_calc._topo_sort([]) == []
        assert ru_calc._topo_sort([n5, n4, n3, n2, n1]) == [n3, n4, n2, n5, n1]
        assert ru_calc._topo_sort([n1, n3, n5]) == [n3, n5, n1]
        n6 = build_node('n6')
        with pytest.raises(ValueError, match=fr'Could not topo-sort, nodes \[{n6}\] do not match the graph nodes'):
            ru_calc._topo_sort([n1, n2, n6])


class BOPNode:
    pass


class TestBOPSAndVirtualGraph:
    def test_compute_regular_node_bops(self, fw_impl_mock, fw_info_mock):
        fw_info_mock.get_kernel_op_attributes = lambda node_type: ['foo'] if node_type == BOPNode else []
        fw_impl_mock.get_node_mac_operations = lambda n, fw_info: 42 if n.name == 'n2' else 0

        # a quantized, w quantized
        graph, n1, n2, _ = self._build_regular_node_graph(True, True)
        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)

        assert ru_calc.compute_node_bops(n2, TIC.AnyQuantized, BM.Float) == 42*32*32
        assert ru_calc.compute_node_bops(n2, TIC.AnyQuantized, BM.Q8Bit) == 42*8*8
        assert ru_calc.compute_node_bops(n2, TIC.AnyQuantized, BM.QMinBit) == 42*7*2
        assert ru_calc.compute_node_bops(n2, TIC.AnyQuantized, BM.QMaxBit) == 42*16*4

        assert ru_calc.compute_node_bops(n2, TIC.Any, BM.QMaxBit) == 42*16*4
        assert ru_calc.compute_node_bops(n2, None, BM.QMaxBit) == 42*16*4

        assert ru_calc.compute_node_bops(n1, None, BM.Float) == 0

        # a quantized, w not quantized - should be selected by AnyQuantized
        graph, _, n2, _ = self._build_regular_node_graph(enable_aq=True, enable_wq=False)
        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        assert ru_calc.compute_node_bops(n2, TIC.AnyQuantized, BM.QMinBit) == 42 * 7 * 32

        # a not quantized, w quantized - should be selected by AnyQuantized
        graph, _, n2, _ = self._build_regular_node_graph(enable_aq=False, enable_wq=True)
        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        assert ru_calc.compute_node_bops(n2, TIC.AnyQuantized, BM.QMinBit) == 42 * 32 * 2

        # a not quantized, w not quantized - should not be selected by AnyQuantized
        graph, _, n2, _ = self._build_regular_node_graph(enable_aq=False, enable_wq=False)
        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        assert ru_calc.compute_node_bops(n2, TIC.AnyQuantized, BM.QMinBit) == 0
        assert ru_calc.compute_node_bops(n2, TIC.Any, BM.QMinBit) == 42 * 32 * 32

    def test_compute_regular_node_bops_custom(self, fw_impl_mock, fw_info_mock):
        fw_info_mock.get_kernel_op_attributes = lambda node_type: ['foo'] if node_type == BOPNode else []
        fw_impl_mock.get_node_mac_operations = lambda n, fw_info: 42 if n.name == 'n2' else 0

        custom_qc1 = build_qc(3)
        custom_qc2 = build_qc(5, w_attr={'foo': (6, True)})
        a_cfg = {'n1': custom_qc1.activation_quantization_cfg,
                 'n2': custom_qc2.activation_quantization_cfg,
                 'n3': Mock()}

        graph, _, n2, _ = self._build_regular_node_graph(enable_aq=False, enable_wq=False)
        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        assert ru_calc.compute_node_bops(n2, TIC.Any, BM.QCustom,
                                         act_qcs=a_cfg, w_qc=custom_qc2.weights_quantization_cfg) == 42 * 3 * 6
        assert ru_calc.compute_node_bops(n2, None, BM.QCustom,
                                         act_qcs=a_cfg, w_qc=custom_qc2.weights_quantization_cfg) == 42 * 3 * 6
        assert ru_calc.compute_node_bops(n2, TIC.AnyQuantized, BM.QCustom,
                                         act_qcs=a_cfg, w_qc=custom_qc2.weights_quantization_cfg) == 0

    def test_compute_node_bops_default_qc(self, fw_impl_mock, fw_info_mock):
        fw_info_mock.get_kernel_op_attributes = lambda node_type: ['foo'] if node_type == BOPNode else []
        fw_impl_mock.get_node_mac_operations = lambda n, fw_info: 42 if n.name == 'n2' else 0

        n1 = build_node('n1', qcs=[build_qc(7)], output_shape=(None, 5, 10))
        n2 = build_node('n2', layer_class=BOPNode, output_shape=(None, 2, 111, 3),
                        canonical_weights={'foo': np.zeros((3, 14))},
                        qcs=[build_qc(w_attr={'foo': (6, True)})])
        n3 = build_node('n3', qcs=[build_qc()], output_shape=(None, 17))
        graph = Graph('g', input_nodes=[n1], nodes=[n2], output_nodes=[n3],
                      edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0)])
        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        assert ru_calc.compute_node_bops(n2, TIC.Any, BM.QCustom) == 42 * 7 * 6

    def test_compute_virtual_aw_node_bops_fully_quantized(self, fw_impl_mock, fw_info_mock):
        # all quantized
        g, _, a1w2, a2, a2w3, w3, a3 = self._build_virtual_node_graph(fw_impl_mock, fw_info_mock,
                                                                  quantize_a1=True, quantize_w1=True,
                                                                  quantize_a2=True, quantize_w2=True)
        ru_calc = ResourceUtilizationCalculator(g, fw_impl_mock, fw_info_mock)

        assert ru_calc.compute_node_bops(a1w2, TIC.AnyQuantized, BM.Float) == 42 * 32 * 32
        assert ru_calc.compute_node_bops(a1w2, TIC.AnyQuantized, BM.Q8Bit) == 42 * 8 * 8
        assert ru_calc.compute_node_bops(a1w2, TIC.AnyQuantized, BM.QMinBit) == 42 * 4 * 6
        assert ru_calc.compute_node_bops(a1w2, TIC.AnyQuantized, BM.QMaxBit) == 42 * 16 * 16

        assert ru_calc.compute_node_bops(a2w3, TIC.AnyQuantized, BM.Float) == 142 * 32 * 32
        assert ru_calc.compute_node_bops(a2w3, TIC.AnyQuantized, BM.Q8Bit) == 142 * 8 * 8
        assert ru_calc.compute_node_bops(a2w3, TIC.AnyQuantized, BM.QMinBit) == 142 * 2 * 2
        assert ru_calc.compute_node_bops(a2w3, TIC.AnyQuantized, BM.QMaxBit) == 142 * 5 * 6

        assert ru_calc.compute_node_bops(a2, TIC.AnyQuantized, BM.Float) == 0

    def test_compute_virtual_aw_node_bops_half_quantized(self, fw_impl_mock, fw_info_mock):
        g, _, a1w2, a2, a2w3, w3, a3 = self._build_virtual_node_graph(fw_impl_mock, fw_info_mock,
                                                                   quantize_a1=True, quantize_w1=False,
                                                                   quantize_a2=False, quantize_w2=True)
        ru_calc = ResourceUtilizationCalculator(g, fw_impl_mock, fw_info_mock)
        assert ru_calc.compute_node_bops(a1w2, TIC.AnyQuantized, BM.QMaxBit) == 42 * 16 * 32
        assert ru_calc.compute_node_bops(a2w3, TIC.AnyQuantized, BM.QMaxBit) == 142 * 32 * 6

    def test_compute_virtual_aw_node_bops_unquantized(self, fw_impl_mock, fw_info_mock):
        g, _, a1w2, a2, a2w3, w3, a3 = self._build_virtual_node_graph(fw_impl_mock, fw_info_mock,
                                                                   quantize_a1=False, quantize_w1=False,
                                                                   quantize_a2=False, quantize_w2=False)
        ru_calc = ResourceUtilizationCalculator(g, fw_impl_mock, fw_info_mock)
        assert ru_calc.compute_node_bops(a1w2, TIC.AnyQuantized, BM.QMaxBit) == 0
        assert ru_calc.compute_node_bops(a2w3, TIC.AnyQuantized, BM.QMaxBit) == 0

        assert ru_calc.compute_node_bops(a1w2, TIC.Any, BM.QMaxBit) == 42 * 32 * 32
        assert ru_calc.compute_node_bops(a2w3, TIC.Any, BM.QMaxBit) == 142 * 32 * 32

    def test_compute_virtual_aw_node_bops_custom(self, fw_impl_mock, fw_info_mock):
        g, _, a1w2, a2, a2w3, w3, a3 = self._build_virtual_node_graph(fw_impl_mock, fw_info_mock,
                                                                   quantize_a1=False, quantize_w1=False,
                                                                   quantize_a2=True, quantize_w2=True)
        custom_qc_a1w2 = build_qc(5, w_attr={'foo': (6, True)})
        custom_qc_a2w3 = build_qc(7, w_attr={'bar': (3, True)})
        a_cfg = {a1w2.name: custom_qc_a1w2.activation_quantization_cfg,
                 a2w3.name: custom_qc_a2w3.activation_quantization_cfg}
        w_a1w2 = custom_qc_a1w2.weights_quantization_cfg
        w_a2w3 = custom_qc_a2w3.weights_quantization_cfg

        ru_calc = ResourceUtilizationCalculator(g, fw_impl_mock, fw_info_mock)
        assert ru_calc.compute_node_bops(a1w2, TIC.Any, BM.QCustom, act_qcs=a_cfg, w_qc=w_a1w2) == 42 * 5 * 6
        assert ru_calc.compute_node_bops(a1w2, TIC.AnyQuantized, BM.QCustom, act_qcs=a_cfg, w_qc=w_a1w2) == 0

        assert ru_calc.compute_node_bops(a2w3, TIC.AnyQuantized, BM.QCustom, act_qcs=a_cfg, w_qc=w_a2w3) == 142 * 7 * 3

    @pytest.mark.parametrize('bm', set(BitwidthMode) - {BM.QCustom})
    def test_node_bops_unexpected_custom_qcs(self, graph_mock, fw_impl_mock, fw_info_mock, bm):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match=ru_calc.unexpected_qc_error):
            ru_calc.compute_node_bops(Mock(), TIC.Any, bm, act_qcs=Mock())

        with pytest.raises(ValueError, match=ru_calc.unexpected_qc_error):
            ru_calc.compute_node_bops(Mock(), TIC.Any, bm, w_qc=Mock())

    def test_node_bops_invalid_custom_qcs(self, graph_mock, fw_impl_mock, fw_info_mock):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match=ru_calc.unexpected_qc_nodes_error):
            ru_calc.compute_node_bops(Mock(), TIC.Any, BM.QCustom, act_qcs={'unknown': Mock()})

    @pytest.mark.parametrize('target_criterion', set(TIC) - {TIC.Any, TIC.AnyQuantized})
    def test_node_bops_invalid_criterion(self, graph_mock, fw_impl_mock, fw_info_mock, target_criterion):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match='BOPS computation is supported only for Any and AnyQuantized targets.'):
            ru_calc.compute_node_bops(Mock(), target_criterion, BM.Float)

    def test_compute_bops(self, fw_impl_mock, fw_info_mock,):
        class BOPNode2:
            pass

        n1 = build_node('n1', qcs=[build_qc(16, True)], output_shape=(None, 5, 10))
        n2 = build_node('n2', layer_class=BOPNode, output_shape=(None, 2, 111, 3),
                        canonical_weights={'mp': np.zeros((3, 14))},
                        qcs=[
                            build_qc(7, w_attr={'mp': (4, True)}),
                            build_qc(7, w_attr={'mp': (2, True)})
                        ])

        n3 = build_node('n3', layer_class=BOPNode2, output_shape=(None, 17),
                        canonical_weights={'sp': np.zeros((3, 14, 15))},
                        qcs=[build_qc(w_attr={'sp': (5, True)})])
        graph = Graph('g', input_nodes=[n1], nodes=[n2], output_nodes=[n3],
                      edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0)])

        fw_info_mock.get_kernel_op_attributes = lambda node_type: {BOPNode: ['mp'], BOPNode2: ['sp']}.get(node_type, [])
        fw_impl_mock.get_node_mac_operations = lambda n, fw_info: {'n2': 42, 'n3': 630}.get(n.name, 0)
        topo = graph.get_topo_sorted_nodes()
        graph.get_topo_sorted_nodes = Mock(return_value=topo[::-1])

        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        total, detailed = ru_calc.compute_bops(TIC.Any, BM.QMaxBit)
        assert list(detailed.keys()) == ['n3', 'n2']
        assert detailed == {'n2': 42 * 16 * 4,
                            'n3': 630 * 7 * 5}
        assert total == 42 * 64 + 630 * 35

        custom_qc = build_qc(9, w_attr={'mp': (7, True)})
        a_cfg = {'n1': build_qc(4, False).activation_quantization_cfg}
        w_cfg = {'n2': custom_qc.weights_quantization_cfg}
        total, detailed = ru_calc.compute_bops(TIC.Any, BM.QCustom, a_cfg, w_cfg)
        assert detailed == {'n2': 42 * 32 * 7,
                            'n3': 630 * 7 * 5}

    def test_compute_virtual_graph_resources(self, fw_impl_mock, fw_info_mock):
        g, _, a1w2, a2, a2w3, w3, a3 = self._build_virtual_node_graph(fw_impl_mock, fw_info_mock, True, True, True, True)
        ru_calc = ResourceUtilizationCalculator(g, fw_impl_mock, fw_info_mock)
        ru, detailed = ru_calc.compute_resource_utilization(TIC.Any, BM.QMaxBit, return_detailed=True)
        assert (sorted(list(detailed[RUTarget.ACTIVATION].values())) ==
                sorted([24, 24 + 50*2, 50*2+88*5/8, 88*5/8 + 28*7/8, 28*7/8])), detailed[RUTarget.ACTIVATION]
        assert detailed[RUTarget.WEIGHTS] == {a1w2.name: 42*2, a2w3.name: 142*6/8}
        assert detailed[RUTarget.BOPS] == {a1w2.name: 42*16*16, a2w3.name: 142*5*6}
        assert ru == ResourceUtilization(weights_memory=84 + 142*6/8,
                                         activation_memory=155,
                                         total_memory=155 + (84 + 142*6/8),
                                         bops=42*256+142*30)

    def test_virtual_graph_with_virtual_weight(self, fw_impl_mock, fw_info_mock):
        # virtual weight node wasn't merged into virtual composed node
        _, n_in, a1w2, a2, a2w3, w3, a3 = self._build_virtual_node_graph(fw_impl_mock, fw_info_mock, True, True, True, True)
        g = Graph('g', nodes=[w3], input_nodes=[n_in], output_nodes=[a3],
                  edge_list=[Edge(n_in, w3, 0, 0), Edge(w3, a3, 0, 0)])
        ru_calc = ResourceUtilizationCalculator(g, fw_impl_mock, fw_info_mock)
        ru, detailed = ru_calc.compute_resource_utilization(TIC.Any, BM.QMaxBit, return_detailed=True)
        assert list(detailed[RUTarget.WEIGHTS].values()) == [142 * 6 / 8]
        assert detailed[RUTarget.BOPS] == {}
        # the extra cut that is created by virtual weight node. The rest of the cuts must be correct.
        wa_cut = 2*28*7/8
        assert sorted(list(detailed[RUTarget.ACTIVATION].values())) == sorted([24, 24+28*7/8, 28*7/8, wa_cut])
        assert ru == ResourceUtilization(weights_memory=142 * 6 / 8,
                                         activation_memory=wa_cut,
                                         total_memory=wa_cut + (142 * 6 / 8),
                                         bops=0)

    def test_multi_output_input_activation(self, fw_impl_mock, fw_info_mock):
        """ No bops should be calculated for weight node if its input activation has multiple outputs. """
        n_in = build_node('in', qcs=[build_qc()], output_shape=(None, 2, 3, 4))
        n2 = build_node('n2', layer_class=BOPNode, output_shape=(None, 2, 44),
                        canonical_weights={'foo': np.zeros((3, 14))},
                        qcs=[
                            build_qc(2, w_attr={'foo': (16, True)}),
                            build_qc(3, w_attr={'foo': (10, True)}),
                            build_qc(4, w_attr={'foo': (7, True)}),
                            build_qc(5, w_attr={'foo': (6, True)}),
                        ])
        n_out = build_node('out', qcs=[build_qc()], output_shape=(None, 27))
        g = Graph('g', input_nodes=[n_in], nodes=[n2], output_nodes=[n_out],
                  edge_list=[Edge(n_in, n2, 0, 0), Edge(n_in, n_out, 0, 0)])

        def get_kernel_attr(node_type):
            return {BOPNode: ['foo']}.get(node_type) or []
        fw_info_mock.get_kernel_op_attributes = get_kernel_attr
        fw_impl_mock.get_node_mac_operations = lambda n, fw_info: {n2: 42}.get(n, 0)

        ru_calc = ResourceUtilizationCalculator(g, fw_impl_mock, fw_info_mock)
        assert ru_calc.compute_bops(TIC.Any, BM.Float) == (0, {})

    def _build_regular_node_graph(self, enable_aq, enable_wq):
        n1 = build_node('n1', qcs=[build_qc(16, enable_aq), build_qc(7, enable_aq)], output_shape=(None, 5, 10))
        n2 = build_node('n2', layer_class=BOPNode, output_shape=(None, 2, 111, 3),
                        canonical_weights={'foo': np.zeros((3, 14)),
                                           'bar': np.zeros((15, 9, 2, 6))},
                        qcs=[
                            build_qc(w_attr={'foo': (4, enable_wq), 'bar': (8, True)}),
                            build_qc(w_attr={'foo': (2, enable_wq), 'bar': (8, True)})
                        ])
        n3 = build_node('n3', qcs=[build_qc()], output_shape=(None, 17))
        graph = Graph('g', input_nodes=[n1], nodes=[n2], output_nodes=[n3],
                      edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0)])
        return graph, n1, n2, n3

    def _build_virtual_node_graph(self, fw_impl_mock, fw_info_mock, quantize_w1, quantize_w2, quantize_a1, quantize_a2):

        class BOPNode2:
            pass

        class ActType:
            pass

        # define original nodes
        n_in = build_node('in', qcs=[build_qc()], output_shape=(None, 2, 3, 4))
        n1 = build_node('n1', qcs=[build_qc(16, quantize_a1),
                                   build_qc(4, quantize_a1)], output_shape=(None, 5, 10))
        n2 = build_node('n2', layer_class=BOPNode, output_shape=(None, 2, 44),
                        canonical_weights={'foo': np.zeros((3, 14))},
                        qcs=[
                            build_qc(2, quantize_a2, w_attr={'foo': (16, quantize_w1)}),
                            build_qc(3, quantize_a2, w_attr={'foo': (10, quantize_w1)}),
                            build_qc(4, quantize_a2, w_attr={'foo': (7, quantize_w1)}),
                            build_qc(5, quantize_a2, w_attr={'foo': (6, quantize_w1)}),
                        ])
        n3 = build_node('n3', layer_class=BOPNode2, output_shape=(None, 28),
                        canonical_weights={'bar': np.zeros((2, 71))},
                        qcs=[
                            build_qc(4, w_attr={'bar': (6, quantize_w2)}),
                            build_qc(5, w_attr={'bar': (5, quantize_w2)}),
                            build_qc(6, w_attr={'bar': (4, quantize_w2)}),
                            build_qc(7, w_attr={'bar': (2, quantize_w2)})
                        ])

        def get_kernel_attr(node_type):
            return {BOPNode: ['foo'], BOPNode2: ['bar']}.get(node_type, [])
        fw_info_mock.get_kernel_op_attributes = get_kernel_attr
        fw_impl_mock.get_node_mac_operations = lambda n, fw_info: {n2: 42, n3: 142}.get(n, 0)

        # virtual aw node made of original nodes
        a1w2 = VirtualActivationWeightsNode(act_node=n1, weights_node=n2, fw_info=fw_info_mock)
        a2 = VirtualSplitActivationNode(n2, ActType, {})
        w3 = VirtualSplitWeightsNode(n3, 'bar')
        # virtual aw node made of virtual split a, w nodes
        a2w3 = VirtualActivationWeightsNode(act_node=a2, weights_node=w3, fw_info=fw_info_mock)
        a3 = VirtualSplitActivationNode(n3, ActType, {})
        g = Graph('g', nodes=[a1w2, a2w3, a3], input_nodes=[n_in], output_nodes=[w3],
                  edge_list=[Edge(n_in, a1w2, 0, 0), Edge(a1w2, a2w3, 0, 0), Edge(a2w3, a3, 0, 0)])
        return g, n_in, a1w2, a2, a2w3, w3, a3
