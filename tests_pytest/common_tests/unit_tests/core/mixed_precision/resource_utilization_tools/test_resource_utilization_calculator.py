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
import random
from types import MethodType
from unittest.mock import Mock

import numpy as np
import pytest
from model_compression_toolkit.core.common.graph.base_graph import OutTensor

from model_compression_toolkit.constants import FLOAT_BITWIDTH, FUSED_LAYER_PATTERN, FUSED_OP_QUANT_CONFIG
from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfo
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
from tests_pytest._test_util.graph_builder_utils import build_node, full_attr_name, build_nbits_qc as build_qc

BM = BitwidthMode
TIC = TargetInclusionCriterion


@pytest.fixture
def graph_mock():
    """ Basic Graph mock with basic retrieve_preserved_quantization_node operation for handling non
    quantization preserving nodes. """
    return Mock(spec_set=Graph, nodes=[], retrieve_preserved_quantization_node=lambda x: x)


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
        ret = ru_calc.compute_resource_utilization(TIC.Any, BM.QDefaultSP, ru_targets=[RUTarget.WEIGHTS],
                                                   return_detailed=detailed)
        ru_calc.compute_weights_utilization.assert_called_once_with(TIC.Any, BM.QDefaultSP, None)
        ru_calc.compute_activations_utilization.assert_not_called()
        ru_calc.compute_bops.assert_not_called()
        self._validate(ret, detailed, ResourceUtilization(weights_memory=21))

    @pytest.mark.parametrize('detailed', [True, False])
    def test_compute_ru_act(self, detailed):
        ru_calc = self.ru_calc
        ret = self.ru_calc.compute_resource_utilization(TIC.Any, BM.QDefaultSP, ru_targets=[RUTarget.ACTIVATION],
                                                        return_detailed=detailed)

        ru_calc.compute_activations_utilization.assert_called_once_with(TIC.Any, BM.QDefaultSP, None)
        ru_calc.compute_weights_utilization.assert_not_called()
        ru_calc.compute_bops.assert_not_called()
        self._validate(ret, detailed, ResourceUtilization(activation_memory=671))

    @pytest.mark.parametrize('detailed', [True, False])
    def test_compute_ru_total(self, detailed):
        ru_calc = self.ru_calc
        ret = ru_calc.compute_resource_utilization(TIC.Any, BM.QDefaultSP, ru_targets=[RUTarget.TOTAL],
                                                   return_detailed=detailed)

        ru_calc.compute_activations_utilization.assert_called_once_with(TIC.Any, BM.QDefaultSP, None)
        ru_calc.compute_weights_utilization.assert_called_once_with(TIC.Any, BM.QDefaultSP, None)
        ru_calc.compute_bops.assert_not_called()
        self._validate(ret, detailed, ResourceUtilization(total_memory=671+21))

    @pytest.mark.parametrize('detailed', [True, False])
    def test_compute_ru_bops(self, detailed):
        ru_calc = self.ru_calc
        ret = ru_calc.compute_resource_utilization(TIC.AnyQuantized, BM.QDefaultSP, ru_targets=[RUTarget.BOPS],
                                                   return_detailed=detailed)

        ru_calc.compute_bops.assert_called_once_with(TIC.AnyQuantized, BM.QDefaultSP, act_qcs=None, w_qcs=None)
        ru_calc.compute_activations_utilization.assert_not_called()
        ru_calc.compute_weights_utilization.assert_not_called()
        self._validate(ret, detailed, ResourceUtilization(bops=42*8*4))

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

    def test_get_a_nbits_configurable_quantization_preserving(self, graph_mock, fw_impl_mock, fw_info_mock):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        node = build_node(qcs=[build_qc(7, a_enable=False, q_preserving=True),
                               build_qc(4, a_enable=False, q_preserving=True),
                               build_qc(2, a_enable=False, q_preserving=True)])
        anode = build_node(qcs=[build_qc(17), build_qc(4), build_qc(1)])
        graph_mock.retrieve_preserved_quantization_node = lambda x: anode
        assert ru_calc._get_activation_nbits(node, BM.Float, None) == FLOAT_BITWIDTH
        assert ru_calc._get_activation_nbits(node, BM.QMinBit, None) == 1
        assert ru_calc._get_activation_nbits(node, BM.QMaxBit, None) == 17

    @pytest.mark.parametrize('node', [
        build_node(qcs=[build_qc(42)]),
        build_node(qcs=[build_qc(42, w_attr={'foo': (4, True)}), build_qc(42, w_attr={'foo': (2, False)})])
    ])
    def test_get_a_nbits_nonconfigurable(self, graph_mock, fw_impl_mock, fw_info_mock, node):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        for bm in set(BitwidthMode) - {BM.Float}:
            assert ru_calc._get_activation_nbits(node, bm, None) == 42
        assert ru_calc._get_activation_nbits(node, BM.Float, None) == FLOAT_BITWIDTH

    @pytest.mark.parametrize('node, qc, exp_nbit', [
        (build_node(qcs=[build_qc(4)]), build_qc(17), 17),
        (build_node(qcs=[build_qc(4)]), build_qc(17, False), 32),
        (build_node(qcs=[build_qc(4, False)]), build_qc(17, True), 17)
    ])
    def test_get_a_nbits_custom(self, graph_mock, fw_impl_mock, fw_info_mock, node, qc, exp_nbit):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        assert ru_calc._get_activation_nbits(node, BM.QCustom, qc.activation_quantization_cfg) == exp_nbit

    @pytest.mark.parametrize('anode, node, qc, exp_nbit', [
        (None, build_node(qcs=[build_qc(4)]), build_qc(17, False, q_preserving=True), 32),
        (build_node(qcs=[build_qc(3)]), build_node(qcs=[build_qc(4, False, q_preserving=True)]), None, 3)
    ])
    def test_get_a_nbits_custom_quantization_preserving(self, graph_mock, fw_impl_mock, fw_info_mock, anode, node, qc, exp_nbit):
        graph_mock.retrieve_preserved_quantization_node = lambda x: anode
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        acs = None if qc is None else qc.activation_quantization_cfg
        assert ru_calc._get_activation_nbits(node, BM.QCustom, acs) == exp_nbit

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
        qp = build_node('qp', qcs=[build_qc(4, False, q_preserving=True)])

        graph_mock.nodes = [sp1, sp2, sp3, mp, noq, qp]
        graph_mock.fusing_info = FusingInfo(fusing_data={'FusedNode_n1_n2': (sp1, sp2)})
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)

        assert len(TIC) == 5, 'enum changed, update tests'
        assert ru_calc._get_target_activation_nodes(TIC.QConfigurable, include_reused=True) == [sp1, mp]
        assert ru_calc._get_target_activation_nodes(TIC.QConfigurable, include_reused=False) == [sp1]

        assert ru_calc._get_target_activation_nodes(TIC.QNonConfigurable, include_reused=True) == [sp2, sp3]
        assert ru_calc._get_target_activation_nodes(TIC.QNonConfigurable, include_reused=False) == [sp2]

        assert ru_calc._get_target_activation_nodes(TIC.AnyQuantized, include_reused=True) == [sp1, sp2, sp3, mp, qp]
        assert ru_calc._get_target_activation_nodes(TIC.AnyQuantized, include_reused=False) == [sp1, sp2, qp]

        assert ru_calc._get_target_activation_nodes(TIC.AnyQuantizedNonFused, include_reused=True) == [sp2, sp3, mp, qp]
        assert ru_calc._get_target_activation_nodes(TIC.AnyQuantizedNonFused, include_reused=False) == [sp2, qp]

        assert ru_calc._get_target_activation_nodes(TIC.Any, include_reused=True) == [sp1, sp2, sp3, mp, noq, qp]
        assert ru_calc._get_target_activation_nodes(TIC.Any, include_reused=False) == [sp1, sp2, noq, qp]
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
        qp = build_node('qp', output_shape=(None, 15, 9), qcs=[build_qc(a_enable=False, q_preserving=True)])
        noq = build_node('noq', output_shape=(None, 15, 9), qcs=[build_qc(a_enable=False)])
        graph_mock.nodes = [mp_reuse, qp, noq]
        graph_mock.retrieve_preserved_quantization_node = lambda n: mp_reuse if n is qp else n

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
        # quantization preserving uses custom_qc.
        res = ru_calc.compute_node_activation_tensor_utilization(qp, TIC.AnyQuantized, BM.QCustom, custom_qc)
        assert res == Utilization(135, 270.)
        # not a target node
        res = ru_calc.compute_node_activation_tensor_utilization(noq, TIC.AnyQuantized, BM.QCustom, custom_qc)
        assert res == Utilization(0, 0)

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
        n1_qp = build_node('n1_qp', qcs=[build_qc(a_enable=False, q_preserving=True)],
                           input_shape=(None, 10, 20, 3), output_shape=(None, 10, 20, 3))
        n2 = build_node('n2', qcs=[build_qc()], input_shape=(None, 10, 20, 3), output_shape=(None, 5, 10))
        n3 = build_node('n3', qcs=[build_qc()], input_shape=(None, 5, 10), output_shape=(None, 5, 10))
        n4 = build_node('n4', qcs=[build_qc()], input_shape=(None, 5, 10, 32), output_shape=(None, 5, 10, 32))
        edges = [Edge(n1, n1_qp, 0, 0), Edge(n1_qp, n2, 0, 0),
                 Edge(n2, n3, 0, 0), Edge(n3, n4, 0, 0)]
        graph = Graph('g', input_nodes=[n1], nodes=[n1_qp, n2, n3], output_nodes=[n4], edge_list=edges)
        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)
        # wrap the real implementation
        maxcut_spy = mocker.patch('model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.'
                                  'resource_utilization_calculator.compute_graph_max_cut', wraps=compute_graph_max_cut)

        # trigger cuts cache computation
        cuts_cache = ru_calc.cuts

        # verify the cache
        assert len(cuts_cache) == 6
        assert all(isinstance(k, Cut) for k in cuts_cache.keys())
        # for each cut we save a list of its nodes
        cuts_nodes = {tuple(sorted(n.name for n in nodes)) for nodes in cuts_cache.values()}
        assert cuts_nodes == {('n1',), ('n4',), ('n1', 'n1_qp'), ('n1_qp', 'n2'), ('n2', 'n3'), ('n3', 'n4')}

        # verify cuts computation only happens the first time
        cuts_cache2 = ru_calc.cuts
        maxcut_spy.assert_called_once()
        assert cuts_cache2 == cuts_cache

        # map from node names to cuts to retrieve the cuts
        nodes_to_cuts = {tuple(sorted(elem.node_name for elem in cut.mem_elements.elements)): cut
                         for cut in cuts_cache.keys()}
        cut1 = nodes_to_cuts[('n1',)]
        cut11 = nodes_to_cuts[('n1', 'n1_qp')]
        cut12 = nodes_to_cuts[('n1_qp', 'n2')]
        cut23 = nodes_to_cuts[('n2', 'n3')]
        cut34 = nodes_to_cuts[('n3', 'n4')]
        cut4 = nodes_to_cuts[('n4',)]

        # compute utilization to check everything works together with real maxcut
        total, per_cut, per_cut_per_node = ru_calc.compute_activation_utilization_by_cut(target_criterion=TIC.AnyQuantized,
                                                                                         bitwidth_mode=BM.QDefaultSP)

        assert per_cut_per_node == {cut1: {'n1': Utilization(10 * 20 * 3, 600)},
                                    cut11: {'n1': Utilization(10 * 20 * 3, 600), 'n1_qp': Utilization(10 * 20 * 3, 600)},
                                    cut12: {'n1_qp': Utilization(10 * 20 * 3, 600),
                                            'n2': Utilization(5 * 10, 50)},
                                    cut23: {'n2': Utilization(5*10, 50),
                                            'n3': Utilization(5*10, 50)},
                                    cut34: {'n3': Utilization(5*10, 50),
                                            'n4': Utilization(5*10*32, 1600)},
                                    cut4: {'n4': Utilization(5 * 10 * 32, 1600)}}
        assert per_cut == {
            nodes_to_cuts[('n1',)]: Utilization(600, 600),
            nodes_to_cuts[('n1', 'n1_qp')]: Utilization(1200, 1200),
            nodes_to_cuts[('n1_qp', 'n2')]: Utilization(650, 650),
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
        qp = build_node('qp', qcs=[build_qc(2, a_enable=False, q_preserving=True),
                                   build_qc(4, a_enable=False, q_preserving=True)], output_shape=(None, 150))

        nodes = [mp_reuse, mp, noq, sp, mp2, qp]
        graph_mock.nodes = nodes
        # use the Graph original method (need to bind it to graph_mock instance)
        graph_mock.find_node_by_name = MethodType(Graph.find_node_by_name, graph_mock)
        graph_mock.retrieve_preserved_quantization_node = lambda x: mp2 if x.name == 'qp' else x

        graph_mock.fusing_info = FusingInfo(fusing_data={'FusedNode_sp_mp':(sp, mp)})

        # we should not use total size, setting it to bad number
        cut_elems1 = MemoryElements(elements={ActivationMemoryTensor(mp_reuse.output_shape, 'mp_reuse', 0)}, total_size=-1)
        cut_elems2 = MemoryElements(elements={ActivationMemoryTensor(mp_reuse.output_shape, 'mp_reuse', 0),
                                              ActivationMemoryTensor(mp.output_shape, 'mp', 0),
                                              ActivationMemoryTensor(noq.output_shape, 'noq', 0),
                                              ActivationMemoryTensor(sp.output_shape, 'sp', 0)}, total_size=-1)
        cut_elems3 = MemoryElements(elements={ActivationMemoryTensor(sp.output_shape, 'sp', 0),
                                              ActivationMemoryTensor(noq.output_shape, 'noq', 0)}, total_size=-1)
        cut_elems4 = MemoryElements(elements={ActivationMemoryTensor(mp2.output_shape, 'mp2', 0),
                                              ActivationMemoryTensor(mp2.output_shape, 'qp', 0)}, total_size=-1)

        cuts = [Cut([], set(), mem_elements=cut_elems)
                for cut_elems in [cut_elems1, cut_elems2, cut_elems3, cut_elems4]]
        mocker.patch.object(ResourceUtilizationCalculator, '_compute_cuts', Mock(return_value=cuts))
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        return ru_calc, cuts, nodes

    @pytest.mark.parametrize('seed', list(range(42, 52)))
    @pytest.mark.parametrize("disable_quantization", [True, False])
    def test_compute_cuts_random_fusion_valid_utilization(self, seed, disable_quantization, fw_impl_mock, fw_info_mock, mocker):
        random.seed(seed)

        num_nodes = random.randint(5, 8)
        node_names = [f"n{i}" for i in range(num_nodes)]
        nodes = []
        edges = []
        classes = []

        # Build nodes with matching input/output shapes
        input_shape = (None, random.randint(5, 10), random.randint(5, 10))
        for i, name in enumerate(node_names):
            output_shape = (None, random.randint(5, 10), random.randint(5, 10)) if i < num_nodes - 1 else input_shape
            layer_class = f"class_{i}"
            node = build_node(name, layer_class=layer_class, qcs=[build_qc()],
                              input_shape=input_shape, output_shape=output_shape)
            nodes.append(node)
            classes.append(layer_class)
            input_shape = output_shape

        for i in range(num_nodes - 1):
            edges.append(Edge(nodes[i], nodes[i + 1], 0, 0))

        # Generate random fused groups
        fused_patterns = []
        fused_data = {}
        i = 1
        while i < num_nodes - 1:
            if random.random() < 0.5:
                fuse_len = random.choice([2, 3])
                if i + fuse_len <= num_nodes:
                    fused = tuple(nodes[j] for j in range(i, i + fuse_len))
                    fused_name = f"FusedNode_{'_'.join(n.name for n in fused)}"
                    fused_pattern = {FUSED_LAYER_PATTERN: [n.layer_class for n in fused], FUSED_OP_QUANT_CONFIG: None}
                    fused_patterns.append(fused_pattern)
                    fused_data[fused_name] = fused
                    i += fuse_len
                else:
                    break
            else:
                i += 1

        fusing_info = FusingInfo(fusing_patterns=fused_patterns, fusing_data=fused_data)
        graph = Graph("g", input_nodes=[nodes[0]], nodes=nodes,
                      output_nodes=[OutTensor(node=nodes[-1], node_out_index=0)], edge_list=edges)
        graph.fusing_info = fusing_info

        if disable_quantization:
            graph.disable_fused_nodes_activation_quantization()

        graph.find_node_by_name = MethodType(Graph.find_node_by_name, graph)

        ru_calc = ResourceUtilizationCalculator(graph, fw_impl_mock, fw_info_mock)

        # Patch max cut computation
        mocker.patch(
            'model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.'
            'resource_utilization_calculator.compute_graph_max_cut',
            wraps=compute_graph_max_cut
        )

        cuts = ru_calc.cuts

        # --- Assert cut structure ---
        assert all(isinstance(c, Cut) for c in cuts)
        for cut_nodes in cuts.values():
            assert all(isinstance(n.name, str) for n in cut_nodes)

        # --- Utilization ---
        total, per_cut, per_cut_per_node = ru_calc.compute_activation_utilization_by_cut(
            target_criterion=TIC.AnyQuantized, bitwidth_mode=BM.QDefaultSP
        )

        # Structure checks
        assert isinstance(per_cut, dict)
        assert isinstance(per_cut_per_node, dict)
        assert all(isinstance(k, Cut) for k in per_cut)
        assert all(isinstance(k, Cut) for k in per_cut_per_node)
        assert all(isinstance(v, Utilization) for v in per_cut.values())
        assert all(isinstance(vv, Utilization) for v in per_cut_per_node.values() for vv in v.values())

        # Value checks: per_cut == sum(per_cut_per_node)
        for cut, node_utils in per_cut_per_node.items():
            summed = sum((u for u in node_utils.values()), Utilization(0, 0))
            assert per_cut[cut] == summed

        # Total check
        assert total == max(u.bytes for u in per_cut.values())

        # Check the utilization bytes per node
        for cut, node_utils in per_cut_per_node.items():
            for node_name, util in node_utils.items():
                node = next((n for n in nodes if n.name == node_name), None)
                assert node is not None, f"Node {node_name} not found in graph"

                expected_volume = 1
                for dim in node.output_shape:
                    if dim is not None:
                        expected_volume *= dim

                assert util.bytes == expected_volume, (
                    f"Utilization mismatch for node '{node_name}': "
                    f"got {util.bytes}, expected {expected_volume} from shape {node.output_shape}"
                )

    def test_get_cut_target_nodes(self, prepare_compute_cuts):
        ru_calc, (cut1, cut2, cut3, cut4), (mp_reuse, mp, noq, sp, mp2, qp) = prepare_compute_cuts
        assert len(TIC) == 5
        sorted_res = lambda res: sorted(res, key=lambda n: n.name)
        assert sorted_res(ru_calc._get_cut_target_nodes(cut2, TIC.Any)) == [mp, mp_reuse, noq, sp]
        assert sorted_res(ru_calc._get_cut_target_nodes(cut2, TIC.AnyQuantized)) == [mp, mp_reuse, sp]
        assert sorted_res(ru_calc._get_cut_target_nodes(cut2, TIC.QConfigurable)) == [mp, mp_reuse]
        assert sorted_res(ru_calc._get_cut_target_nodes(cut2, TIC.QNonConfigurable)) == [sp]
        assert sorted_res(ru_calc._get_cut_target_nodes(cut2, TIC.AnyQuantizedNonFused)) == [mp, mp_reuse]

    def test_compute_act_utilization_by_cut(self, prepare_compute_cuts):
        ru_calc, (cut1, cut2, cut3, cut4), (mp_reuse, mp, noq, sp, mp2, qp) = prepare_compute_cuts

        ru_calc.compute_node_activation_tensor_utilization = Mock(wraps=ru_calc.compute_node_activation_tensor_utilization)
        ru_calc._get_cut_target_nodes = Mock(wraps=ru_calc._get_cut_target_nodes)

        qcs = {'mp_reuse': build_qc(7), 'mp': build_qc(10), 'noq': build_qc(4, True), 'mp2': build_qc(4, False)}
        qcs = {k: v.activation_quantization_cfg for k, v in qcs.items()}
        total, per_cut, per_cut_node = ru_calc.compute_activation_utilization_by_cut(TIC.AnyQuantized, BM.QCustom, qcs)

        cut_nodes_calls = ru_calc._get_cut_target_nodes.call_args_list
        assert len(cut_nodes_calls) == 4
        assert {call.args[0] for call in cut_nodes_calls} == {cut1, cut2, cut3, cut4}
        assert {call.args[1] for call in cut_nodes_calls } == {TIC.AnyQuantized}

        compute_tensor_calls = sorted(ru_calc.compute_node_activation_tensor_utilization.call_args_list,
                                      key=lambda call: call.args[0].name)
        assert len(compute_tensor_calls) == 7
        assert compute_tensor_calls[0].args == (mp, TIC.AnyQuantized, BM.QCustom, qcs['mp'])
        assert compute_tensor_calls[-1].args == (sp, TIC.AnyQuantized, BM.QCustom, None)

        assert len(per_cut_node) == 4
        assert per_cut_node[cut1] == {'mp_reuse': Utilization(24, 21.)}
        assert per_cut_node[cut2] == {'mp_reuse': Utilization(24, 21.),
                                      'mp': Utilization(50, 62.5),
                                      'sp': Utilization(200, 75.)}
        assert per_cut_node[cut3] == {'sp': Utilization(200, 75.)}
        assert per_cut_node[cut4] == {'mp2': Utilization(150, 600.), 'qp': Utilization(150, 600.)}

        assert per_cut == {cut1: Utilization(24, 21.),
                           cut2: Utilization(274, 158.5),
                           cut3: Utilization(200, 75.),
                           cut4: Utilization(300, 1200.),
                           }
        assert total == 1200.

    def test_compute_act_utilization_by_cut_no_cut_nodes(self, prepare_compute_cuts):
        ru_calc, (cut1, cut2, cut3, cut4), (mp_reuse, mp, noq, sp, mp2, qp) = prepare_compute_cuts

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

        # non-configurable attr with multiple qcs with same w precision
        for bm in set(BitwidthMode) - {BM.Float}:
            assert ru_calc._get_weight_nbits(node, 'sp', bm, w_qc=None) == 5
        assert ru_calc._get_weight_nbits(node, 'sp', BM.Float, w_qc=None) == FLOAT_BITWIDTH
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
        assert len(TIC) == 5, 'enum changed, update the test'
        assert ru_calc._get_target_weight_attrs(node, TIC.QConfigurable) == full_attr_name(['foo'])
        assert ru_calc._get_target_weight_attrs(node, TIC.QNonConfigurable) == full_attr_name(['bar', 1, 2])
        assert ru_calc._get_target_weight_attrs(node, TIC.AnyQuantized) == full_attr_name(['foo', 'bar', 1, 2])
        assert ru_calc._get_target_weight_attrs(node, TIC.Any) == full_attr_name(['foo', 'bar', 'baz', 1, 2])
        assert ru_calc._get_target_weight_attrs(node, TIC.AnyQuantizedNonFused) == full_attr_name(['foo', 'bar', 1, 2])

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
            build_node('n1', qcs=[build_qc()]),
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

    @pytest.mark.parametrize('target_criterion', set(TIC) - {TIC.Any, TIC.AnyQuantized, TIC.AnyQuantizedNonFused})
    def test_node_bops_invalid_criterion(self, graph_mock, fw_impl_mock, fw_info_mock, target_criterion):
        ru_calc = ResourceUtilizationCalculator(graph_mock, fw_impl_mock, fw_info_mock)
        with pytest.raises(ValueError, match='BOPS computation is supported only for Any, AnyQuantized and AnyQuantizedNonFused targets.'):
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
        assert ru_calc.compute_bops(TIC.Any, BM.QMaxBit) == (42*8*16, {'n2': 42*8*16})

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
