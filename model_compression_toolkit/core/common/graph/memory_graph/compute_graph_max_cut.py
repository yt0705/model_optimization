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
from collections import namedtuple
from typing import Tuple, List

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import OPERATORS_SCHEDULING, MAX_CUT, CUTS, FUSED_NODES_MAPPING
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.memory_graph.cut import Cut
from model_compression_toolkit.core.common.graph.memory_graph.max_cut_astar import MaxCutAstar
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph

SchedulerInfo = namedtuple('SchedulerInfo', [OPERATORS_SCHEDULING, MAX_CUT, CUTS, FUSED_NODES_MAPPING])


def compute_graph_max_cut(memory_graph: MemoryGraph,
                          n_iter: int = 50,
                          astar_n_iter: int = 1000,
                          eps: float = 1e-2) -> Tuple[List[BaseNode], float, List[Cut]]:
    """
    A wrapper function to compute max cut and schedule for a given model.
    It runs iterations of AStar search on the given memory graph with a dynamically updating estimation bound.

    Args:
        memory_graph: A MemoryGraph object to run the search on.
        n_iter: Limit on the number of AStar searches.
        astar_n_iter: Limit on the number of expansion iterations in a single AStar search.
        eps: Small value for defining a sufficient gap around an optimal solution in which the search would finish.

    Returns: A solution of the AStar search (schedule, max cut cost, cuts route).

    """
    max_cut_astar = MaxCutAstar(memory_graph=memory_graph)
    last_result = (None, 0, None)
    l_bound = memory_graph.memory_lbound_single_op
    u_bound = 2 * sum([t.total_size for t in memory_graph.b_nodes]) - l_bound
    it = 0
    while it < n_iter:
        estimate = (u_bound + l_bound) / 2
        # Add a timeout of 5 minutes to the solver from the 2nd iteration.
        try:
            schedule, max_cut_size, cuts = max_cut_astar.solve(estimate=estimate, iter_limit=astar_n_iter,
                                                               time_limit=None if it == 0 else 300)
        except TimeoutError:  # pragma: no cover
            # TODO: add test for this.
            if last_result[0] is None:
                Logger.critical(f"Max-cut solver stopped on timeout in iteration {it} before finding a solution.")
            else:
                Logger.warning(f"Max-cut solver stopped on timeout in iteration {it}.")
                return last_result

        if schedule is None:
            l_bound = estimate
        else:
            u_bound = min(estimate, max_cut_size)
            last_result = (schedule, max_cut_size, cuts)

            if l_bound * (1 + eps) >= u_bound:
                return last_result

        it += 1

    return last_result
