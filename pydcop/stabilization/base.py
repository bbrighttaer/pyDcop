import logging
from collections import namedtuple
from typing import List, Iterable

from pydcop.algorithms import ComputationDef
from pydcop.computations_graph.constraints_hypergraph import ConstraintLink
from pydcop.computations_graph.dynamic_graph import DynamicComputationNode
from pydcop.computations_graph.pseudotree import PseudoTreeLink, get_dfs_relations
from pydcop.dcop.relations import Constraint
from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation
from pydcop.infrastructure.discovery import Discovery

Neighbor = namedtuple(
    'Neighbor',
    field_names=['agent_id', 'address', 'computations'],
)


class DynamicGraphConstructionComputation(MessagePassingComputation):
    """
    This serves as a protocol class for all dynamic graph construction algorithms.
    """

    def __init__(self, algo_name: str, agent: DynamicAgent, discovery: Discovery):
        super(DynamicGraphConstructionComputation, self).__init__(name=f'{algo_name}-{agent.name}')

        self.logger = logging.getLogger(f'pydcop.computation.{self.name}')

        self.discovery = discovery
        self.agent = agent
        self.address = agent.address
        self._dcop_comps = []

        self.parent: Neighbor = None
        self.children: List[Neighbor] = []
        self.neighbor_comps: List[str] = []

        # added to avoid AttributeError on metric collection
        self.cycle_count = 0

    @property
    def neighbors(self) -> List[Neighbor]:
        nodes = self.children
        if self.parent:
            nodes += [self.parent]
        return nodes

    @property
    def computations(self) -> List[MessagePassingComputation]:
        return self._dcop_comps

    def add_computation(self, comp):
        self.logger.debug(f'Adding computation: {str(comp)}')
        self._dcop_comps.append(comp)
        self.agent.run(comp.name)

    def register_neighbor(self, neighbor: Neighbor):
        configure = False
        for comp in neighbor.computations:
            self.discovery.register_computation(
                computation=comp,
                agent=neighbor.agent_id,
                address=neighbor.address,
                publish=False,
            )
            self.neighbor_comps.append(comp)
            configure = True
        if configure:
            self.configure_computations()

    def configure_computations(self):
        """
        todo: refactor this method to avoid setting all constraints and links on every call
        """
        for computation in self.computations:
            # update computation definition
            if hasattr(computation, 'computation_def'):
                computation_def: ComputationDef = computation.computation_def

                if computation_def.exec_mode is not 'dynamic':
                    computation_def.exec_mode = 'dynamic'

                # configure computation node constraints and links
                self._configure(computation)

    def _configure(self, computation):
        """
        Creates a computation node.
        """
        # get constraints
        constraints = []
        links = []
        dynamic_node: DynamicComputationNode = computation.computation_def.node
        var_constraints: Iterable[Constraint] = dynamic_node.var_constraints

        for n_comp in self.neighbor_comps:
            for constraint in var_constraints:
                if n_comp in constraint.scope_names:
                    constraints.append(constraint)

                    if dynamic_node.type == 'VariableComputationNode':
                        links.append(
                            ConstraintLink(name=constraint.name, nodes=[v.name for v in constraint.dimensions])
                        )
                    elif dynamic_node.type == 'PseudoTreeComputation':
                        link_type = 'parent' if self.parent and self.parent.agent_id == \
                                                self.discovery.computation_agent(n_comp) else 'children'
                        links.append(
                            PseudoTreeLink(
                                link_type=link_type,
                                source=dynamic_node.variable.name,
                                target=n_comp,
                            )
                        )

        # set node properties for dcop computation
        dynamic_node.constraints = constraints
        dynamic_node.links = links
        dynamic_node.neighbors = list(set(n for l in links for n in l.nodes if n != dynamic_node.name))

    def _execute_computations(self, exec_order):
        for computation in self.computations:
            if hasattr(computation, 'computation_def'):
                algo_exec_order = computation.computation_def.algo.params.get('execution_order', None)
                if algo_exec_order is None or algo_exec_order == exec_order:
                    computation.start()

