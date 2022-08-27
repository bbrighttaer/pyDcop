import logging
import threading
from collections import namedtuple
from threading import Event
from typing import List, Iterable, Callable, Union

from pydcop.algorithms import ComputationDef
from pydcop.computations_graph.constraints_hypergraph import ConstraintLink
from pydcop.computations_graph.dynamic_graph import DynamicComputationNode
from pydcop.computations_graph.pseudotree import PseudoTreeLink, get_dfs_relations
from pydcop.dcop.relations import Constraint
from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation, Message
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

        self._periodic_calls_cancel_list = []

        # supposed to be overridden by subclass to handle messages
        self._msg_handlers = {}

    @property
    def neighbors(self) -> List[Neighbor]:
        nodes = self.children
        if self.parent:
            nodes += [self.parent]
        return nodes

    @property
    def computations(self) -> List[MessagePassingComputation]:
        return self._dcop_comps

    def find_neighbor_by_agent_id(self, agent_id) -> Union[Neighbor, None]:
        for n in self.neighbors:
            if n.agent_id == agent_id:
                return n

    def on_message(self, sender: str, msg: Message, t: float):
        try:
            self._msg_handlers[msg.type](sender, msg)
        except KeyError:
            self.logger.error(f'Could not find function callback for msg type: {msg.type}')

    def add_computation(self, comp):
        """
        Adds a computation to those handled by the dynamic graph algorithm.

        Parameters
        ----------
        comp: MessagePassingComputation
            The computation to be added
        """
        self.logger.debug(f'Adding computation: {str(comp)}')
        self._dcop_comps.append(comp)
        self.agent.run(comp.name)

    def register_neighbor(self, neighbor: Neighbor, callback: Callable = None):
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
        self.logger.debug(f'registered neighbor {neighbor.agent_id}, comps={self.neighbor_comps}')
        if configure:
            self.configure_computations()

            if callback:
                callback(neighbor)

    def unregister_neighbor(self, neighbor: Neighbor, callback: Callable = None):
        self.logger.debug(f'Unregistering neighbor {neighbor.agent_id}')

        if neighbor:
            # remove all computations stored at the dynamic algorithm level
            comps = self.discovery.agent_computations(neighbor.agent_id)
            self.logger.debug(f'comps: {comps}, n_comps: {self.neighbor_comps}')
            for c in comps:
                if c in self.neighbor_comps:
                    self.neighbor_comps.remove(c)

            # unregister agent and computations from discovery
            self.discovery.unregister_agent(neighbor.agent_id, publish=False)

            # remove associations
            if self.parent == neighbor:
                neighbor_type = 'parent'
                self.parent = None
            else:
                neighbor_type = 'child'
                self.children.remove(neighbor)

            # fire callback
            if callback:
                callback(neighbor=neighbor, neighbor_type=neighbor_type)

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

    def _execute_computations(self, exec_order=None, is_reconfiguration=False):
        if self.neighbors:
            for computation in self.computations:
                if hasattr(computation, 'computation_def'):
                    algo_exec_order = computation.computation_def.algo.params.get('execution_order', None)
                    if algo_exec_order is None or algo_exec_order == exec_order or is_reconfiguration:
                        self.logger.debug(f'Executing dcop, neighbors {computation.computation_def.node.neighbors}')
                        computation.start()
        else:
            self.logger.debug('No neighbors available for computation')

    def on_stop(self):
        # cancel any background process
        for cancel in self._periodic_calls_cancel_list:
            cancel()

        # stop all computations
        for comp in self.computations:
            comp.stop()

    def _periodic_action(self, interval: int, func, *args, **kwargs):
        stopped = Event()
        def loop():
            while not self.agent._stopping.is_set() and not stopped.wait(interval):
                func(*args, **kwargs)
        threading.Thread(target=loop).start()
        return stopped.set
