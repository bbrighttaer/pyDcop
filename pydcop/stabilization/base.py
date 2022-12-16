import logging
import threading
import time
from threading import Event
from typing import List, Iterable, Callable, Union

from pydcop.algorithms import ComputationDef
from pydcop.computations_graph import constraints_hypergraph, pseudotree
from pydcop.computations_graph.constraints_hypergraph import ConstraintLink
from pydcop.computations_graph.dynamic_graph import DynamicComputationNode
from pydcop.computations_graph.pseudotree import PseudoTreeLink
from pydcop.dcop.relations import Constraint
from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.communication import MSG_MGT
from pydcop.infrastructure.computations import MessagePassingComputation, Message
from pydcop.infrastructure.discovery import Discovery
from pydcop.infrastructure.orchestratedagents import ORCHESTRATOR_MGT
from pydcop.infrastructure.orchestrator import GraphConnectionMessage
from pydcop.stabilization import Neighbor


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

        self.parent: Union[Neighbor, None] = None
        self.children: List[Neighbor] = []
        self.neighbor_comps: List[str] = []

        # tracks the last time a neighbor sent a message
        self.last_contact_time = {}

        # added to avoid AttributeError on metric collection
        self.cycle_count = 0

        self._periodic_calls_cancel_list = []

        # supposed to be overridden by subclass to handle messages
        self._msg_handlers = {}

        # callbacks
        self._on_computation_added_cb: Union[Callable[[MessagePassingComputation], None], None] = None

    @property
    def neighbors(self) -> List[Neighbor]:
        nodes = list(self.children)
        if self.parent:
            nodes += [self.parent]
        return nodes

    @property
    def neighbor_ids(self) -> List[str]:
        return [n.agent_id for n in self.neighbors]

    @property
    def computations(self) -> List[MessagePassingComputation]:
        return self._dcop_comps

    def find_neighbor_by_agent_id(self, agent_id) -> Union[Neighbor, None]:
        for n in self.neighbors:
            if n.agent_id == agent_id:
                return n

    def on_message(self, sender: str, msg: Message, t: float):
        try:
            if hasattr(msg, 'agent_id'):
                self.last_contact_time[msg.agent_id] = time.time()
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

        if self._on_computation_added_cb:
            self._on_computation_added_cb(comp)

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

                if computation_def.exec_mode != 'dynamic':
                    computation_def.exec_mode = 'dynamic'

                # configure computation node constraints and links
                self._configure(computation)

    def _configure(self, computation):
        """
        Sets up a computation node.
        """
        # get constraints
        constraints = []
        links = []
        dynamic_node: DynamicComputationNode = computation.computation_def.node
        var_constraints: Iterable[Constraint] = dynamic_node.var_constraints

        for n_comp in self.neighbor_comps:
            for constraint in var_constraints:
                if n_comp in constraint.scope_names and n_comp in self.discovery.computations():
                    constraints.append(constraint)

                    if dynamic_node.type == constraints_hypergraph.GRAPH_NODE_TYPE:
                        links.append(
                            ConstraintLink(name=constraint.name, nodes=[v.name for v in constraint.dimensions])
                        )
                    elif dynamic_node.type == pseudotree.GRAPH_NODE_TYPE:
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
        self.logger.debug(f'constraints = {constraints}, links = {links}')
        dynamic_node.neighbors = list(set(n for l in links for n in l.nodes if n != dynamic_node.name))

    def execute_computations(self, exec_order=None, is_reconfiguration=False):
        if self.neighbors:
            for computation in self.computations:
                if hasattr(computation, 'computation_def'):
                    algo_exec_order = computation.computation_def.algo.params.get('execution_order', None)
                    if algo_exec_order is None or algo_exec_order == exec_order or is_reconfiguration:
                        self.logger.debug(f'Executing dcop, neighbors {computation.computation_def.node.neighbors}')
                        time.sleep(.3)
                        computation.start()
        else:
            self.logger.debug('No neighbors available for computation')

    def on_start(self):
        super(DynamicGraphConstructionComputation, self).on_start()

        # start DCOP computations
        self.agent.run([c.name for c in self.computations])

    def on_stop(self):
        # cancel any background process
        for cancel in self._periodic_calls_cancel_list:
            cancel()

        # stop all computations
        for comp in self.computations:
            # self.discovery.unregister_computation(comp.name, agent=self.agent.name)
            comp.stop()

        # report removal
        # self.post_msg(
        #     ORCHESTRATOR_MGT,
        #     GraphConnectionMessage(
        #         action='remove_node',
        #         node1=self.agent.name,
        #         node2=None,
        #     ),
        #     MSG_MGT
        # )

    def _periodic_action(self, interval: int, func, *args, **kwargs):
        stopped = Event()
        def loop():
            while not self.agent._stopping.is_set() and not stopped.wait(interval):
                func(*args, **kwargs)
        threading.Thread(target=loop).start()
        return stopped.set

    def _has_constraint_with(self, neighbor_comps: List[str]) -> bool:
        """
        Checks if the given neighbor computations has at least one constraint with the current agent.
        This is implemented to facilitate simulation. In other scenarios where a default constraint function is used or
        the same constraint function is used by all connections, this function may not be needed.

        Parameters
        ----------
        neighbor_comps: list
            The list of neighbor computations to check.

        Returns
        -------
            True if at least one constraint exists between the current agent (var computation) and any of the given
            neighbor (var) computations. Otherwise, False.
        """
        for computation in self.computations:
            var_constraints: Iterable[Constraint] = computation.computation_def.node.var_constraints
            for constraint in var_constraints:
                common = set(constraint.scope_names) & set(neighbor_comps)
                if common:
                    return True
        return False
