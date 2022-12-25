import datetime
import logging
import time
from typing import List, Callable, Union

from pydcop.computations_graph import constraints_hypergraph, pseudotree
from pydcop.computations_graph.dynamic_graph import DynamicComputationNode
from pydcop.computations_graph.ordered_graph import ConstraintLink
from pydcop.computations_graph.pseudotree import PseudoTreeLink
from pydcop.dcop.objects import Variable, VariableDomain
from pydcop.dcop.relations import DynamicEnvironmentSimulationRelation
from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation, Message, register
from pydcop.infrastructure.discovery import Discovery
from pydcop.infrastructure.message_types import ConstraintEvaluationResponse, AgentMovedMessage
from pydcop.infrastructure.orchestrator import SimTimeStepChanged, DcopExecutionMessage, RunAgentMessage
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

        self.parent: Union[Neighbor, None] = None
        self.children: List[Neighbor] = []
        self.domain = []

        # tracks the last time a neighbor sent a message
        self.last_contact_time = {}

        # added to avoid AttributeError on metric collection
        self.cycle_count = 0

        # supposed to be overridden by subclass to handle messages
        self._msg_handlers = {
            'sim_time_step_change': self.receive_sim_step_changed,
        }

        # records the current position of the agent in the environment
        self.current_position = None

    @property
    def neighbors(self) -> List[Neighbor]:
        nodes = list(self.children)
        if self.parent:
            nodes += [self.parent]
        return nodes

    @property
    def neighbor_ids(self) -> List[str]:
        return [n.agent_id for n in self.neighbors]

    def receive_sim_step_changed(self, sender: str, msg: SimTimeStepChanged):
        """
        Handles simulation time step changed events.

        Parameters
        ----------
        sender
        msg

        Returns
        -------

        """
        pass

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

    def register_neighbor(self, neighbor: Neighbor, callback: Callable = None):
        for comp in neighbor.computations:
            self.discovery.register_computation(
                computation=comp,
                agent=neighbor.agent_id,
                address=neighbor.address,
                publish=False,
            )
        self.logger.debug(f'registered neighbor {neighbor.agent_id}, comps={neighbor.computations}')

        if callback:
            callback(neighbor)

    def unregister_neighbor(self, neighbor: Neighbor, callback: Callable = None):
        self.logger.debug(f'Unregistering neighbor {neighbor.agent_id}')

        if neighbor:
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

    def execute_computations(self, exec_order: str = None):
        self.logger.info(f'Executing agent DCOP computations: {exec_order}')

        # send DCOP execution message
        for computation in self.agent.computations():
            if hasattr(computation, 'computation_def') and computation.computation_def is not None:
                # send management command to run the computation
                if not computation.is_running:
                    self.post_msg(
                        target='_mgt_' + self.agent.name,
                        msg=RunAgentMessage([computation.name])
                    )

                # send message to running computation
                self.post_msg(
                    target=computation.name,
                    msg=DcopExecutionMessage(data={
                        'parent': self.parent,
                        'children': self.children,
                        'domain': self.domain,
                        'exec_order': exec_order,
                        'current_position': self.current_position,
                    }),
                )


class DynamicDcopComputationMixin:
    """
    A mixin class to provide Dynamic computation (::class::VariableComputation) methods to classic DCOP algorithms
    """

    def start_dcop(self):
        raise NotImplementedError('Implement start_dcop in inherited class')

    @register('dcop_execution_message')
    def _on_dcop_execution_message(self, sender: str, recv_msg: DcopExecutionMessage, t: int):
        self.logger.info(f'DCOP execution message: {recv_msg}')

        # when starting, record the starting position of the agent.
        # subsequent position information are carried by AgentMovedMessage
        if not hasattr(self, 'position_history'):
            self.record_current_position(recv_msg.data['current_position'])

        ts = str(datetime.datetime.now().timestamp())

        # get msg components
        parent: Neighbor = recv_msg.data['parent']
        children: List[Neighbor] = recv_msg.data['children']

        # DCOP components
        self.variable.domain.values = recv_msg.data['domain']

        # update DCOP properties
        constraints = []
        links = []
        dynamic_node: DynamicComputationNode = self.computation_def.node

        if dynamic_node.type == constraints_hypergraph.GRAPH_NODE_TYPE:
            neighbors = [parent] if parent else []
            neighbors += children

            # extract dcop computation names from neighbor computations' list
            for i, n in enumerate(neighbors):
                for j, comp_name in enumerate(n.computations):
                    if 'var' in comp_name:
                        variables = [
                            Variable(self.name, VariableDomain(self.name, self.name, recv_msg.data['domain'])),
                            Variable(comp_name, VariableDomain(comp_name, comp_name, []))
                        ]
                        constraint = DynamicEnvironmentSimulationRelation(f'c-{self.name}-{i}{j}-{ts}', self, variables)
                        constraints.append(constraint)
                        links.append(
                            ConstraintLink(
                                name=constraint.name,
                                nodes=[self.name, comp_name],
                            )
                        )

        elif dynamic_node.type == pseudotree.GRAPH_NODE_TYPE:
            for i, n in enumerate(children):
                for j, comp_name in enumerate(n.computations):
                    if 'var' in comp_name:
                        variables = [
                            Variable(self.name, VariableDomain(self.name, self.name, recv_msg.data['domain'])),
                            Variable(comp_name, VariableDomain(comp_name, comp_name, []))
                        ]
                        constraint = DynamicEnvironmentSimulationRelation(f'c-{self.name}-{i}{j}-{ts}', self, variables)
                        constraints.append(constraint)
                        links.append(
                            PseudoTreeLink(
                                link_type='children',
                                source=self.name,
                                target=comp_name,
                            )
                        )
            if parent:
                for i, comp_name in enumerate(parent.computations):
                    if 'var' in comp_name:
                        variables = [
                            Variable(self.name, VariableDomain(self.name, self.name, recv_msg.data['domain'])),
                            Variable(comp_name, VariableDomain(comp_name, comp_name, []))
                        ]
                        constraint = DynamicEnvironmentSimulationRelation(f'c-{self.name}-p{i}-{ts}', self, variables)
                        constraints.append(constraint)
                        links.append(
                            PseudoTreeLink(
                                link_type='parent',
                                source=self.name,
                                target=comp_name,
                            )
                        )

        # set node properties for dcop computation
        dynamic_node.constraints = constraints
        dynamic_node.links = links
        dynamic_node.neighbors = list(set(n for l in links for n in l.nodes if n != dynamic_node.name))
        self.logger.debug(f'constraints = {constraints}, links = {links}, neighbors = {dynamic_node.neighbors}')

        # trigger DCOP computation
        self.start_dcop()

    @register('constraint_evaluation_response')
    def _on_constraint_evaluation_response(self, sender: str, recv_msg: ConstraintEvaluationResponse, t: int):
        self.logger.debug(f'Received constraint evaluation response: {recv_msg} from {sender}')

        # set return value on the constraint that made the request
        for c in self.computation_def.node.constraints:
            if c.name == recv_msg.constraint_name and isinstance(c, DynamicEnvironmentSimulationRelation):
                c.set_return_value(recv_msg.value)

    @register('agent_moved')
    def _on_agent_moved_msg(self, sender: str, recv_msg: AgentMovedMessage, t: int):
        self.record_current_position(recv_msg.position)

    def record_current_position(self, position):
        if not hasattr(self, 'position_history'):
            self.position_history = []
        self.position_history.append(position)


