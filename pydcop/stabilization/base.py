import datetime
import logging
import threading
import time
from collections import defaultdict
from typing import List, Callable, Union

from pydcop.computations_graph import constraints_hypergraph, pseudotree
from pydcop.computations_graph.dynamic_graph import DynamicComputationNode
from pydcop.computations_graph.ordered_graph import ConstraintLink
from pydcop.computations_graph.pseudotree import PseudoTreeLink
from pydcop.dcop.objects import Variable, VariableDomain
from pydcop.dcop.relations import AsyncNaryFunctionRelation, AsyncNAryMatrixRelation
from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation, register
from pydcop.infrastructure.discovery import Discovery
from pydcop.infrastructure.message_types import ConstraintEvaluationResponse, AgentMovedMessage, SimTimeStepChanged, \
    DcopExecutionMessage, DcopConfigurationMessage, DcopInitializationMessage, Message
from pydcop.infrastructure.orchestrator import RunAgentMessage
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
        self.neighbor_domains = {}

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

        self.configure_dcop_computation()

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
        prior_neighbors = set(self.neighbor_ids)

        self.logger.info(f'Received simulation time step changed: {msg}')
        self.domain = msg.data['agent_domain']
        self.current_position = msg.data['current_position']
        self.neighbor_domains = msg.data['neighbor_domains']

        # initialize dcop algorithm
        self.initialize_computations()

        agents_in_comm_range = set(msg.data['agents_in_comm_range'])

        # remove agents that are out of range
        self.inspect_connections(agents_in_comm_range)

        # configuration
        self.logger.debug('configure call in time step changed receiver')
        self.configure_dcop_computation()

        is_affected = prior_neighbors != agents_in_comm_range

        if is_affected:
            self.logger.debug(f'Neighborhood change detected')
            # broadcast connection request
            self.connect()
        else:
            self.logger.debug('No neighborhood change detected')
            self.execute_computations(exec_order='no-new-neighbor')

    def initialize_computations(self):
        # send DCOP initialization message
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
                    msg=DcopInitializationMessage(),
                )

    def configure_dcop_computation(self):
        self.logger.info('Configuring DCOP computations')

        # send DCOP execution message
        for computation in self.agent.computations():
            if hasattr(computation, 'computation_def') and computation.computation_def is not None:
                # send message to running computation
                self.post_msg(
                    target=computation.name,
                    msg=DcopConfigurationMessage(data={
                        'parent': self.parent,
                        'children': self.children,
                        'domain': self.domain,
                        'current_position': self.current_position,
                        'neighbor_domains': self.neighbor_domains,
                    }),
                )

    def execute_computations(self, exec_order: str = None):
        self.logger.info(f'Executing agent DCOP computations: {exec_order}')

        # send DCOP execution message
        for computation in self.agent.computations():
            if hasattr(computation, 'computation_def') and computation.computation_def is not None:
                # send message to running computation
                self.post_msg(
                    target=computation.name,
                    msg=DcopExecutionMessage(),
                )

    def connect(self):
        raise NotImplementedError('Connection function has not been implemented yet')

    def inspect_connections(self, agents_in_comm_range):
        raise NotImplementedError('Connection inspection function has not been implemented yet')


class DynamicDcopComputationMixin:
    """
    A mixin class to provide Dynamic computation (::class::VariableComputation) methods to classic DCOP algorithms
    """

    def __init__(self):
        self.position_history = []
        self.async_func_return_val = {}

    def initialize(self):
        raise NotImplementedError('Implement initialization method in child class')

    def start_dcop(self):
        raise NotImplementedError('Implement start_dcop method in child class')

    def on_computation_node_configured_cb(self):
        """
        Called when the computation definition node has been configured. This can be leveraged in the DCOP algorithm
        implementation to execute any ops that should follow after the computation node has been set up.
        """
        ...

    @register('dcop_initialization_message')
    def _on_dcop_initialization_message(self, sender: str, recv_msg: DcopInitializationMessage, t: int):
        self.initialize()

    @register('dcop_configuration_message')
    def _on_dcop_configuration_message(self, sender: str, recv_msg: DcopConfigurationMessage, t: int):
        # todo: refactor this method, it should not recreate all existing/unchanged connections
        self.logger.debug(f'DCOP configuration message: {recv_msg}')

        neighbor_domains = recv_msg.data['neighbor_domains']

        # get msg components
        parent: Neighbor = recv_msg.data['parent']
        children: List[Neighbor] = recv_msg.data['children']

        # DCOP components
        variable = Variable(self.name, VariableDomain(self.name, self.name, recv_msg.data['domain']))
        self.variable = variable

        # update DCOP properties
        constraints = []
        links = []
        dynamic_node: DynamicComputationNode = self.computation_def.node

        relation_class = get_relation_class(self.computation_def.algo.algo)

        if dynamic_node.type == constraints_hypergraph.GRAPH_NODE_TYPE:
            neighbors = [parent] if parent else []
            neighbors += children

            # extract dcop computation names from neighbor computations' list
            for i, n in enumerate(neighbors):
                for j, comp_name in enumerate(n.computations):
                    if 'var' in comp_name:
                        variables = [
                            variable,
                            Variable(comp_name, VariableDomain(
                                comp_name, comp_name, neighbor_domains[comp_name.replace('a', 'var')]
                            ))
                        ]
                        constraint = relation_class(self, variables, name=f'c-{self.name}-{i}{j}')
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
                            variable,
                            Variable(comp_name, VariableDomain(
                                comp_name, comp_name, neighbor_domains[comp_name.replace('var', 'a')]
                            ))
                        ]
                        constraint = relation_class(self, variables, name=f'c-{self.name}-{i}{j}')
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
                            variable,
                            Variable(comp_name, VariableDomain(
                                comp_name, comp_name, neighbor_domains[comp_name.replace('var', 'a')]
                            ))
                        ]
                        constraint = relation_class(self, variables, name=f'c-{self.name}-p{i}')
                        constraints.append(constraint)
                        links.append(
                            PseudoTreeLink(
                                link_type='parent',
                                source=self.name,
                                target=comp_name,
                            )
                        )

        # set node properties for dcop computation
        dynamic_node.constraints = constraints or [relation_class(self, [variable], name=f'c-{self.name}')]
        dynamic_node.links = links
        dynamic_node.neighbors = list(set(n for l in links for n in l.nodes if n != dynamic_node.name))
        self.logger.debug(f'constraints = {constraints}, links = {links}, neighbors = {dynamic_node.neighbors}')

        # callback
        self.on_computation_node_configured_cb()

    @register('dcop_execution_message')
    def _on_dcop_execution_message(self, sender: str, recv_msg: DcopExecutionMessage, t: int):
        self.logger.info(f'DCOP execution message: {recv_msg}')

        # trigger DCOP computation
        self.start_dcop()

    @register('constraint_evaluation_response')
    def _on_constraint_evaluation_response(self, sender: str, recv_msg: ConstraintEvaluationResponse, t: int):
        self.logger.debug(f'Received constraint evaluation response: {recv_msg} from {sender}')
        self.async_func_return_val[recv_msg.constraint_name] = recv_msg.value

    @register('agent_moved')
    def _on_agent_moved_msg(self, sender: str, recv_msg: AgentMovedMessage, t: int):
        self.position_history.append(f'from {recv_msg.prev_position} to {recv_msg.new_position}')


def get_relation_class(algo_name):
    """
    Select relation/constraint class based on DCOP algorithm use
    """
    relation_class = {
        'cocoa': AsyncNaryFunctionRelation,
        'ddpop': AsyncNAryMatrixRelation,
    }.get(algo_name)
    return relation_class
