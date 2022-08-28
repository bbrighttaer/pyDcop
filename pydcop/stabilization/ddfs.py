from threading import Event, Thread
from typing import Iterable, Dict

from pydcop.computations_graph.dynamic_graph import DynamicComputationNode
from pydcop.dcop.relations import Constraint
from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation, message_type
from pydcop.infrastructure.discovery import Discovery
from pydcop.stabilization import Neighbor, AgentID, MaxDegree
from pydcop.stabilization.base import DynamicGraphConstructionComputation

NAME = 'DDFS'

MaxDegreeRequest = message_type(
    'max_degree_request',
    fields=['agent_id'],
)

MaxDegreeResponse = message_type(
    'max_degree_response',
    fields=['agent_id', 'max_degree'],
)


def build_stabilization_computation(agent: DynamicAgent, discovery: Discovery) -> MessagePassingComputation:
    """
    Builds a computation for D-DCOP simulation using the stabilization approach to handling dynamics.
    Parameters
    ----------
    agent: Agent
        the agent the computation is replicating for. it
        contains necessary info like hosting and route cost.
    discovery: Discovery
    Returns
    -------
    A computation object to dynamically construct a local interaction graph for the agent.
    """
    return DistributedDFS(NAME, agent, discovery)


class DistributedDFS(DynamicGraphConstructionComputation):
    """
    Implementation of the DDFS algorithm for dynamic DCOP
    """

    def __init__(self, name, agent: DynamicAgent, discovery: Discovery):
        super(DistributedDFS, self).__init__(name, agent, discovery)

        self._registered_neighbors: Dict[AgentID, Neighbor] = {}
        self._max_degree_register: Dict[AgentID, MaxDegree] = {}

        self._msg_handlers = {
            'max_degree_request': self._on_max_degree_request,
            'max_degree_response': self._on_max_degree_response,
        }

        self._on_computation_added_cb = self._subscribe_to_neighbor_computations
        self.num_acquaintances = 0

        self._shutdown = False
        self._split_event = Event()
        self._all_neighbor_max_degrees_event = Event()
        self._execute_dcop_event = Event()
        self.t = Thread(target=self._lambda_split, name=f'thread_{self.name}_bg_process')
        self.t.daemon = True

    def on_start(self):
        self.logger.debug(f'On start of {self.name}')
        self.t.start()

    def _subscribe_to_neighbor_computations(self, computation: MessagePassingComputation):
        self.logger.debug(f'Subscribing to neighbor computations of {computation.name}')

        dynamic_node: DynamicComputationNode = computation.computation_def.node
        var_constraints: Iterable[Constraint] = dynamic_node.var_constraints
        self.num_acquaintances = len(var_constraints)

        for constraint in var_constraints:
            self.logger.debug(f'constraint {constraint.name} scope = {constraint.scope_names}')
            for comp_name in constraint.scope_names:
                if comp_name != dynamic_node.name:
                    self.discovery.subscribe_computation(
                        computation=comp_name,
                        cb=self._on_neighbor_computation_added_and_removed,
                    )

    def _on_neighbor_computation_added_and_removed(self, cb_type: str, computation: str, agent_name: str):
        if cb_type == 'computation_added':
            self.logger.debug(f'On neighbor computation added: {cb_type}, {computation}, {agent_name}')

            # register corresponding DDFS computation of neighbor.
            # since calling discovery.register_computation will trigger a circular call of this func,
            # it is registered directly.
            ddfs_comp = f'{NAME}-{agent_name}'
            self.discovery._computations_data[ddfs_comp] = agent_name

            self.neighbor_comps.append(ddfs_comp)
            self.neighbor_comps.append(computation)

            # update list of active neighbors
            self._registered_neighbors[agent_name] = Neighbor(
                agent_id=agent_name,
                address=self.discovery.agent_address(agent_name),
                computations=[ddfs_comp, computation]
            )

            # ask for this neighbor's max-degree
            self.post_msg(
                target=ddfs_comp,
                msg=MaxDegreeRequest(agent_id=self.agent.name),
            )
        elif cb_type == 'computation_removed':
            self.logger.debug(f'Removing neighbor computation {cb_type}, {computation}')
            self._unregister_neighbor_comp(computation)

        self._split_event.set()

    def _unregister_neighbor_comp(self, computation: str):
        agent = self.discovery.computation_agent(computation)
        if agent in self._registered_neighbors:
            self._registered_neighbors.pop(agent)

        if computation in self.neighbor_comps:
            self.neighbor_comps.remove(computation)
            self.neighbor_comps.remove(f'{NAME}-{agent}')

        self._split_event.set()

    def _on_max_degree_request(self, sender: str, msg: MaxDegreeRequest):
        self.post_msg(
            target=sender,
            msg=MaxDegreeResponse(
                agent_id=self.agent.name,
                max_degree=self.num_acquaintances,
            )
        )

    def _on_max_degree_response(self, sender: str, msg: MaxDegreeResponse):
        self._max_degree_register[msg.agent_id] = msg.max_degree
        if len(self._max_degree_register) == len(self._registered_neighbors):
            self._all_neighbor_max_degrees_event.set()

    def _lambda_split(self):
        # monitor affected status
        while self._split_event.wait():
            if self._shutdown:
                break

            # wait for all available neighbors
            self._all_neighbor_max_degrees_event.wait()

            # split neighbors
            self._split()

            # set flags
            self.logger.debug(f'Resetting lambda_split flags')
            self._all_neighbor_max_degrees_event.clear()
            self._split_event.clear()

    def _split(self):
        self.logger.debug('splitting neighbors')

        parent = None
        children = []
        self.logger.debug(f'max-degree: {self._max_degree_register}')
        for agt in self._registered_neighbors:
            neighbor: Neighbor = self._registered_neighbors[agt]
            if self._max_degree_register[agt] < self.num_acquaintances or (
                    self._max_degree_register[agt] == self.num_acquaintances and neighbor.agent_id < self.agent.name
            ):
                children.append(neighbor)
            else:
                parent = neighbor
        self.parent = parent
        self.children = children
        self.logger.debug(f'Parent = {self.parent}, children = {self.children}')

        # reconfigure properties for dcop computation
        configured = False
        for computation in self.computations:
            self._configure(computation)
            configured = True
        if configured:
            self.execute_computations(is_reconfiguration=True)



