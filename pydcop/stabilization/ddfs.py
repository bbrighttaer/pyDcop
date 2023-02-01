from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.communication import MSG_ALGO, MSG_MGT
from pydcop.infrastructure.computations import MessagePassingComputation
from pydcop.infrastructure.discovery import Discovery, AsyncBroadcastMessage
from pydcop.infrastructure.message_types import message_type, GraphConnectionMessage, SimTimeStepChangeMsgAck
from pydcop.infrastructure.orchestratedagents import ORCHESTRATOR_DIRECTORY, ORCHESTRATOR_MGT
from pydcop.stabilization import Neighbor, transient_communication
from pydcop.stabilization.base import DynamicGraphConstructionComputation

Announce = message_type(
    'announce',
    fields=['agent_id', 'address', 'comps'],
)

NeighborDataRequest = message_type(
    'neighbor_data_request',
    fields=['agent_id', 'address'],
)


NeighborData = message_type(
    'neighbor_data',
    fields=['agent_id', 'num_neighbors', 'address', 'comps'],
)

PositionMsg = message_type(
    'position_msg',
    fields=['agent_id', 'position'],
)

ValueMsg = message_type(
    'value_msg',
    fields=['value'],
)

ChildMsg = message_type(
    'child_msg',
    fields=['agent_id'],
)

PseudoChildMsg = message_type(
    'pseudo_child_msg',
    fields=['agent_id'],
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
    return DistributedDFS(DistributedDFS.NAME, agent, discovery)


class DistributedDFS(DynamicGraphConstructionComputation):
    """
    Implementation of the DDFS algorithm for dynamic DCOP
    """

    NAME = 'DDFS'

    def __init__(self, name, agent: DynamicAgent, discovery: Discovery):
        super(DistributedDFS, self).__init__(name, agent, discovery)

        self._neighbors = {}
        self._num_neighbors_of_neighbor = {}
        self._value_msg_senders = []
        self._parents_level_dict = {}
        self._children_temp = []

        self._max = 0
        self._parents = []

        # props for handling paused msgs
        # self._paused_ann_msgs_queue = Queue()

        self._msg_handlers.update({
            'announce': self._receive_announce,
            'neighbor_data': self._on_receive_neighbor_data,
            'position_msg': self._on_position_msg,
            'value_msg': self._on_value_msg,
            'pseudo_child_msg': self._on_pseudo_child_msg,
            'child_msg': self._on_child_msg,
        })

    def on_start(self):
        super(DistributedDFS, self).on_start()
        self.logger.debug(f'On start of {self.name}')

    def connect(self):
        self.logger.debug('Connecting...')
        # publish Announce msg
        self.post_msg(
            target=ORCHESTRATOR_DIRECTORY,
            msg=AsyncBroadcastMessage(message=Announce(
                agent_id=self.agent.name,
                address=self.address,
                comps=[c.name for c in self.agent.computations()]
            ),
                originator=self.name,
                recipient_prefix=self.NAME
            ),
            prio=MSG_ALGO,
            on_error='fail',
        )

    def _receive_announce(self, sender: str, msg: Announce):
        self.logger.debug(f'Received announce msg from {sender}: {msg}')

        # if all neighbors have responded, request neighbor information
        if self.agents_in_comm_range:
            dest_comp = f'{self.NAME}-{msg.agent_id}'
            with transient_communication(self.discovery, dest_comp, msg.agent_id, msg.address):
                self.post_msg(
                    target=dest_comp,
                    msg=NeighborData(
                        agent_id=self.agent.name,
                        num_neighbors=len(self.agents_in_comm_range),
                        address=self.address,
                        comps=[c.name for c in self.agent.computations()],
                    ),
                )

    def _on_receive_neighbor_data(self, sender: str, msg: NeighborData):
        """
        Receives response to an Announce/Broadcast message that was sent earlier.
        """
        self.logger.debug(f'Received neighbor data from {sender}: {msg}')
        self._num_neighbors_of_neighbor[msg.agent_id] = msg.num_neighbors

        # record neighbor information
        self._neighbors[msg.agent_id] = Neighbor(
            agent_id=msg.agent_id,
            address=msg.address,
            computations=msg.comps,
        )

        # perform max-degree splitting of neighbors
        if len(self._neighbors) == len(self.agents_in_comm_range):
            self._split_neighbors()

    def on_neighbor_removed(self, neighbor: Neighbor, *args, **kwargs):
        ...

    def _split_neighbors(self):
        """
        split neighbors into children and parents
        """
        self.logger.debug('Splitting neighbors')

        # base class props
        self.parent = None
        self.children.clear()
        self.pseudo_parents.clear()
        self.pseudo_children.clear()
        self._parents.clear()
        self._children_temp.clear()

        self.logger.debug(f'num_neighbors_of_neighbor: {self._num_neighbors_of_neighbor}')

        for agt, num_neighbors in self._num_neighbors_of_neighbor.items():
            neighbor = self._neighbors[agt]

            if num_neighbors < len(self._neighbors) \
                    or (num_neighbors == len(self._neighbors) and self.agent.name < agt):
                self._children_temp.append(neighbor)
            else:
                self._parents.append(neighbor)

        self.logger.debug(f'After splitting: children_temp={self._children_temp}, parents_temp={self._parents}')

        # if this agent is a leaf, begin level calculation for ordering parents
        if len(self._children_temp) == 0 and self._parents:
            self._max = 1

            for p in self._parents:
                dest_comp = f'{self.NAME}-{p.agent_id}'
                with transient_communication(self.discovery, dest_comp, p.agent_id, p.address):
                    self.post_msg(
                        target=dest_comp,
                        msg=ValueMsg(value=self._max),
                    )

    def _register(self, neighbor: Neighbor, update_graph=True):
        # registration and configuration
        self.register_neighbor(neighbor)

        # report connection to graph UI
        if update_graph:
            self.post_msg(
                ORCHESTRATOR_MGT,
                GraphConnectionMessage(
                    action='add',
                    node1=self.agent.name,
                    node2=neighbor.agent_id,
                ),
                MSG_MGT,
            )

    def _on_value_msg(self, sender: str, msg: ValueMsg):
        self.logger.debug(f'Received value msg from {sender}: {msg}')
        if sender not in self._value_msg_senders:
            self._value_msg_senders.append(sender)

            if self._max < msg.value:
                self._max = msg.value
            self.logger.debug(f'received: {self._value_msg_senders}, ctemp: {self._children_temp}')
            if len(self._value_msg_senders) == len(self._children_temp):
                self._max += 1

                # send max value info to ancestors
                for p in self._parents:
                    dest_comp = f'{self.NAME}-{p.agent_id}'
                    with transient_communication(self.discovery, dest_comp, p.agent_id, p.address):
                        self.post_msg(
                            target=dest_comp,
                            msg=ValueMsg(value=self._max),
                        )

                # send position to descendants
                for agt in self._children_temp:
                    dest_comp = f'{self.NAME}-{agt.agent_id}'
                    with transient_communication(self.discovery, dest_comp, agt.agent_id, agt.address):
                        self.post_msg(
                            target=f'{self.NAME}-{agt.agent_id}',
                            msg=PositionMsg(agent_id=self.agent.name, position=self._max),
                        )

    def _clear_temp_connection_variables(self):
        """
        Reset properties used to established connection and variable ordering
        """
        self._neighbors.clear()
        self.agents_in_comm_range.clear()

        # DDFS-specific props
        self._max = 0
        self._num_neighbors_of_neighbor.clear()
        self._parents.clear()
        self._value_msg_senders.clear()
        self._children_temp.clear()
        self._parents_level_dict.clear()

    def _on_position_msg(self, sender: str, msg: PositionMsg):
        self.logger.debug(f'Received position msg from {sender}: {msg}')
        self._parents_level_dict[msg.agent_id] = msg.position

        if len(self._parents_level_dict) == len(self._parents):
            parents = sorted(self._parents, key=lambda p: self._parents_level_dict[p.agent_id])
            self.parent = parents.pop(0)
            self.pseudo_parents = parents

            # send child msg to parent
            dest_comp = f'{self.NAME}-{self.parent.agent_id}'
            with transient_communication(self.discovery, dest_comp, self.parent.agent_id, self.parent.address):
                self.post_msg(
                    target=dest_comp,
                    msg=ChildMsg(agent_id=self.agent.name),
                )
            self._register(self._neighbors[self.parent.agent_id], update_graph=False)
            self.logger.debug(f'Added {self.parent.agent_id} as parent')

            # send pseudo-child messages
            for p in parents:
                self.logger.debug(f'Added {p.agent_id} as pseudo-parent')
                dest_comp = f'{self.NAME}-{p.agent_id}'
                with transient_communication(self.discovery, dest_comp, p.agent_id, p.address):
                    self.post_msg(
                        target=f'{self.NAME}-{p.agent_id}',
                        msg=PseudoChildMsg(agent_id=self.agent.name),
                    )
                self._register(self._neighbors[p.agent_id], update_graph=False)

            if len(self._children_temp) == 0:
                self.execute_computations(exec_order='child-call')

    def _on_pseudo_child_msg(self, sender: str, msg: PseudoChildMsg):
        self.logger.debug(f'Received pseudo-child msg from {sender}')
        pseudo_child = self._neighbors[msg.agent_id]
        self.pseudo_children.append(pseudo_child)
        self._register(pseudo_child)
        self.logger.debug(f'Added {msg.agent_id} as pseudo-child')

    def _on_child_msg(self, sender: str, msg: ChildMsg):
        self.logger.debug(f'Received child msg from {sender}')
        child = self._neighbors[msg.agent_id]
        self.children.append(child)
        self._register(child)
        self.logger.debug(f'Added {msg.agent_id} as child')

    def receive_sim_step_changed(self, sender: str, msg):
        """
        Handles simulation time step changed events.

        Parameters
        ----------
        sender
        msg

        Returns
        -------

        """
        # clear variables used to facilitate DDFS, so they can be reused in next time step
        self._clear_temp_connection_variables()

        self.logger.info(f'Received simulation time step changed: {msg}')
        self.domain = msg.data['agent_domain']
        self.current_position = msg.data['current_position']
        self.neighbor_domains = msg.data['neighbor_domains']

        # initialize dcop algorithm
        self.initialize_computations()

        self.agents_in_comm_range = set(msg.data['agents_in_comm_range'])
        self.post_msg(
            target=ORCHESTRATOR_DIRECTORY,
            msg=SimTimeStepChangeMsgAck(),
        )

        prior_neighbors = set(self.neighbor_ids)

        # remove agents that are out of range
        self.inspect_connections(self.agents_in_comm_range)

        # configuration
        self.logger.debug('configure call in time step changed receiver')
        self.configure_dcop_computation()

        is_affected = prior_neighbors != self.agents_in_comm_range
        self.logger.debug(f'Prior={prior_neighbors}, in-range: {self.agents_in_comm_range}')

        if is_affected:
            self.logger.debug(f'Neighborhood change detected')
            # broadcast connection request
            self.connect()
        else:
            self.logger.debug('No neighborhood change detected')
            self.execute_computations(exec_order='no-new-neighbor')

        # if there are any pending announce msgs process them
        # while True:
        #     try:
        #         sender, msg = self._paused_ann_msgs_queue.get_nowait()
        #         self._receive_announce(sender, msg)
        #     except Empty:
        #         break

