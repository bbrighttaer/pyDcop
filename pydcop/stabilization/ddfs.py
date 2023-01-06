import threading
from queue import Queue

from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.communication import MSG_ALGO, MSG_MGT
from pydcop.infrastructure.computations import MessagePassingComputation
from pydcop.infrastructure.discovery import Discovery, BroadcastMessage
from pydcop.infrastructure.message_types import message_type, GraphConnectionMessage
from pydcop.infrastructure.orchestratedagents import ORCHESTRATOR_DIRECTORY, ORCHESTRATOR_MGT
from pydcop.stabilization import Neighbor, transient_communication
from pydcop.stabilization.base import DynamicGraphConstructionComputation

NAME = 'DDFS'

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
    fields=['agent_id', 'num_neighbors'],
)

PositionMsg = message_type(
    'position_msg',
    fields=['agent_id', 'position'],
)

ValueMsg = message_type(
    'value_msg',
    fields=['value'],
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
    return DistributedDFS(NAME, agent, discovery)


class DistributedDFS(DynamicGraphConstructionComputation):
    """
    Implementation of the DDFS algorithm for dynamic DCOP
    """

    def __init__(self, name, agent: DynamicAgent, discovery: Discovery):
        super(DistributedDFS, self).__init__(name, agent, discovery)

        self._neighbors = {}
        self._num_neighbors_of_neighbor = {}
        self._value_msg_senders = []
        self._parents_level = {}

        self._max = 0
        self._parents = []
        self.pseudo_parents = []
        self.pseudo_children = []

        # props for handling paused msgs
        self._paused_msgs_queue = Queue()
        self._paused_msgs_thread = threading.Thread(target=self._paused_msg_handler, daemon=True)

        self._msg_handlers.update({
            'announce': self._receive_announce,
            'neighbor_data': self._on_receive_neighbor_data,
            # 'neighbor_data_response': self._on_receive_neighbor_data_response,
            # 'position_msg': self._on_position_msg,
            # 'value_msg': self._on_value_msg,
            # 'pseudo_child_msg': self._on_pseudo_child_msg,
        })

    def on_start(self):
        super(DistributedDFS, self).on_start()
        self.logger.debug(f'On start of {self.name}')
        self._paused_msgs_thread.start()

    def _paused_msg_handler(self):
        while True:
            func, sender, msg = self._paused_msgs_queue.get()
            # self.logger.debug(f'Paused msgs: calling {func} with args [sender={sender}, msg={msg}]')
            func(sender, msg)

    def connect(self):
        self.logger.debug(f'{self.name} connect call')

        # publish Announce msg
        self.post_msg(
            target=ORCHESTRATOR_DIRECTORY,
            msg=BroadcastMessage(message=Announce(
                agent_id=self.agent.name,
                address=self.address,
                comps=[c.name for c in self.agent.computations()]
            ),
                originator=self.name,
                recipient_prefix=NAME
            ),
            prio=MSG_ALGO,
            on_error='fail',
        )

    def _receive_announce(self, sender: str, msg: Announce):
        self.logger.debug(f'Received announce msg from {sender}: {msg}')

        # record neighbor information
        if msg.agent_id not in self._neighbors:
            self._neighbors[msg.agent_id] = Neighbor(
                agent_id=msg.agent_id,
                address=msg.address,
                computations=msg.comps,
            )

            # if all neighbors have responded, request neighbor information
            if len(self.agents_in_comm_range) == len(self._neighbors):
                self.logger.debug(f'number of neighbors = {len(self._neighbors)}')
                for agent_id in self._neighbors:
                    dest_comp = f'{NAME}-{agent_id}'
                    n = self._neighbors[agent_id]
                    with transient_communication(self.discovery, dest_comp, n.agent_id, n.address):
                        self.post_msg(
                            target=dest_comp,
                            msg=NeighborData(agent_id=self.agent.name, num_neighbors=len(self._neighbors)),
                        )
            else:
                self._paused_msgs_queue.put((self._receive_announce, sender, msg))
                self._neighbors.pop(msg.agent_id)

    # def _on_receive_neighbor_data_request(self, sender: str, msg: NeighborDataRequest):
    #     self.logger.debug(f'Received neighbor data request from {sender}: {msg}')
    #     with transient_communication(self.discovery, sender, msg.agent_id, msg.address):
    #         self.post_msg(
    #             target=sender,
    #             msg=NeighborDataResponse(num_neighbors=len(self._neighbors))
    #         )

    def _on_receive_neighbor_data(self, sender: str, msg: NeighborData):
        self.logger.debug(f'Received neighbor data from {sender}: {msg}')
        self._num_neighbors_of_neighbor[msg.agent_id] = msg.num_neighbors

        # perform max-degree splitting of neighbors
        if len(self._neighbors) == len(self._num_neighbors_of_neighbor):
            self.logger.debug('Splitting neighbors')
            # self._split_neighbors()
        else:
            self._paused_msgs_queue.put((self._on_receive_neighbor_data, sender, msg))

    def on_neighbor_removed(self, neighbor: Neighbor, *args, **kwargs):
        self._neighbors.pop(neighbor.agent_id)

    def _split_neighbors(self):
        """
        split neighbors into children and parents
        """
        # base class props
        self.parent = None
        self.children.clear()
        self.pseudo_parents.clear()
        self.pseudo_children.clear()

        for agt, num_neighbors in self._num_neighbors_of_neighbor.items():
            neighbor = self._neighbors[agt]

            if num_neighbors < len(self._neighbors):
                self.children.append(neighbor)
            else:
                self._parents.append(neighbor)

        # if this agent is a leaf, begin level calculation for ordering parents
        if len(self.children) == 0 and self._parents:
            self._max += 1
            for p in self._parents:
                dest_comp = f'{NAME}-{p.agent_id}'
                with transient_communication(self.discovery, dest_comp, p.agent_id, p.address):
                    self.post_msg(
                        target=dest_comp,
                        msg=ValueMsg(value=self._max),
                    )

    def _register(self, neighbor: Neighbor):
        # registration and configuration
        self.register_neighbor(neighbor)

        # report connection to graph UI
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

            if len(self._value_msg_senders) == len(self.children):
                self._max += 1
                if self._parents:  # is not root
                    for p in self._parents:
                        dest_comp = f'{NAME}-{p.agent_id}'
                        with transient_communication(self.discovery, dest_comp, p.agent_id, p.address):
                            self.post_msg(
                                target=dest_comp,
                                msg=ValueMsg(value=self._max),
                            )
                else:  # is root
                    for agt in self.children:
                        dest_comp = f'{NAME}-{agt.agent_id}'
                        with transient_communication(self.discovery, dest_comp, agt.agent_id, agt.address):
                            self.post_msg(
                                target=f'{NAME}-{agt.agent_id}',
                                msg=PositionMsg(agent_id=self.agent.name, position=self._max),
                            )

                # --------------------------------------------------------------------- #
                # reset properties used to established connection and variable ordering
                # --------------------------------------------------------------------- #

                self.agents_in_comm_range.clear()

                # DDFS-specific props
                self._neighbors.clear()
                self._num_neighbors_of_neighbor.clear()
                self._parents.clear()
                self._value_msg_senders.clear()

    def _on_position_msg(self, sender: str, msg: PositionMsg):
        self.logger.debug(f'Received position msg from {sender}: {msg}')
        self._parents_level[msg.agent_id] = msg.position

        if len(self._parents_level) == len(self._parents):
            parents = sorted(self._parents, key=lambda p: self._parents_level[p.agent_id])
            self.parent = parents.pop(0)
            self.pseudo_parents = parents

            # send pseudo-child messages
            for p in parents:
                dest_comp = f'{NAME}-{p.agent_id}'
                with transient_communication(self.discovery, dest_comp, p.agent_id, p.address):
                    self.post_msg(
                        target=f'{NAME}-{p.agent_id}',
                        msg=PseudoChildMsg(agent_id=self.agent.name),
                    )

    def _on_pseudo_child_msg(self, sender: str, msg: PseudoChildMsg):
        self.logger.debug(f'Received pseudo-child msg from {sender}')
        self.pseudo_children.append(msg.agent_id)


