import enum
import random
import time
from queue import Queue, Empty
from threading import Event
from typing import List, Dict

from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.communication import ComputationMessage, MSG_ALGO
from pydcop.infrastructure.computations import MessagePassingComputation, message_type
from pydcop.infrastructure.discovery import Discovery, BroadcastMessage
from pydcop.infrastructure.orchestrator import ORCHESTRATOR
from pydcop.stabilization.base import DynamicGraphConstructionComputation, Neighbor

NAME = 'DIGCA'

Seconds = int
AgentID = str


# states
class State(enum.Enum):
    ACTIVE = 1
    INACTIVE = 0


# message types
Announce = message_type(
    'announce',
    fields=['agent_id', 'address'],
)

AnnounceResponse = message_type(
    'announce_response',
    fields=['agent_id', 'address'],
)

AddMe = message_type(
    'add_me',
    fields=['agent_id', 'address', 'comps'],
)

ChildAdded = message_type(
    'child_added',
    fields=['agent_id', 'address', 'comps'],
)

AlreadyActive = message_type(
    'already_active',
    fields=['agent_id', 'address'],
)

ParentAssigned = message_type(
    'parent_assigned',
    fields=['agent_id', 'address'],
)

Ping = message_type(
    'ping',
    fields=['agent_id', 'address'],
)

PingResponse = message_type(
    'ping_response',
    fields=['agent_id', 'address'],
)

KeepAlive = message_type(
    'keep_alive',
    fields=['agent_id', 'address'],
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
    return DIGCA(agent, discovery)


class DIGCA(DynamicGraphConstructionComputation):
    """
    Implements the Dynamic Interaction Graph Construction Algorithm.
    """

    def __init__(self, agent: DynamicAgent, discovery: Discovery):
        super(DIGCA, self).__init__(NAME, agent, discovery)

        self.state = State.INACTIVE
        self.announce_response_list = []
        self.connect_interval: Seconds = 5
        self.announce_response_listening_time: Seconds = 1

        self.keep_alive_msg_queue = []
        self.keep_alive_check_interval: Seconds = 5
        self.keep_alive_msg_interval: Seconds = 2

        self._msg_handlers = {
            'announce': self._receive_announce,
            'announce_response': self._receive_announce_response,
            'add_me': self._receive_add_me,
            'child_added': self._receive_child_added,
            'parent_assigned': self._receive_parent_assigned,
            'already_active': self._receive_already_active,
            'keep_alive': self._receive_keep_alive,
        }

    def on_start(self):
        self.logger.debug(f'On start of {self.name}')
        # make a first connect call on startup
        self._connect()

        # make subsequent (repeated) connect calls in `self.connect_interval` seconds
        self.add_periodic_action(self.connect_interval, self._connect)

        # start processes for connection maintenance
        self.add_periodic_action(self.keep_alive_check_interval, self._inspect_connections)
        self.add_periodic_action(self.keep_alive_msg_interval, self._send_keep_alive_msg)

    def _connect(self):
        if self.state == State.INACTIVE and not self.parent:
            self.logger.debug('DIGCA is connecting...')

            # publish Announce msg
            self.post_msg(
                '_discovery_' + ORCHESTRATOR,
                BroadcastMessage(message=Announce(
                    agent_id=self.agent.name,
                    address=self.address
                ),
                    originator=self.name,
                    recipient_prefix=NAME
                )
            )

            # wait for AnnounceResponses from available agents
            time.sleep(self.announce_response_listening_time)

            # select an agent from the list of respondents (if any)
            # self.logger.debug(f'Selecting from announce response list: {self.announce_response_list}')
            if self.announce_response_list:
                selected_agent = random.choice(self.announce_response_list)

                # construct add-me msg
                full_msg = ComputationMessage(
                    src_comp=self.name,
                    dest_comp=f'{NAME}-{selected_agent.agent_id}',
                    msg=AddMe(
                        agent_id=self.agent.name,
                        address=self.address,
                        comps=[c.name for c in self.computations] + [self.name]
                    ),
                    msg_type=MSG_ALGO,
                )

                # send add-me msg
                selected_agent.address.receive_msg(
                    src_agent=self.agent.name,
                    dest_agent=selected_agent.agent_id,
                    msg=full_msg,
                )

                # update state
                self.state = State.ACTIVE

            # clear announce response list
            self.announce_response_list.clear()

    def _receive_announce(self, sender: str, msg: Announce):
        if self.state == State.INACTIVE and self._phi(msg.agent_id):
            self.logger.debug(f'Sending announce response to {msg.agent_id}')

            # construct announce response msg
            full_msg = ComputationMessage(
                src_comp=self.name,
                dest_comp=f'{NAME}-{msg.agent_id}',
                msg=AnnounceResponse(agent_id=self.agent.name, address=self.address),
                msg_type=MSG_ALGO,
            )

            # send announce response msg
            msg.address.receive_msg(
                src_agent=self.agent.name,
                dest_agent=msg.agent_id,
                msg=full_msg,
            )

    def _receive_announce_response(self, sender: str, msg: AnnounceResponse):
        if self.state == State.INACTIVE:
            self.announce_response_list.append(msg)

    def _receive_add_me(self, sender: str, msg: AddMe):
        if self.state == State.INACTIVE:
            # add agent to registers
            neighbor = Neighbor(agent_id=msg.agent_id, address=msg.address, computations=msg.comps)
            self.children.append(neighbor)

            # registration and configuration
            self.register_neighbor(neighbor)

            # construct child-added msg
            full_msg = ComputationMessage(
                src_comp=self.name,
                dest_comp=f'{NAME}-{msg.agent_id}',
                msg=ChildAdded(
                    agent_id=self.agent.name,
                    address=self.address,
                    comps=[c.name for c in self.computations] + [self.name]
                ),
                msg_type=MSG_ALGO,
            )

            # send child-added msg
            msg.address.receive_msg(
                src_agent=self.agent.name,
                dest_agent=msg.agent_id,
                msg=full_msg,
            )

        else:
            # construct already-active msg
            full_msg = ComputationMessage(
                src_comp=self.name,
                dest_comp=f'{NAME}-{msg.agent_id}',
                msg=AlreadyActive(agent_id=self.agent.name, address=self.address),
                msg_type=MSG_ALGO,
            )

            # send already-active msg
            msg.address.receive_msg(
                src_agent=self.agent.name,
                dest_agent=msg.agent_id,
                msg=full_msg,
            )

    def _receive_child_added(self, sender: str, msg: ChildAdded):
        if self.state == State.ACTIVE and not self.parent:
            # assign parent
            self.parent = Neighbor(agent_id=msg.agent_id, address=msg.address, computations=msg.comps)

            # registration and configuration
            self.register_neighbor(self.parent)

            # construct parent-assigned msg
            full_msg = ComputationMessage(
                src_comp=self.name,
                dest_comp=f'{NAME}-{msg.agent_id}',
                msg=ParentAssigned(agent_id=self.agent.name, address=self.address),
                msg_type=MSG_ALGO,
            )

            # send parent-assigned msg
            msg.address.receive_msg(
                src_agent=self.agent.name,
                dest_agent=msg.agent_id,
                msg=full_msg,
            )

            # execute computation if order is bottom-up
            self._execute_computations('bottom-up')

            # report connection to graph UI

            self.logger.info(f'Added as child of {msg.agent_id}')

    def _receive_already_active(self, sender: str, msg: AlreadyActive):
        self.state = State.INACTIVE

    def _receive_parent_assigned(self, sender: str, msg: ParentAssigned):
        self.logger.info(f'Assigned as parent of {msg.agent_id}')

        # execute computation (if topdown/async)
        self._execute_computations('top-down')

    def _phi(self, agt_id):
        return self.agent.name < agt_id

    """ Connection stabilization section """

    def _send_keep_alive_msg(self):
        for neighbor in self.neighbors:
            self.logger.debug(f'Sending keep alive to {neighbor.agent_id}')
            self.post_msg(
                target=f'{NAME}-{neighbor.agent_id}',
                msg=KeepAlive(
                    agent_id=self.agent.name,
                    address=self.address,
                ),
                prio=0,
            )

    def _receive_keep_alive(self, sender: str, msg: KeepAlive):
        self.logger.debug(f'Received keep alive msg: {msg}')
        if msg.agent_id not in self.keep_alive_msg_queue:
            self.keep_alive_msg_queue.append(msg.agent_id)

    def _inspect_connections(self):
        """
        Checks the keep alive queue and remove connections deemed stale.
        """
        self.logger.debug(f'Inspecting connections: {self.keep_alive_msg_queue}')
        affected = False
        for neighbor in self.neighbors:
            if neighbor.agent_id not in self.keep_alive_msg_queue:
                self.logger.debug(f'did not hear from {neighbor.agent_id}')
                self.unregister_neighbor(neighbor, callback=self._on_neighbor_removed)

        if affected:
            self.configure_computations()
            self._execute_computations(is_reconfiguration=True)
        self.keep_alive_msg_queue.clear()

    def _on_neighbor_removed(self, neighbor: Neighbor, *args, **kwargs):
        # if parent was removed, set state to in-active to allow a new parent search (connect call).
        if kwargs.get('neighbor_type', None) == 'parent':
            self.state = State.INACTIVE
