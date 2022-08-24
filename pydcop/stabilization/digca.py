import contextlib
import enum
import random
import threading
import time
from collections import namedtuple
from threading import Event

from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.communication import ComputationMessage, MSG_ALGO
from pydcop.infrastructure.computations import MessagePassingComputation, Message, message_type
from pydcop.infrastructure.discovery import Discovery, BroadcastMessage
from pydcop.infrastructure.orchestrator import ORCHESTRATOR
from pydcop.stabilization.protocol import DynamicGraphConstructionComputation

NAME = 'DIGCA'


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
    fields=['agent_id', 'address'],
)

ChildAdded = message_type(
    'child_added',
    fields=['agent_id', 'address'],
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

# neighbor types
Parent = namedtuple('Parent', field_names=['agent_id', 'address'])
Child = namedtuple('Child', field_names=['agent_id', 'address'])


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
        self.connect_interval = 5  # in seconds
        self.announce_response_listening_time = 1  # in seconds

        self._periodic_calls_cancel_list = []

        self._msg_handlers = {
            'announce': self._receive_announce,
            'announce_response': self._receive_announce_response,
            'add_me': self._receive_add_me,
            'child_added': self._receive_child_added,
            'parent_assigned': self._receive_parent_assigned,
            'already_active': self._receive_already_active,
            'ping': self._receive_ping,
            'ping_response': self._receive_ping_response,
        }

    def on_start(self):
        self.logger.debug(f'On start of {self.name}')
        # make a first connect call on startup
        self.connect()

        # make subsequent (repeated) connect calls in `self.connect_interval` seconds
        cancel_connect = self._periodic_action(self.connect_interval, self.connect)
        self._periodic_calls_cancel_list.append(cancel_connect)

    def on_message(self, sender: str, msg: Message, t: float):
        try:
            self._msg_handlers[msg.type](sender, msg)
        except KeyError:
            self.logger.error(f'Could not find function callback for msg type: {msg.type}')

    def _periodic_action(self, interval: int, func, *args, **kwargs):
        stopped = Event()

        def loop():
            while not self.agent._stopping.is_set() and not stopped.wait(interval):
                func(*args, **kwargs)

        threading.Thread(target=loop).start()
        return stopped.set

    def connect(self):
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
            self.logger.debug(f'Selecting from announce response list: {self.announce_response_list}')
            if self.announce_response_list:
                selected_agent = random.choice(self.announce_response_list)

                # construct add-me msg
                full_msg = ComputationMessage(
                    src_comp=self.name,
                    dest_comp=f'{NAME}-{selected_agent.agent_id}',
                    msg=AddMe(agent_id=self.agent.name, address=self.address),
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
            self.children.append(Child(agent_id=msg.agent_id, address=msg.address))
            self.discovery.register_agent(msg.agent_id, msg.address, publish=False)

            # construct child-added msg
            full_msg = ComputationMessage(
                src_comp=self.name,
                dest_comp=f'{NAME}-{msg.agent_id}',
                msg=ChildAdded(agent_id=self.agent.name, address=self.address),
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
            self.parent = Parent(agent_id=msg.agent_id, address=msg.address)
            self.discovery.register_agent(msg.agent_id, msg.address, publish=False)

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

            # dcop execution if order is bottom-up

            # report connection to graph UI

            self.logger.info(f'Added as child of {msg.agent_id}')

    def _receive_already_active(self, sender: str, msg: AlreadyActive):
        self.state = State.INACTIVE

    def _receive_parent_assigned(self, sender: str, msg: ParentAssigned):
        self.logger.info(f'Assigned as parent of {msg.agent_id}')

        # dcop execution if order is top-down

    def _receive_ping(self, sender: str, msg: Ping):
        ...

    def _receive_ping_response(self, sender: str, msg: PingResponse):
        ...

    def _phi(self, agt_id):
        return self.agent.name < agt_id

    def on_stop(self):
        for cancel in self._periodic_calls_cancel_list:
            cancel()
