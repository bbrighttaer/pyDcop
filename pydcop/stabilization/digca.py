import contextlib
import enum
import random
import threading
import time
from threading import Event

from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.communication import ComputationMessage
from pydcop.infrastructure.computations import MessagePassingComputation, Message, message_type
from pydcop.infrastructure.discovery import Discovery, BroadcastMessage, UnknownComputation
from pydcop.infrastructure.orchestrator import ORCHESTRATOR
from pydcop.stabilization.protocol import DynamicGraphConstructionComputation

NAME = 'DIGCA'


# states
class State(enum.Enum):
    ACTIVE = 1
    INACTIVE = 0


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
        }

    def on_start(self):
        self.logger.debug(f'On start of {self.name}')
        # make a first connect call on startup
        self.connect()

        # make subsequent (repeated) connect calls in `self.interval` seconds
        cancel_connect = self._periodic_action(self.connect_interval, self.connect)
        self._periodic_calls_cancel_list.append(cancel_connect)

    def on_message(self, sender: str, msg: Message, t: float):
        self.logger.debug(f'Received message {msg} from {sender}')

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
        self.logger.debug('DIGCA is connecting...')

        if self.state == State.INACTIVE and not self.parent:
            # publish Announce msg
            self.post_msg(
                '_discovery_' + ORCHESTRATOR,
                BroadcastMessage(message=Announce(
                    agent_id=self.agent.name,
                    address=self.address
                ),
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
                msg = AddMe(agent_id=self.agent.name, address=self.address)
                full_msg = ComputationMessage(
                    src_comp=self.name,
                    dest_comp=f'{NAME}-{msg.agent_id}',
                    msg=msg,
                    msg_type=msg.type
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
            msg = AnnounceResponse(agent_id=self.agent.name, address=self.address, )
            full_msg = ComputationMessage(
                src_comp=self.name,
                dest_comp=f'{NAME}-{msg.agent_id}',
                msg=msg,
                msg_type=msg.type,
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


    def _phi(self, agt_id):
        return self.agent.name < agt_id


    def on_stop(self):
        for cancel in self._periodic_calls_cancel_list:
            cancel()
