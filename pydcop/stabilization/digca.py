import enum
import random
import threading
import time
from typing import Callable

from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.communication import ComputationMessage, MSG_ALGO, MSG_MGT
from pydcop.infrastructure.computations import MessagePassingComputation
from pydcop.infrastructure.discovery import Discovery, BroadcastMessage
from pydcop.infrastructure.message_types import SimTimeStepChanged, message_type
from pydcop.infrastructure.orchestratedagents import ORCHESTRATOR_MGT, ORCHESTRATOR_DIRECTORY
from pydcop.infrastructure.orchestrator import GraphConnectionMessage
from pydcop.stabilization import Neighbor, Seconds, transient_communication
from pydcop.stabilization.base import DynamicGraphConstructionComputation

NAME = 'DIGCA'


# states
class State(enum.Enum):
    ACTIVE = 1
    INACTIVE = 0


# message types
Announce = message_type(
    'announce',
    fields=['agent_id', 'address', 'comps'],
)

AnnounceResponse = message_type(
    'announce_response',
    fields=['agent_id', 'address', 'num_children'],
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
        self.connect_interval: Seconds = 4
        self._affected = False

        self._max_degree = 3

        self._connect_evt = threading.Event()

        self.keep_alive_agents = []
        self.keep_alive_check_interval: Seconds = 7
        self.keep_alive_msg_interval: Seconds = 5

        self._msg_handlers.update({
            'announce': self._receive_announce,
            'announce_response': self._receive_announce_response,
            'add_me': self._receive_add_me,
            'child_added': self._receive_child_added,
            'parent_assigned': self._receive_parent_assigned,
            'already_active': self._receive_already_active,
            'keep_alive': self._receive_keep_alive,
        })

    def on_start(self):
        super(DIGCA, self).on_start()
        self.logger.debug(f'On start of {self.name}')
        # make a first connect call on startup
        # self._connect()

        # make subsequent (repeated) connect calls in `self.connect_interval` seconds
        # self.add_periodic_action(self.connect_interval, self._connect)

        # start processes for connection maintenance
        # self.add_periodic_action(self.keep_alive_check_interval, self._inspect_connections)
        # self.add_periodic_action(self.keep_alive_msg_interval, self._send_keep_alive_msg)

    def connect(self):
        """
        Broadcasts an Announce message to get a parent agent or connect to another agent.

        Parameters
        ----------
        cb A function to be called after the Connect cycle.
        It takes the execution order to be used as an optional argument.

        Returns
        -------

        """
        if self.state == State.INACTIVE and not self.parent:
            self.logger.debug('DIGCA is connecting...')

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

    def _send_add_me_msg(self, sel_agent):
        """
        select an agent from the list of respondents (if any)
        """
        self.logger.debug(f'Selected agent = {sel_agent}')

        # send add-me msg
        dest_comp = f'{NAME}-{sel_agent.agent_id}'
        with transient_communication(self.discovery, dest_comp, sel_agent.agent_id, sel_agent.address):
            self.post_msg(
                target=dest_comp,
                msg=AddMe(
                    agent_id=self.agent.name,
                    address=self.address,
                    comps=[c.name for c in self.agent.computations()]
                ),
            )

        # update state
        self.state = State.ACTIVE

    def _receive_announce(self, sender: str, msg: Announce):
        if self.state == State.INACTIVE and self._phi(msg.agent_id) and len(self.neighbors) < self._max_degree:
            self.logger.debug(f'Sending announce response to {msg.agent_id}')

            # send add-me msg
            dest_comp = f'{NAME}-{msg.agent_id}'
            with transient_communication(self.discovery, dest_comp, msg.agent_id, msg.address):
                self.post_msg(
                    target=dest_comp,
                    msg=AnnounceResponse(
                        agent_id=self.agent.name,
                        address=self.address,
                        num_children=len(self.children),
                    ),
                )

    def _receive_announce_response(self, sender: str, msg: AnnounceResponse):
        self.logger.debug(f'Received announce response from {sender}: {msg}')
        if self.state == State.INACTIVE and self._assess_potential_neighbor(msg):
            if self.name == 'DIGCA-a2':
                if sender == 'DIGCA-a0':
                    self._send_add_me_msg(msg)
            else:
                self._send_add_me_msg(msg)

    def _assess_potential_neighbor(self, msg: AnnounceResponse):
        """
        Checks if an agent meets the local requirements for connection

        Parameters
        ----------
        msg: AnnounceResponse from agent

        Returns
        -------
        True if agent satisfies requirement else False
        """
        return msg.num_children < self._max_degree

    def _receive_add_me(self, sender: str, msg: AddMe):
        if self.state == State.INACTIVE:
            # add agent to registers
            neighbor = Neighbor(agent_id=msg.agent_id, address=msg.address, computations=msg.comps)
            self.children.append(neighbor)

            # registration and configuration
            self.register_neighbor(neighbor)
            self._affected = True

            # construct and send child-added msg
            self.post_msg(
                target=f'{NAME}-{msg.agent_id}',
                msg=ChildAdded(
                    agent_id=self.agent.name,
                    address=self.address,
                    comps=[c.name for c in self.agent.computations()]
                ),
                on_error='fail',
            )

            # report connection to graph UI
            self.post_msg(
                ORCHESTRATOR_MGT,
                GraphConnectionMessage(
                    action='add',
                    node1=self.agent.name,
                    node2=msg.agent_id,
                ),
                MSG_MGT
            )

        else:
            # send already-active msg
            dest_comp = f'{NAME}-{msg.agent_id}'
            with transient_communication(self.discovery, dest_comp, msg.agent_id, msg.address):
                self.post_msg(
                    target=dest_comp,
                    msg=AlreadyActive(agent_id=self.agent.name, address=self.address),
                )

    def _receive_child_added(self, sender: str, msg: ChildAdded):
        if self.state == State.ACTIVE and not self.parent:
            # assign parent
            self.parent = Neighbor(agent_id=msg.agent_id, address=msg.address, computations=msg.comps)

            # registration and configuration
            self.register_neighbor(self.parent)
            self._affected = True

            # construct and send parent-assigned msg
            # self.post_msg(
            #     target=f'{NAME}-{msg.agent_id}',
            #     msg=ParentAssigned(agent_id=self.agent.name, address=self.address),
            #     on_error='fail',
            # )

            # execute computation if order is bottom-up/async
            self.execute_computations('bottom-up')

            # report connection to graph UI

            self.post_msg(
                ORCHESTRATOR_MGT,
                GraphConnectionMessage(
                    action='add',
                    node1=msg.agent_id,
                    node2=self.agent.name,
                ),
                MSG_MGT,
            )

            self.logger.info(f'Added as child of {msg.agent_id}')

            self.state = State.INACTIVE

    def _receive_already_active(self, sender: str, msg: AlreadyActive):
        self.state = State.INACTIVE

    def _receive_parent_assigned(self, sender: str, msg: ParentAssigned):
        self.logger.info(f'Assigned as parent of {msg.agent_id}')

        # execute computation (if topdown/async)
        self.execute_computations('top-down')

    def _phi(self, agt_id) -> bool:
        """
        Implements phi using index of agents extracted from agent IDs. This function assumes that agents are named
        following the pattern `axx`, where xx indicates a number index (e.g. a0, a1, a54, etc.).

        Parameters
        ----------
        agt_id: str
            The id/name of the agent to be compared with.

        Returns
        -------
            Whether the current agent has a lower rank than the given agent in the global agent ordering.
        """
        cur_agt_index = int(self.agent.name[1:])
        other_agt_index = int(agt_id[1:])
        return cur_agt_index < other_agt_index

    """ Connection stabilization section """

    def _send_keep_alive_msg(self):
        for neighbor in self.neighbors:
            self.logger.debug(f'Sending keep alive to {neighbor.agent_id}')
            # direct communication
            # self.post_msg(
            #     target=f'{NAME}-{neighbor.agent_id}',
            #     msg=KeepAlive(
            #         agent_id=self.agent.name,
            #         address=self.address,
            #     ),
            #     prio=0,
            # )

            # communication via orchestrator (serving as base station)
            self.post_msg(
                target=ORCHESTRATOR_DIRECTORY,
                msg=BroadcastMessage(
                    message=KeepAlive(
                        agent_id=self.agent.name,
                        address=self.address,
                    ),
                    originator=self.name,
                    recipient_prefix=f'{NAME}-{neighbor.agent_id}',
                ),
                prio=0,
            )

    def _receive_keep_alive(self, sender: str, msg: KeepAlive):
        self.logger.debug(f'Received keep alive msg: {msg}, current neighbors: {self.neighbor_ids}')
        if msg.agent_id not in self.keep_alive_agents:
            self.keep_alive_agents.append(msg.agent_id)

    def inspect_connections(self, agents_in_comm_range):
        """
        Removes out-of-range neighbors.
        """
        self.logger.debug(f'Inspecting connections: {agents_in_comm_range}')
        for neighbor in self.neighbors:
            if neighbor.agent_id not in agents_in_comm_range:
                self.logger.debug(f'Agent {neighbor.agent_id}')
                self.unregister_neighbor(neighbor, callback=self._on_neighbor_removed)

    def _on_neighbor_removed(self, neighbor: Neighbor, *args, **kwargs):
        # if parent was removed, set state to in-active to allow a new parent search (connect call).
        if kwargs.get('neighbor_type', None) == 'parent':
            self.state = State.INACTIVE
