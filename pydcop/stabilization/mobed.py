from collections import defaultdict

from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.communication import MSG_ALGO
from pydcop.infrastructure.computations import MessagePassingComputation
from pydcop.infrastructure.discovery import Discovery, BroadcastMessage, AsyncBroadcastMessage
from pydcop.infrastructure.message_types import SimTimeStepChangeMsgAck, message_type
from pydcop.infrastructure.orchestratedagents import ORCHESTRATOR_DIRECTORY
from pydcop.stabilization import transient_communication
from pydcop.stabilization.ddfs import DistributedDFS

NeighborDiscovery = message_type(
    'neighbor_discovery', fields=['agent_id', 'address'],
)

AddMe = message_type(
    'add_me', fields=['agent_id', 'address', 'comps', 'exp_num_msgs'],
)

ForwardedAddMe = message_type(
    'forwarded_add_me', fields=['message'],
)

AlreadyActive = message_type(
    'already_active', fields=[],
)

NoTree = message_type(
    'no_tree', fields=[],
)

CanAdd = message_type(
    'can_add', fields=['agent_id', 'address'],
)

ConnectionRequest = message_type(
    'connection_request', fields=['message'],
)


def build_stabilization_computation(agent: DynamicAgent, discovery: Discovery) -> MessagePassingComputation:
    return MOBED(MOBED.NAME, agent, discovery)


class MOBED(DistributedDFS):
    """
    Implementation of the HARP procedure
    """

    NAME = 'MOBED'

    def __init__(self, name, agent: DynamicAgent, discovery: Discovery):
        super(MOBED, self).__init__(name, agent, discovery)

        self._already_active = False
        self._connect_called = False
        self._add_msg_dict = defaultdict(int)
        self._forwarded_msg_to_sender = {}

        self._msg_handlers.update({
            'neighbor_discovery': self._on_receive_neighbor_discovery,
            'already_active': self._on_receive_already_active,
            'no_tree': self._on_receive_no_tree,
            'can_add': self._on_receive_can_add,
            'add_me': self._on_receive_add_me,
            'forwarded_add_me': self._on_receive_forwarded_add_me,
        })

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

        # initialize hierarchy construction algorithm
        self._initialize()

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

            if not self.neighbors:
                self._start_mobed()
        else:
            self.logger.debug('No neighborhood change detected')
            self.execute_computations(exec_order='no-new-neighbor')

    def _initialize(self):
        self._already_active = False
        self._connect_called = False
        self._add_msg_dict.clear()
        self._forwarded_msg_to_sender.clear()

    def _start_mobed(self):
        self.logger.debug('Starting MOBED...')

        # publish Announce msg
        self.post_msg(
            target=ORCHESTRATOR_DIRECTORY,
            msg=AsyncBroadcastMessage(message=NeighborDiscovery(
                agent_id=self.agent.name,
                address=self.address,
            ),
                originator=self.name,
                recipient_prefix=self.NAME,
            ),
            prio=MSG_ALGO,
            on_error='fail',
        )

    def _on_receive_neighbor_discovery(self, sender, msg: NeighborDiscovery):
        self.logger.debug(f'Received neighbor-discovery msg from {sender}: {msg}')

        comp_name = f'{self.NAME}-{msg.agent_id}'

        # if addition operation is in progress, inform sender about it
        if self._already_active:
            with transient_communication(self.discovery, comp_name, msg.agent_id, msg.address):
                self.post_msg(
                    target=comp_name,
                    msg=AlreadyActive(),
                )

        # if no tree then trigger DDFS
        elif not self.neighbors:
            with transient_communication(self.discovery, comp_name, msg.agent_id, msg.address):
                self.post_msg(
                    target=comp_name,
                    msg=NoTree(),
                )
            self.connect()

        # can add a_i to existing tree
        else:
            self._start_agent_addition(agent_data=msg)
            with transient_communication(self.discovery, comp_name, msg.agent_id, msg.address):
                self.post_msg(
                    target=comp_name,
                    msg=CanAdd(agent_id=self.agent.name, address=self.address),
                )

    def _start_agent_addition(self, agent_data: AddMe):
        self.logger.debug(f'Adding {agent_data}')

    def _on_receive_already_active(self, sender: str, msg: AlreadyActive):
        self.logger.debug(f'Received already-active msg from {sender}: {msg}')

    def _on_receive_no_tree(self, sender: str, msg: NoTree):
        self.logger.debug(f'Received no-tree msg from {sender}: {msg}')

    def _on_receive_can_add(self, sender: str, msg: CanAdd):
        self.logger.debug(f'Received can-add msg from {sender}: {msg}')

        # respond
        comp_name = f'{self.NAME}-{msg.agent_id}'
        with transient_communication(self.discovery, comp_name, msg.agent_id, msg.address):
            self.post_msg(
                target=comp_name,
                msg=AddMe(
                    agent_id=self.agent.name,
                    address=self.address,
                    comps=[c.name for c in self.agent.computations()],
                    exp_num_msgs=len(self.agents_in_comm_range),
                )
            )

    def _on_receive_add_me(self, sender: str, msg: AddMe):
        self.logger.debug(f'Received add-me msg from {sender}: {msg}')
        self._handle_add_me(msg, sender)

    def _handle_add_me(self, msg, sender):
        # update register
        self._add_msg_dict[sender] += 1
        if self._add_msg_dict[sender] == msg.exp_num_msgs:  # highest node case
            ...
        elif self.parent:
            self.post_msg(
                target=f'{self.NAME}-{self.parent.agent_id}',
                msg=ForwardedAddMe(message=msg),
            )

    def _on_receive_forwarded_add_me(self, sender: str, msg: ForwardedAddMe):
        self.logger.debug(f'Received forwarded-add-me msg from {sender}: {msg}')
        self._handle_add_me(sender=f'{self.NAME}-{msg.message.agent_id}', msg=msg.message)
        self._forwarded_msg_to_sender[msg.message.agent_id] = sender

    def connect(self):
        super(MOBED, self).connect()
        if not self._connect_called:
            self._connect_called = True
