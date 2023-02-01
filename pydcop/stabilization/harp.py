from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation
from pydcop.infrastructure.discovery import Discovery
from pydcop.infrastructure.message_types import SimTimeStepChanged, SimTimeStepChangeMsgAck, message_type
from pydcop.infrastructure.orchestratedagents import ORCHESTRATOR_DIRECTORY
from pydcop.stabilization.ddfs import DistributedDFS

AnnounceResponse = message_type(
    'announce_response',
    fields=['agent_id', 'address', 'comps'],
)

Start = message_type(
    'start', fields=[],
)

Query = message_type(
    'query', fields=[],
)

Response = message_type(
    'response', fields=['agent_id', 'is_affected']
)

PseudoID = message_type(
    'pseudo_id', fields=['agent_id', 'pseudo_id', 'is_affected'],
)

PseudoIDAck = message_type(
    'pseudo_id-ack', fields=[],
)

Stop = message_type(
    'stop', fields=[],
)

Separator = message_type(
    'separator', fields=['sep'],
)

Constraint = message_type(
    'constraint', fields=['agent_id', 'sep']
)


def build_stabilization_computation(agent: DynamicAgent, discovery: Discovery) -> MessagePassingComputation:
    return HARP(HARP.NAME, agent, discovery)


class HARP(DistributedDFS):
    """
    Implementation of the HARP procedure
    """

    NAME = 'HARP'

    def __init__(self, name, agent: DynamicAgent, discovery: Discovery):
        super(HARP, self).__init__(name, agent, discovery)
        self._started_harp = False
        self._separator_dict = {}
        self._is_affected = False
        self._sent_start = False
        self._response_list = []
        self._pseudo_id_ack_list = []
        self._pseudo_id = self.agent.name

        self._msg_handlers.update({
            'start': self._on_receive_start,
            'query': self._on_receive_query,
            'response': self._on_receive_response,
            'pseudo_id': self._on_receive_pseudo_id,
            'pseudo_id-ack': self._on_receive_pseudo_id_ack,
            'stop': self._on_receive_stop,
            'separator': self._on_receive_separator,
            'constraint': self._on_receive_constraint,
        })

    def receive_sim_step_changed(self, sender: str, msg: SimTimeStepChanged):
        self.logger.info(f'Received simulation time step changed: {msg}')
        self.domain = msg.data['agent_domain']
        self.current_position = msg.data['current_position']
        self.neighbor_domains = msg.data['neighbor_domains']

        self._initialize_graph_alg_props()

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

        self._is_affected = prior_neighbors != self.agents_in_comm_range
        self.logger.debug(f'Prior={prior_neighbors}, in-range: {self.agents_in_comm_range}')

        if self._is_affected:
            self.logger.debug('Neighborhood change detected')
            if not self.neighbors:
                # broadcast connection request
                self.connect()
            else:
                self.start_harp()
        else:
            self.logger.debug('No neighborhood change detected')
            self.execute_computations(exec_order='no-new-neighbor')

    def _initialize_graph_alg_props(self):
        self._sent_start = False
        self._response_list.clear()
        self._pseudo_id_ack_list.clear()
        self._separator_dict.clear()
        self._started_harp = False

    def start_harp(self):
        self.logger.debug('Starting HARP procedure...')
        self._started_harp = True

        if self._is_affected and not self._sent_start:
            self._sent_start = True
            self._send_start_and_query_msgs()

    def _send_start_and_query_msgs(self):
        if self.parent:
            self.post_msg(
                target=self.parent,
                msg=Start(),
            )
        # root case
        if self.neighbors and not self.parent:
            for child in self.children:
                self.post_msg(
                    target=f'{self.NAME}-{child.agent_id}',
                    msg=Query(),
                )

    def _on_receive_start(self, sender, msg):
        self.logger.debug(f'Received start from {sender}')
        if not self._sent_start:
            self._sent_start = True
            self._send_start_and_query_msgs()

    def _on_receive_query(self, sender, msg):
        self.logger.debug(f'Received query from {sender}')

        # forward to children
        for child in self.children:
            self.post_msg(
                target=f'{self.NAME}-{child.agent_id}',
                msg=Query(),
            )

        # leaf case
        if self.parent and not self.children:
            self.post_msg(
                target=f'{self.NAME}-{self.parent.agent_id}',
                msg=Response(agent_id=self.agent.name, is_affected=self._is_affected),
            )

    def _on_receive_response(self, sender, msg):
        self._response_list.append(msg)

        if msg.is_affected:
            self._is_affected = True

        if len(self._response_list) == len(self.children):
            # send response msg to parent
            if self.parent:
                self.post_msg(
                    target=f'{self.NAME}-{self.parent.agent_id}',
                    msg=Response(agent_id=self.agent.name, is_affected=self._is_affected),
                )

            # send pseudo-id to children
            for child in self.children:
                self.post_msg(
                    target=f'{self.NAME}-{child.agent_id}',
                    msg=PseudoID(agent_id=self.agent.name, pseudo_id=self._pseudo_id, is_affected=self._is_affected),
                )

    def _on_receive_pseudo_id(self, sender, msg):
        if not self._is_affected and msg.is_affected:
            sep = set(self._separator_dict.values())
            for ancestor in sep:
                self.post_msg(

                )
        else:
            if not msg.is_affected:
                self.pseudo_id = msg.pseudo_id

            # send pseudo-id to children
            for child in self.children:
                self.post_msg(
                    target=f'{self.NAME}-{child.agent_id}',
                    msg=PseudoID(agent_id=self.agent.name, pseudo_id=self._pseudo_id,
                                 is_affected=self._is_affected),
                )

            # leaf case
            if self.parent and not self.children:
                # send pseudo-ack to parent
                self.post_msg(
                    target=f'{self.NAME}-{self.parent.agent_id}',
                    msg=PseudoIDAck(),
                )

    def _on_receive_pseudo_id_ack(self, sender, msg):
        self._pseudo_id_ack_list.append(msg)

        if len(self.children) == len(self._pseudo_id_ack_list):
            # send pseudo-ack to parent
            self.post_msg(
                target=f'{self.NAME}-{self.parent.agent_id}',
                msg=PseudoIDAck(),
            )

        # root case
        if not self.parent and self.neighbors:
            for child in self.children:
                self.post_msg(
                    target=f'{self.NAME}-{child.agent_id}',
                    msg=Stop(),
                )

    def _on_receive_constraint(self, sender, msg):
        ...

    def _on_receive_constraint_ack(self, sender, msg):
        ...

    def _on_receive_stop(self, sender, msg):
        ...

    def _on_position_msg(self, sender: str, msg):
        super(HARP, self)._on_position_msg(sender, msg)

        # if pseudo-parent(s) is/are present then report to parent
        if self.pseudo_parents:
            self.post_msg(
                target=f'{self.NAME}-{self.parent.agent_id}',
                msg=Separator(pseudo_parents=self.pseudo_parents),
            )

    def _on_receive_pseudo_parents(self, sender, msg: Separator):
        self.logger.debug(f'Received pseudo-parents message from {sender}: {msg}')
        self._separator_dict[sender] = msg.pseudo_parents

        if self.parent:
            self.post_msg(
                target=f'{self.NAME}-{self.parent.agent_id}',
                msg=Separator(sep=self.pseudo_parents + msg.sep),
            )

    def _on_receive_separator(self, sender, msg):
        self.logger.debug(f'Received separator msg from {sender}: {msg}')




