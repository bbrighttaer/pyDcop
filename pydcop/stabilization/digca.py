from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation, Message, message_type
from pydcop.infrastructure.discovery import Discovery, BroadcastMessage
from pydcop.infrastructure.orchestrator import ORCHESTRATOR
from pydcop.stabilization.protocol import DynamicGraphConstructionComputation

NAME = 'DIGCA'

# states
INACTIVE = 'inactive'
ACTIVE = 'active'

AnnounceMessage = message_type(
    'AnnounceMessage',
    fields=['agent_id', 'address']
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

    def on_start(self):
        self.logger.debug(f'On start of {self.name}')
        self.connect()

    def on_message(self, sender: str, msg: Message, t: float):
        self.logger.debug(f'Received message {msg} from {sender}')

    def connect(self):
        self.logger.debug('DIGCA is connecting...')
        self.post_msg(
            '_discovery_' + ORCHESTRATOR,
            BroadcastMessage(message=AnnounceMessage(
                agent_id=self.agent.name,
                address=self.address
            ),
                recipient_prefix=NAME
            )
        )
