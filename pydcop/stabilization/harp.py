from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation, message_type
from pydcop.infrastructure.discovery import Discovery
from pydcop.stabilization.ddfs import DistributedDFS

NAME = 'HARP'


Start = message_type(
    'start',
    fields=['agent_id'],
)

Query = message_type(
    'query',
    fields=['agent_id'],
)

Response = message_type(
    'response',
    fields=['agent_id', 'am_affected'],
)

PseudoId = message_type(
    'pseudo_id',
    fields=['agent_id', 'pseudo_id', 'am_affected'],
)

PseudoIdAck = message_type(
    'pseudo_id_ack',
    fields=['agent_id'],
)

Constraint = message_type(
    'constraint',
    fields=['agent_id', 'sep'],
)

ConstraintAck = message_type(
    'constraint_ack',
    fields=['agent_id'],
)


def build_stabilization_computation(agent: DynamicAgent, discovery: Discovery) -> MessagePassingComputation:
    return HARP(NAME, agent, discovery)


class HARP(DistributedDFS):
    """
    Implementation of the HARP procedure
    """

    def _split(self):
        self.logger.debug(f'Starting HARP procedure')



