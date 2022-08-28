from collections import namedtuple

Neighbor = namedtuple(
    'Neighbor',
    field_names=['agent_id', 'address', 'computations'],
)
Seconds = int
AgentID = str
MaxDegree = int
