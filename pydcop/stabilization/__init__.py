from collections import namedtuple

from pydcop.infrastructure.discovery import Discovery

Neighbor = namedtuple(
    'Neighbor',
    field_names=['agent_id', 'address', 'computations'],
)
Seconds = int
AgentID = str
MaxDegree = int


class transient_communication:
    """
    Facilitates sending a message to an agent that is not a neighbor yet.
    It temporarily registers the agent in the discovery (without publishing to orchestrator)
    and then unregister the agent details, since it's not a neighbor yet.
    The temporary registration is necessary for the communication layer to identify the address
    of the agent.
    """

    def __init__(self, discovery: Discovery, agent: AgentID, agent_address):
        self._discovery = discovery
        self._agent = agent
        self._agent_address = agent_address

    def __enter__(self):
        self._discovery.register_agent(self._agent, self._agent_address, publish=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._discovery.unregister_agent(self._agent, publish=False)
