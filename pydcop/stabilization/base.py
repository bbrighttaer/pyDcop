import logging
import time
from typing import List, Callable, Union

from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation, Message
from pydcop.infrastructure.discovery import Discovery
from pydcop.stabilization import Neighbor


class DynamicGraphConstructionComputation(MessagePassingComputation):
    """
    This serves as a protocol class for all dynamic graph construction algorithms.
    """

    def __init__(self, algo_name: str, agent: DynamicAgent, discovery: Discovery):
        super(DynamicGraphConstructionComputation, self).__init__(name=f'{algo_name}-{agent.name}')

        self.logger = logging.getLogger(f'pydcop.computation.{self.name}')

        self.discovery = discovery
        self.agent = agent
        self.address = agent.address

        self.parent: Union[Neighbor, None] = None
        self.children: List[Neighbor] = []

        # tracks the last time a neighbor sent a message
        self.last_contact_time = {}

        # added to avoid AttributeError on metric collection
        self.cycle_count = 0

        # supposed to be overridden by subclass to handle messages
        self._msg_handlers = {}

    @property
    def neighbors(self) -> List[Neighbor]:
        nodes = list(self.children)
        if self.parent:
            nodes += [self.parent]
        return nodes

    @property
    def neighbor_ids(self) -> List[str]:
        return [n.agent_id for n in self.neighbors]

    def find_neighbor_by_agent_id(self, agent_id) -> Union[Neighbor, None]:
        for n in self.neighbors:
            if n.agent_id == agent_id:
                return n

    def on_message(self, sender: str, msg: Message, t: float):
        try:
            if hasattr(msg, 'agent_id'):
                self.last_contact_time[msg.agent_id] = time.time()
            self._msg_handlers[msg.type](sender, msg)
        except KeyError:
            self.logger.error(f'Could not find function callback for msg type: {msg.type}')

    def register_neighbor(self, neighbor: Neighbor, callback: Callable = None):
        for comp in neighbor.computations:
            self.discovery.register_computation(
                computation=comp,
                agent=neighbor.agent_id,
                address=neighbor.address,
                publish=False,
            )
        self.logger.debug(f'registered neighbor {neighbor.agent_id}, comps={neighbor.computations}')

        if callback:
            callback(neighbor)

    def unregister_neighbor(self, neighbor: Neighbor, callback: Callable = None):
        self.logger.debug(f'Unregistering neighbor {neighbor.agent_id}')

        if neighbor:
            # unregister agent and computations from discovery
            self.discovery.unregister_agent(neighbor.agent_id, publish=False)

            # remove associations
            if self.parent == neighbor:
                neighbor_type = 'parent'
                self.parent = None
            else:
                neighbor_type = 'child'
                self.children.remove(neighbor)

            # fire callback
            if callback:
                callback(neighbor=neighbor, neighbor_type=neighbor_type)

    def execute_computations(self, exec_order: str = None):
        self.logger.info('Executing agent DCOP computations')
