import logging
from typing import List

from pydcop.infrastructure.agents import DynamicAgent
from pydcop.infrastructure.computations import MessagePassingComputation
from pydcop.infrastructure.discovery import Discovery


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
        self._dcop_comps = []

        self.parent = None
        self.children = []

        # added to avoid AttributeError on metric collection
        self.cycle_count = 0

    @property
    def dcop_computations(self) -> List[MessagePassingComputation]:
        return self._dcop_comps

    @dcop_computations.setter
    def dcop_computations(self, comp):
        self.logger.debug(f'Setting DCOP computation: {str(comp)}')
        self._dcop_comps = comp

