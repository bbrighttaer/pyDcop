import logging

from pydcop.algorithms.fmddcop import ModelFreeDynamicDCOP
from pydcop.fmddcop.robocup.pyrus.pyruslib.player.basic_client import BasicClient
from pydcop.fmddcop.robocup.pyrus.pyruslib.player.player_agent import PlayerAgent


class Player(PlayerAgent):

    def __init__(self, computation: ModelFreeDynamicDCOP = None):
        super(Player, self).__init__()
        self.initial_pos = None
        self._computation: ModelFreeDynamicDCOP = computation

        self._configure_computation()

    def _configure_computation(self):
        if self._computation:
            self.logger = logging.getLogger(f'player-{self._computation.name}')

            # set constraint callbacks of the computation/algorithm
            self._computation.coordination_constraint_cb = self.coordination_constraint
            self._computation.unary_constraint_cb = self.unary_constraint
            self._computation.coordination_data_cb = self.get_coordination_data

    def start(self):
        client = BasicClient()
        self.init(client)
        client.run(agent=self)

    def get_coordination_data(self):
        return {
            'position': self.get_position(),
        }

    def set_computation(self, comp):
        self._computation = comp
        self._configure_computation()

    def coordination_constraint(self, *args, **kwargs):
        return 0

    def unary_constraint(self, *args, **kwargs):
        return 0

    def get_position(self):
        return 0, 0

