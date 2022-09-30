"""
The bridge between DCOP and Robocup 2D Sim
"""
import threading

from pydcop.algorithms.fmddcop import ModelFreeDynamicDCOP
from pydcop.fmddcop.fmddcopagent import OrchestratedFMDDCOPAgent
from pydcop.fmddcop.robocup.players import Player, PLAYER_MAPPING

TEAM_NAME = 'dcop-11'


class RoboCupAgent(OrchestratedFMDDCOPAgent):

    def __init__(self, *args, **kwargs):
        super(RoboCupAgent, self).__init__(*args, **kwargs)

        # assumes agent name/ids are in the form `axx` e.g. a0, a1, etc...
        self._player_no = int(self.name[1:])
        self._player = None

        self._sim_start_thread = threading.Thread(target=self._connect_and_play, daemon=True)

    def _on_start(self):
        r = super()._on_start()
        self.logger.debug('Robocup agent started')
        self._sim_start_thread.start()
        return r

    def _connect_and_play(self):
        self.logger.debug('Waiting for computation to be deployed')
        self.computation_ready_evt.wait()
        self.logger.debug('Starting simulation')
        self._create_player()
        if self._player:
            self._player.connect('localhost', 6000, TEAM_NAME)
            self._player.play()

    def _create_player(self):
        # retrieve computation and create player
        computation = None
        for comp in self.computations():
            if isinstance(comp, ModelFreeDynamicDCOP):
                computation = comp
                break
        if computation is None:
            raise RuntimeError('Robocup simulation requires a model-free D-DCOP')
        self._player: Player = PLAYER_MAPPING[self._player_no](computation)



