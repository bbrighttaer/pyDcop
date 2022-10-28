"""
The bridge between DCOP and Robocup 2D Sim
"""
import threading
import time

from pydcop.algorithms.fmddcop import ModelFreeDynamicDCOP
from pydcop.fmddcop.fmddcopagent import OrchestratedFMDDCOPAgent
from pydcop.fmddcop.robocup.pyrus.player import Player
from pydcop.fmddcop.robocup.soccerpy.players import SOCCER_PY_PLAYER_MAPPING

PLAYERS = []


def start_team():
    for _ in range(11):
        threading.Thread(target=_start_player, daemon=True).start()
        time.sleep(0.01)


def _start_player():
    player = Player()
    PLAYERS.append(player)
    player.start()


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
        # self._create_soccerpy_player()
        self._create_pyrus_player()
        if self._player:
            self._player.start()

    def _get_computation(self):
        computation = None
        for comp in self.computations():
            if isinstance(comp, ModelFreeDynamicDCOP):
                computation = comp
                break
        return computation

    def _create_soccerpy_player(self):
        # retrieve computation and create player
        computation = self._get_computation()
        assert computation, 'Robocup simulation requires a model-free D-DCOP'
        self._player = SOCCER_PY_PLAYER_MAPPING[self._player_no](computation)

    def _create_pyrus_player(self):
        computation = self._get_computation()
        assert computation, 'Robocup simulation requires a model-free D-DCOP'
        self._player = None
        for player in PLAYERS:
            if player.world().self_unum() == self._player_no:
                self._player = player
                break
        if self._player:
            self._player.set_computation(computation)
            self.logger.debug(f'Computation for {self._player_no} set successfully')
        else:
            self.logger.debug(f'Soccer player {self._player_no} not found')
