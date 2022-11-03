"""
The bridge between DCOP and Robocup 2D Sim
"""
import threading

from pydcop.algorithms.fmddcop import ModelFreeDynamicDCOP
from pydcop.fmddcop.fmddcopagent import OrchestratedFMDDCOPAgent

COMPUTATIONS = {}


class RoboCupAgent(OrchestratedFMDDCOPAgent):

    def __init__(self, *args, **kwargs):
        super(RoboCupAgent, self).__init__(*args, **kwargs)

        # assumes agent name/ids are in the form `axx` e.g. a0, a1, etc...
        self._player_no = int(self.name[1:])

        self._sim_start_thread = threading.Thread(target=self._wait_for_computation_setup, daemon=True)

    def _on_start(self):
        r = super()._on_start()
        self.logger.debug('Robocup agent started')
        self._sim_start_thread.start()
        return r

    def _wait_for_computation_setup(self):
        self.logger.debug('Waiting for computation to be deployed')
        self.computation_ready_evt.wait()
        self.logger.debug('Starting simulation')
        COMPUTATIONS[self._player_no] = self._get_computation()

    def _get_computation(self):
        computation = None
        for comp in self.computations():
            if isinstance(comp, ModelFreeDynamicDCOP):
                computation = comp
                break
        return computation
