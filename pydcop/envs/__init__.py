from typing import List

from pydcop.infrastructure.computations import MessagePassingComputation


class SimulationEnvironment(MessagePassingComputation):
    """
    Base class for all simulation environments used for D-DCOP
    """

    def __init__(self, name):
        super(SimulationEnvironment, self).__init__(name)
        self._state_history: List[TimeStep] = []

    def step(self):
        ...

    def display(self):
        ...

    @property
    def history(self):
        return self._state_history


class TimeStep:
    """
    Models a single time step of a simulation
    """

    def __init__(self, step_i, *args, **kwargs):
        self._i = step_i
