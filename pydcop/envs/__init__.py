from typing import List, Tuple

from pydcop.infrastructure.computations import MessagePassingComputation


class SimulationEnvironment(MessagePassingComputation):
    """
    Base class for all simulation environments used for D-DCOP
    """

    def __init__(self, name, time_step_delay, scenario):
        super(SimulationEnvironment, self).__init__(name)
        self._events_iterator = iter(scenario) if scenario else None
        self._state_history = []
        self.time_step_delay = time_step_delay
        self.agents = {}

    def step(self):
        ...

    def set_scenario(self, scenario):
        self._events_iterator = iter(scenario) if scenario else None

    def display(self):
        self.logger.info(str(self))

    @property
    def history(self):
        return self._state_history

    def __str__(self):
        return str(self.history)

    def on_simulation_ended(self):
        ...

    def run_stabilization_computation(self, agent):
        ...

    def remove_agent(self, agent):
        ...

    def next_time_step(self):
        ...

    def get_time_step_end_data(self, agent_id):
        ...

    def evaluate_constraint(self):
        ...

    def get_agents_in_communication_range(self, agent_id):
        ...

    def send_constraint_evaluation_response(self, target, constraint_name, value):
        ...

    def calculate_global_score(self) -> Tuple[int, float]:
        """
        Calculates the global cost or utility of the DCOP problem at a particular snapshot of the simulation.
        Returns
        -------
        Returns a tuple: (number of constraints violated, cost/utility)
        """
        raise NotImplementedError('Global score logic is missing')

    def on_action_selection(self, on_action_cb, sender: str, msg: 'ValueChangeMessage', t: float):
        """
        Applies selected value in the environment.
        """
        ...

    def before_time_step_changed(self):
        ...


class TimeStep:
    """
    Models a single time step of a simulation
    """

    def __init__(self, step_i, state):
        self._i = step_i
        self._state = state

    def __str__(self):
        return f't-{self._i}, state: {self._state}'
