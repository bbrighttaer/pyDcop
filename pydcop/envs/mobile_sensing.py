import random
import numpy as np

from pydcop.envs import SimulationEnvironment, TimeStep

seed = 7
random.seed(seed)


class GridWorld(SimulationEnvironment):
    name = 'GridWorld'

    def __init__(self, size, num_agents, num_targets, scenario):
        super(GridWorld, self).__init__(self.name, time_step_delay=2)
        self._events_iterator = iter(scenario)
        self.grid_size = size
        self.grid = {}
        self._current_time_step = -1
        self.num_targets = num_targets
        self._targets = {}
        self._events = ['add-agent', 'remove-agent', 'target_disabled', 'target_enabled', 'no-op']
        self._y = self._events.index('no-op')
        s = len(self._events)
        # v = np.random.rand(s)
        # self._initial_distribution = v / v.sum()
        m = np.random.rand(s, s)
        self._transition_function = m / m.sum(axis=1).reshape(-1, 1)

    def on_start(self):
        self.logger.debug('Started GridWorld simulation environment')
        self._create_cells()
        self._initialize_targets()

    def _initialize_targets(self):
        # get all possible positions
        cell_ids = list(self.grid.keys())

        for i in range(self.num_targets):
            # uniformly sample a position for this target
            selected_cell_id = random.choice(cell_ids)
            selected_cell = self.grid[selected_cell_id]

            # create the target
            target = Target(
                target_id=i,
                cell=selected_cell,
                cov_req=5,
            )

            # add target to cell
            self._targets[target.target_id] = target
            selected_cell.add(target)

    def on_stop(self):
        self.logger.debug('Stopped GridWorld simulation environment')

    def step(self):
        try:
            self._next_time_step()

            # self.logger.debug(str(self.history[-1]))

            evt = next(self._events_iterator)
            if not evt.is_delay:
                for a in evt.actions:
                    if a.type == 'add_agent':
                        self.logger.info('Event action: Adding agent %s ', a)
                        self.run_stabilization_computation(a.args['agent'])

                    elif a.type == 'remove_agent':
                        self.logger.info('Event action: Remove agent %s ', a)
                        self.remove_agent(a.args['agent'])
        except StopIteration:
            self.on_simulation_ended()

    def _next_time_step(self):
        self._current_time_step += 1
        grid_str = str([str(v) for v in self.grid.values()])
        self._state_history.append(str(TimeStep(self._current_time_step, state=grid_str)))

    def _create_cells(self):
        for i in range(1, self.grid_size + 1):
            for j in range(1, self.grid_size + 1):
                cell = GridCell(i, j)
                self.grid[cell.cell_id] = cell


class GridCell:
    """
    Models a cell in the GridWorld environment.

    The cell and its neighbors:
    .-----------.------.------------.
    | left_up   | up   | right_up   |
    :-----------+------+------------:
    | left      | cell | right      |
    :-----------+------+------------:
    | left_down | down | right_down |
    '-----------'------'------------'
    """

    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.cell_id = f'{i}-{j}'
        self.contents = []

    def add(self, o):
        self.contents.append(o)

    def __str__(self):
        return f'{self.cell_id}: {str([str(c) for c in self.contents])}'


class MobileSensor:

    def __init__(self, player_id):
        super().__init__()
        self.player_id = player_id
        self.client = None
        self.current_position = None
        self.credibility = 5
        self.sensing_range = 1
        self.mobility_range = 2
        self.connectivity_range = 3


class Target:

    def __init__(self, target_id, cell, cov_req):
        self.target_id = target_id
        self.current_cell = cell
        self.coverage_requirement = cov_req
        self.is_active = True

    def __str__(self):
        return f'Target(target_id={self.target_id}, cov_req={self.coverage_requirement}, is_active={self.is_active})'
