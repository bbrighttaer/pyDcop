import random
import time

import numpy as np

from pydcop.envs import SimulationEnvironment

seed = 7
random.seed(seed)


class GridWorld(SimulationEnvironment):
    name = 'GridWorld'

    def __init__(self, size, num_targets, scenario):
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
            self.logger.debug(self.history)

            evt = next(self._events_iterator)
            if not evt.is_delay:
                for a in evt.actions:
                    if a.type == 'add_agent':
                        self.logger.info('Event action: Adding agent %s ', a)
                        self.run_stabilization_computation(a.args['agent'])

                    elif a.type == 'remove_agent':
                        self.logger.info('Event action: Remove agent %s ', a)
                        self.remove_agent(a.args['agent'])

            # wait for a while before signaling agents
            time.sleep(.1)
            self.next_time_step()
        except StopIteration:
            self.on_simulation_ended()

    def run_stabilization_computation(self, agent):
        # get all possible positions
        cell_ids = list(self.grid.keys())

        # uniformly sample a position for this target
        selected_cell_id = random.choice(cell_ids)
        selected_cell = self.grid[selected_cell_id]

        # create agent in the environment
        msa = MobileSensingAgent(agent, selected_cell)
        self.agents[msa.player_id] = msa

        # add sensor to cell
        selected_cell.add(msa)

    def remove_agent(self, agent):
        # remove agent from agents list
        msa = self.agents.pop(agent)

        # remove agent from currently occupied cell
        cell: GridCell = msa.current_cell
        cell.contents.pop(cell.contents.index(msa))

    def next_time_step(self):
        self._current_time_step += 1
        grid = [str(v) for v in self.grid.values()]
        self._state_history.append((f't={str(self._current_time_step)}', grid))

    def _create_cells(self):
        for i in range(1, self.grid_size + 1):
            for j in range(1, self.grid_size + 1):
                cell = GridCell(i, j)
                self.grid[cell.cell_id] = cell

    def get_time_step_end_data(self, agent_id):
        return {
            'current_position': None,
            'score': None,  # score in the just ended time step
            'agents_in_comm_range': None,
        }

    def get_agents_in_communication_range(self, agent_id) -> list:
        nearby_agents = []
        cells_to_inspect = []
        agent: MobileSensingAgent = self.agents[agent_id]
        cell: GridCell = agent.current_cell

        # compile list of cells to inspect
        cells_to_inspect.append(cell)
        if agent.communication_range > 1:
            up = self.grid.get(cell.up(), None)
            down = self.grid.get(cell.down(), None)
            left = self.grid.get(cell.left(), None)
            right = self.grid.get(cell.right(), None)
            left_up = self.grid.get(cell.left_up(), None)
            right_up = self.grid.get(cell.right_up(), None)
            left_down = self.grid.get(cell.left_down(), None)
            right_down = self.grid.get(cell.right_down(), None)
            cells_to_inspect += [up, down, left, right, left_up, right_up, left_down, right_down]

        # inspect cells for agents
        for cell in cells_to_inspect:
            if cell:
                for obj in cell.contents:
                    if isinstance(obj, MobileSensingAgent) and obj.player_id != agent_id:
                        nearby_agents.append(obj.player_id)

        return nearby_agents


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

    def up(self):
        return str(self.i - 1) + '-' + str(self.j)

    def down(self):
        return str(self.i + 1) + '-' + str(self.j)

    def left(self):
        return str(self.i) + '-' + str(self.j - 1)

    def right(self):
        return str(self.i) + '-' + str(self.j + 1)

    def left_up(self):
        return str(self.i - 1) + '-' + str(self.j - 1)

    def left_down(self):
        return str(self.i + 1) + '-' + str(self.j - 1)

    def right_up(self):
        return str(self.i - 1) + '-' + str(self.j + 1)

    def right_down(self):
        return str(self.i + 1) + '-' + str(self.j + 1)

    def __str__(self):
        return f'{self.cell_id}: {str([str(c) for c in self.contents])}'


class MobileSensingAgent:

    def __init__(self, player_id, cell):
        super().__init__()
        self.player_id = player_id
        self.current_cell = cell
        self.credibility = 5
        self.sensing_range = 1
        self.mobility_range = 2
        self.communication_range = 3

    def __str__(self):
        return f'Agent(id={self.player_id}, cred={self.credibility})'


class Target:

    def __init__(self, target_id, cell, cov_req):
        self.target_id = target_id
        self.current_cell = cell
        self.coverage_requirement = cov_req
        self.is_active = True

    def __str__(self):
        return f'Target(target_id={self.target_id}, cov_req={self.coverage_requirement}, is_active={self.is_active})'
