import random

from pydcop.envs import SimulationEnvironment

GRID_SIZE = 3
seed = 7
random.seed(seed)


class GridWorld(SimulationEnvironment):

    def __init__(self, size, num_agents):
        super(GridWorld, self).__init__('GridWorld')
        self.grid = {}
        self.client = None

    def on_start(self):
        self.logger.debug('Started GridWorld simulation environment')

    def on_stop(self):
        self.logger.debug('Stopped GridWorld simulation environment')

    def step(self):
        ...

    def display(self):
        ...
