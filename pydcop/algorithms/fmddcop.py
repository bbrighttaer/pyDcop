import threading
from collections import defaultdict, namedtuple

import numpy as np

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import VariableComputation, message_type, register
from pydcop.fmddcop.math import math

GRAPH_TYPE = "constraints_hypergraph"
MSG_PRIORITY = 1
alg_params = [
    AlgoParameterDef("learning_rate", "float", default_value=0.01),
]

Action = int

CoordinationMsg = message_type('coordination_msg', fields=[])
CoordinationMsgResp = message_type('coordination_msg_resp', fields=['data'])

Experience = namedtuple('experience', ['s', 'a', 'r', 's_prime'])


def build_computation(comp_def: ComputationDef):
    return FMDDCOP(comp_def)


class ExpBuffer:

    def __init__(self, n=100):
        self._buffer = []
        self._size = n
        self._pointer = 0

    def add(self, e):
        if len(self._buffer) < self._size:
            self._buffer.append(e)
            self._pointer += 1
        else:
            self._pointer %= self._size
            self._buffer[self._pointer] = e

    def sample(self, size=50):
        exps = np.random.choice(self._buffer, size, replace=False)
        return exps


exp_buffer = ExpBuffer()


class ModelFreeDynamicDCOP(VariableComputation):

    def __init__(self, *args, **kwargs):
        super(ModelFreeDynamicDCOP, self).__init__(*args, **kwargs)
        self.coordination_constraint_cb = None
        self.unary_constraint_cb = None
        self.coordination_data_cb = None
        self._time_step = -1

        self._set_observation_cb = None

    def set_observation(self, obs: dict):
        self.logger.debug(f'Received observation: {obs}')

        # call registered callback
        if callable(self._set_observation_cb):
            self._set_observation_cb(obs)

    def resolve_decision_variable(self) -> Action:
        self._time_step += 1
        self.logger.debug(f'Resolving value of decision variable')
        return -1


class FMDDCOP(ModelFreeDynamicDCOP):

    def __init__(self, comp_def: ComputationDef):
        super(FMDDCOP, self).__init__(comp_def.node.variable, comp_def)
        self._domain = comp_def.node.variable.domain
        self._parameters = None
        self._set_observation_cb = self._observation_cb

        self._obs_history = []
        self._value_history = []
        self._value_selection_count = {k: 1 for k in self._domain}

        self._neighbor_data = {}
        self._coordination_data_evt = threading.Event()

        self._cost_coefficient = 5
        self._lr = 0.1

        self._current_exp = None

    @register('coordination_msg')
    def _on_coordination_msg(self, variable_name: str, recv_msg: CoordinationMsg, t: int):
        self.logger.debug(f'Received coordination message')

        # send coordination response
        self.post_msg(
            target=variable_name,
            msg=CoordinationMsgResp(data=self.coordination_data_cb()),
        )

    @register('coordination_msg_resp')
    def _on_coordination_msg_resp(self, variable_name: str, recv_msg: CoordinationMsgResp, t: int):
        self.logger.debug(f'Received coordination response message: {recv_msg}')
        self._neighbor_data[variable_name] = recv_msg

        # wait for data from all neighbors before proceeding
        if len(self.neighbors) == len(self._neighbor_data):
            self._coordination_data_evt.set()

    def _observation_cb(self, obs):
        self._obs_history.append(obs)

        # initialize trainable parameters
        self._initialize_parameters(obs)

        # send coordination msg
        self.post_to_all_neighbors(
            msg=CoordinationMsg(),
        )

    def _initialize_parameters(self, obs):
        if self._parameters is None:
            self._parameters = np.random.randn(len(self._domain), len(obs))

    def _value_function(self):
        s = self._obs_history[-1].values()

    def on_start(self):
        self.logger.debug("Started FMD-DCOP")

    def resolve_decision_variable(self) -> Action:
        self._time_step += 1
        self.logger.debug(f'Resolving value of decision variable')

        # ensure observation and coordination data are ready for processing
        self._coordination_data_evt.wait()
        self._coordination_data_evt.clear()  # clear for next check

        # update parameters
        self._update_value_params()

        # select value/action for decision variable
        val = self._select_value(state=self._obs_history[-1])

        return val

    def _update_value_params(self):
        pass

    def _compute_utility(self):
        # coordination constraints
        c_val = self.coordination_constraint_cb(**self._neighbor_data)

        # unary constraints
        u_val = self.unary_constraint_cb(**self._obs_history[-1])

        # value change cost
        val_change_cost = 0
        if self._value_history and len(self._value_history) > 1:
            x_t_1 = self._value_history[-2]
            x_t = self._value_history[-1]
            val_change_cost = self._cost_coefficient * int(x_t != x_t_1)
        util = u_val + c_val - val_change_cost
        return util

    def _select_value(self, state: dict):
        obs = np.array(list(state.values())).reshape(-1, 1)
        q_vals = self._parameters @ obs
        v_counts = np.array(list(self._value_selection_count.values())).reshape(-1, 1)
        f = q_vals + np.sqrt(2 * np.log(v_counts) * (len(self._domain) / v_counts))
        val = self._domain[np.argmax(f).astype(int)]

        self._value_selection_count[val] += 1
        self._value_history.append(val)

        if self._current_exp:
            self._current_exp.s_prime = state
        self._current_exp = Experience(state, val, self._compute_utility(), None)
        exp_buffer.add(self._current_exp)

        return val
