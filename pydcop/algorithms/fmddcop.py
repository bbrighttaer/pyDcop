"""
uses DQN implementation found here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
import random
import threading
import time
from collections import defaultdict, namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import VariableComputation, message_type, register
from pydcop.fmddcop.math import math

GRAPH_TYPE = "constraints_hypergraph"
MSG_PRIORITY = 1
alg_params = [
    AlgoParameterDef("learning_rate", "float", default_value=0.01),
]

Action = int

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# message types
CoordinationMsg = message_type('coordination_msg', fields=[])
CoordinationMsgResp = message_type('coordination_msg_resp', fields=['data'])
Gain = message_type('gain', fields=['val'])
GainRequest = message_type('gain_request', fields=[])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        # self.linear1 = nn.Linear(input_size, 10)
        self.output = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x = self.linear1(x)
        # x = F.relu(x)
        return self.output(x)


def build_computation(comp_def: ComputationDef):
    return FMDDCOP(comp_def)


class ModelFreeDynamicDCOP(VariableComputation):

    def __init__(self, *args, **kwargs):
        super(ModelFreeDynamicDCOP, self).__init__(*args, **kwargs)
        self.coordination_constraint_cb = None
        self.unary_constraint_cb = None
        self.coordination_data_cb = None
        self._time_step = -1

        self._set_observation_cb = None

    # def set_observation(self, obs: dict):
    #     self.logger.debug(f'Received observation: {obs}')
    #
    #     # call registered callback
    #     if callable(self._set_observation_cb):
    #         self._set_observation_cb(obs)

    def resolve_decision_variable(self, obs: dict) -> Action:
        self._time_step += 1
        self.logger.debug(f'Resolving value of decision variable')
        return -1


class FMDDCOP(ModelFreeDynamicDCOP):

    def __init__(self, comp_def: ComputationDef):
        super(FMDDCOP, self).__init__(comp_def.node.variable, comp_def)
        # the domain or action space
        self._domain = comp_def.node.variable.domain

        # helpers to construct training samples
        self._hist_len = 10
        self._obs_history = deque(maxlen=self._hist_len)
        self._value_history = deque(maxlen=self._hist_len)
        self._util_history = deque(maxlen=self._hist_len)

        # model and learning props
        self._model = None
        self._model_target = None
        self._optimizer = None
        self._buffer = ReplayMemory(1000)
        self._cost_coefficient = 5
        self._lr = 0.1
        self._batch_size = 16
        self._gamma = 0.999
        self._target_update = 10
        self._num_training = 0

        # props for gathering neighbor data for local utility estimation
        self._neighbor_data = {}
        self._coordination_data_evt = threading.Event()

        # props for managing utility estimation
        self._max_util_iter = 1
        self._gain = float('-inf')
        self._gain_msgs = []
        self._expected_total_num_msgs = self._max_util_iter * len(self.neighbors) + 1

    def on_start(self):
        self.logger.debug('Started FMD-DCOP')
        self._start_model_trainer()

    def _start_model_trainer(self):
        trainer = threading.Timer(1, self._train_dqn_model)
        trainer.start()

    @register('coordination_msg')
    def _on_coordination_msg(self, variable_name: str, recv_msg: CoordinationMsg, t: int):
        # send coordination response
        self.post_msg(
            target=variable_name,
            msg=CoordinationMsgResp(data=self.coordination_data_cb()),
        )

    @register('coordination_msg_resp')
    def _on_coordination_msg_resp(self, variable_name: str, recv_msg: CoordinationMsgResp, t: int):
        self._neighbor_data[variable_name] = recv_msg

        # wait for data from all neighbors before proceeding
        if len(self.neighbors) == len(self._neighbor_data):
            self._coordination_data_evt.set()

    @register('gain_request')
    def _on_receive_gain_request(self, variable_name: str, recv_msg: GainRequest, t: int):
        self.post_msg(
            target=variable_name,
            msg=Gain(self._gain),
        )

    @register('gain')
    def _on_receive_gain(self, variable_name: str, recv_msg: Gain, t: int):
        self._gain_msgs.append((variable_name, recv_msg.val))

    def _process_gain_msg(self):
        # wait for neighbor msgs
        while len(self._gain_msgs) != self._expected_total_num_msgs:
            continue

        # ignore neighbors that returned None gain and find the max gain reported
        _, self.gain = max(self._gain_msgs, key=lambda x: x[1])

        # if all possible gain msgs have been received, create and store a transition
        self._buffer.push(
            self._obs_history.pop(),
            self._value_history.pop(),
            self._obs_history[-1],
            self.gain,
        )
        self._gain_msgs.clear()

    def _set_observation(self, obs: list):
        self._obs_history.append(obs)

        # initialize trainable parameters
        if self._model is None:
            self._initialize_parameters(obs)

        # send coordination msg
        self.post_to_all_neighbors(
            msg=CoordinationMsg(),
        )

    def _initialize_parameters(self, obs):
        self._model = DQN(len(obs), len(self._domain))
        self._model_target = DQN(len(obs), len(self._domain))
        self._model_target.load_state_dict(self._model.state_dict())
        self._optimizer = torch.optim.RMSprop(self._model.parameters())

    def resolve_decision_variable(self, obs: dict) -> Action:
        self._time_step += 1

        obs_array = list(obs.values())
        self._set_observation(obs_array)

        # ensure observation and coordination data are ready for processing
        self._coordination_data_evt.wait()
        self._coordination_data_evt.clear()  # clear for next check

        if len(self._obs_history) > 1:
            # calculate utility
            gain = self._compute_local_utility(obs)
            self._gain_msgs.append((self.name, gain))

            # request gain messages from neighbors
            self.post_to_all_neighbors(
                msg=GainRequest(),
            )

            self._process_gain_msg()

        # select value/action for decision variable
        val = self._select_value(state=obs_array)

        return val

    def _update_value_params(self):
        pass

    def _compute_local_utility(self, obs: dict):
        # coordination constraints
        c_val = self.coordination_constraint_cb(**self._neighbor_data)
        self._neighbor_data.clear()  # reset for next collation

        # unary constraints
        u_val = self.unary_constraint_cb(**obs)

        # value change cost
        val_change_cost = 0
        if len(self._value_history) > 1:
            x_t_1 = self._value_history[-2]
            x_t = self._value_history[-1]
            val_change_cost = self._cost_coefficient * int(x_t != x_t_1)

        # previous util
        p_util = 0
        if len(self._util_history) > 0:
            p_util = self._util_history.pop()

        # current util
        c_util = u_val + c_val - val_change_cost

        # calculate gain in util
        gain = c_util - p_util

        # maintain history
        self._util_history.append(c_util)

        return gain

    @torch.no_grad()
    def _select_value(self, state: list):
        state = torch.tensor(state).reshape(1, -1)
        val = self._model(state).max(1)[1]
        val = val.item()
        self._value_history.append(val)
        return val

    def _train_dqn_model(self):
        train = False
        if len(self._buffer) >= self._batch_size:
            train = True

        if train:
            # sample batch from buffer
            transitions = self._buffer.sample(self._batch_size)

            # transpose batch of transitions to transition of batch-arrays
            batch = Transition(*zip(*transitions))

            state_batch = torch.tensor(batch.state)
            action_batch = torch.tensor(batch.action).view(-1, 1)
            reward_batch = torch.tensor(batch.reward).view(-1, 1)
            next_state_batch = torch.tensor(batch.next_state)

            # compute Q(s_t, a)
            state_action_values = self._model(state_batch).gather(1, action_batch)

            # compute V(s_{t+1}) for all next states
            next_state_values = self._model_target(next_state_batch).max(1)[0].detach()

            # compute the expected Q values
            expected_state_action_values = (next_state_values * self._gamma) * reward_batch

            # compute loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # optimize the model
            self._optimizer.zero_grad()
            loss.backward()
            for param in self._model.parameters():
                param.grad.data.clamp_(-1, 1)
            self._optimizer.step()

            self._num_training += 1

            # update target network
            if self._num_training % self._target_update == 0:
                self._model_target.load_state_dict(self._model.state_dict())

        # schedule next training
        self._start_model_trainer()

