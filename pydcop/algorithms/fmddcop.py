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
from pydcop.infrastructure.computations import VariableComputation, register
from pydcop.infrastructure.message_types import message_type
from pydcop.fmddcop.math import math

GRAPH_TYPE = "constraints_hypergraph"
MSG_PRIORITY = 1
alg_params = [
    AlgoParameterDef("learning_rate", "float", default_value=0.01),
]

Action = int

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'next_state_action', 'reward'))

# message types
CoordinationMsg = message_type('coordination_msg', fields=['n_encoding'])
CoordinationMsgResp = message_type('coordination_msg_resp', fields=['data'])
Gain = message_type('gain', fields=['val'])
GainRequest = message_type('gain_request', fields=[])

seed = 7
random.seed(seed)
torch.manual_seed(seed)


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


class ValueFunctionModel(nn.Module):

    def __init__(self, num_neighbors: int, neighborhood_dim: int, action_vector_dim: int):
        super(ValueFunctionModel, self).__init__()
        self.linear1 = nn.Linear((num_neighbors + 1) * neighborhood_dim, neighborhood_dim)
        self.bn = nn.BatchNorm1d(neighborhood_dim)
        self.output = nn.Linear(neighborhood_dim + action_vector_dim, 1)

    def neighborhood_encoding(self, x):
        x = self.linear1(x)
        if x.shape[0] == 1:
            self.bn.eval()
        x = self.bn(x)
        x = F.leaky_relu(x)
        return x

    def forward(self, n_x, a_x):
        n_x = self.neighborhood_encoding(n_x)
        x = self.output(torch.concat([n_x, a_x], dim=1))
        return x


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

    def resolve_decision_variable(self, obs: dict, action_candidates: list, action_eval: list) -> Action:
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

        # model and learning props
        self._model: ValueFunctionModel = None
        self._model_target: ValueFunctionModel = None
        self._optimizer = None
        self._buffer = ReplayMemory(1000)
        self._cost_coefficient = 1
        self._lr = 0.1
        self._batch_size = 16
        self._gamma = 0.999
        self._target_update = 10
        self._num_training = 0

        self._action_vector_dim = 5
        self._obs_data_dim = 5
        self._obs_data = {
            var: [0.] * self._obs_data_dim for var in self.neighbors
        }

        # props for gathering neighbor data for local utility computation
        self._neighbor_data = {}
        self._coordination_data_evt = threading.Event()

    def on_start(self):
        self.logger.debug('Started FMD-DCOP')
        self._start_model_trainer()

    def _start_model_trainer(self):
        trainer = threading.Timer(1, self._train_dqn_model)
        trainer.start()

    @register('coordination_msg')
    def _on_coordination_msg(self, variable_name: str, recv_msg: CoordinationMsg, t: int):
        # update neighborhood information of sender
        self._obs_data[variable_name] = recv_msg.n_encoding

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

    def _set_observation(self, obs: list):
        self._obs_data[self.name] = obs

        # initialize trainable parameters
        if self._model is None:
            self._initialize_parameters()

        self._model.eval()

        # get neighborhood encoding
        x = torch.cat([torch.tensor(v) for v in self._obs_data.values()])
        n_encoding = self._model.neighborhood_encoding(x.view(1, -1))
        n_encoding = n_encoding.detach()

        # send coordination msg
        self.post_to_all_neighbors(
            msg=CoordinationMsg(n_encoding=n_encoding.squeeze().tolist()),
        )

        return n_encoding

    def _initialize_parameters(self):
        self._model = ValueFunctionModel(
            num_neighbors=len(self.neighbors),
            neighborhood_dim=self._obs_data_dim,
            action_vector_dim=self._action_vector_dim,
        )
        self._model_target = ValueFunctionModel(
            num_neighbors=len(self.neighbors),
            neighborhood_dim=self._obs_data_dim,
            action_vector_dim=self._action_vector_dim,
        )
        self._model_target.load_state_dict(self._model.state_dict())
        self._optimizer = torch.optim.RMSprop(self._model.parameters())

    def resolve_decision_variable(self, obs: dict, action_candidates: list, action_eval: list) -> Action:
        self._time_step += 1

        obs_array = list(obs.values())
        n_encoding = self._set_observation(obs_array)

        # select value/action for decision variable
        val = self._select_value(n_encoding, action_candidates)

        # cache info for creating experience later
        self._obs_history.append((
            list(self._obs_data.values()),  # self and neighbor data,
            action_candidates[val],  # selected action vector,
            action_eval[val],  # selected action's evaluation,
        ))

        # create experience if possible
        if len(self._obs_history) > 1:
            s, a, v = self._obs_history.popleft()
            s_prime = list(self._obs_data.values())
            a_prime = action_candidates[val]
            util = self._compute_local_utility(a, a_prime, v)
            self._buffer.push(s, a, s_prime, a_prime, util)

        return val

    def _update_value_params(self):
        pass

    def _compute_local_utility(self, a1, a2, a1_v):
        # ensure coordination responses have been received
        self._coordination_data_evt.wait()
        self._coordination_data_evt.clear()

        # coordination constraints
        c_val = self.coordination_constraint_cb(**self._neighbor_data)
        self._neighbor_data.clear()  # reset for next collation

        # unary constraints
        u_val = a1_v  # self.unary_constraint_cb(**obs)

        # value change cost (compare type of actions)
        a1 = a1[-3:]
        a2 = a2[-3:]
        val_change_cost = self._cost_coefficient * int(a1.index(1) != a2.index(1))

        # current util
        c_util = u_val + c_val - val_change_cost

        return c_util

    def _select_value(self, n_encoding, action_vectors):
        self._model.eval()
        action_vectors = torch.tensor(action_vectors)
        n_encoding = n_encoding.repeat(action_vectors.shape[0], 1)
        x = torch.cat([n_encoding, action_vectors], dim=1).float()
        output = self._model.output(x)

        # epsilon-greedy
        if random.random() < 0.1:
            val = random.randint(0, len(action_vectors) - 1)
        else:
            val = output._max(0)[1]
            val = val.item()

        return val

    def _train_dqn_model(self):
        if self._model:
            self._model.train()

        can_train = False
        if len(self._buffer) >= self._batch_size:
            can_train = True

        if can_train:
            with torch.set_grad_enabled(True):
                # sample batch from buffer
                transitions = self._buffer.sample(self._batch_size)

                # transpose batch of transitions to transition of batch-arrays
                batch = Transition(*zip(*transitions))

                state_batch = torch.tensor(batch.state).view(self._batch_size, -1).float()
                action_batch = torch.tensor(batch.action).view(self._batch_size, -1).float()
                reward_batch = torch.tensor(batch.reward).view(-1, 1).float()
                next_state_batch = torch.tensor(batch.next_state).view(self._batch_size, -1).float()
                next_state_action_batch = torch.tensor(batch.next_state_action).view(self._batch_size, -1).float()

                # compute Q(s_t, a)
                state_action_values = self._model(state_batch, action_batch)

                # compute V(s_{t+1}) for all next states
                next_state_values = self._model_target(next_state_batch, next_state_action_batch).detach()

                # compute the expected Q values
                expected_state_action_values = (next_state_values * self._gamma) * reward_batch

                # compute loss
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values)

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
