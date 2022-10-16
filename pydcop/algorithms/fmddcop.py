import numpy as np

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import VariableComputation, message_type, register

GRAPH_TYPE = "constraints_hypergraph"
MSG_PRIORITY = 1
alg_params = [
    AlgoParameterDef("learning_rate", "float", default_value=0.01),
]

Action = int

CoordinationMessage = message_type('coordination_msg', fields=[])
CoordinationMessageResp = message_type('coordination_msg_resp', fields=['x', 'y'])


def build_computation(comp_def: ComputationDef):
    return FMDDCOP(comp_def)


class ModelFreeDynamicDCOP(VariableComputation):

    def __init__(self, *args, **kwargs):
        super(ModelFreeDynamicDCOP, self).__init__(*args, **kwargs)
        self.coordination_constraint_cb = None
        self.unary_constraint_cb = None
        self._obs_history = []
        self._time_step = -1

        self._set_observation_cb = None

    def set_observation(self, obs: dict):
        self.logger.debug(f'Received observation: {obs}')
        self._obs_history.append(obs)

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
        self._set_observation_cb = self._initialize_parameters

    @register('coordination_msg')
    def _on_coordination_msg(self, variable_name: str, recv_msg: CoordinationMessage, t: int):
        self.logger.debug(f'Received coordination message')

    @register('coordination_msg_resp')
    def _on_coordination_msg_resp(self, variable_name: str, recv_msg: CoordinationMessageResp, t: int):
        self.logger.debug(f'Received coordination response message')

    def _initialize_parameters(self, obs):
        if self._parameters is None:
            self._parameters = np.random.randn(len(self._domain), len(obs))

    def _value_function(self):
        s = self._obs_history[-1].values()

    def on_start(self):
        self.logger.debug("Started FMD-DCOP")
