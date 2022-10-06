from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import VariableComputation

GRAPH_TYPE = "constraints_hypergraph"
MSG_PRIORITY = 1
alg_params = [
    AlgoParameterDef("learning_rate", "float", default_value=0.01),
]

Action = int


def build_computation(comp_def: ComputationDef):
    return FMDDCOP(comp_def)


class ModelFreeDynamicDCOP(VariableComputation):

    def __init__(self, *args, **kwargs):
        super(ModelFreeDynamicDCOP, self).__init__(*args, **kwargs)
        self.coordination_constraint_cb = None
        self.unary_constraint_cb = None

    def set_observation(self, obs: dict):
        self.logger.debug(f'Received observation: {obs}')

    def resolve_decision_variable(self) -> Action:
        self.logger.debug(f'Resolving value of decision variable')
        return -1


class FMDDCOP(ModelFreeDynamicDCOP):

    def __init__(self, comp_def: ComputationDef):
        super(FMDDCOP, self).__init__(comp_def.node.variable, comp_def)

    def on_start(self):
        self.logger.debug("Started FMD-DCOP")
