from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import VariableComputation

GRAPH_TYPE = "constraints_hypergraph"
MSG_PRIORITY = 1
alg_params = [
    AlgoParameterDef("learning_rate", "float", default_value=0.01),
]


def build_computation(comp_def: ComputationDef):
    return FMDDCOP(comp_def)


class FMDDCOP(VariableComputation):

    def __init__(self, comp_def: ComputationDef):
        super(FMDDCOP, self).__init__(comp_def.node.variable, comp_def)

    def on_start(self):
        self.logger.debug("Started FMD-DCOP")
