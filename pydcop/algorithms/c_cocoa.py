from collections import defaultdict

from pydcop.algorithms import ComputationDef, AlgoParameterDef
from pydcop.algorithms.cocoa import CoCoA, CoCoAMessage
from pydcop.dcop.relations import constraint_from_str
from pydcop.infrastructure.computations import register

GRAPH_TYPE = "constraints_hypergraph"

algo_params = [
    AlgoParameterDef("lr", "float", None, 0.1),
    AlgoParameterDef("max_iter", "int", None, 100),
]


def build_computation(comp_def: ComputationDef):

    computation = CCoCoA(comp_def)
    return computation


def computation_memory(*args):
    raise NotImplementedError("CoCoA has no computation memory implementation (yet)")


def communication_load(*args):
    raise NotImplementedError("CoCoA has no communication_load implementation (yet)")


class CCoCoA(CoCoA):
    """
    Implementation of the C-CoCoA algorithm in:

    Sarker, A., Choudhury, M., & Khan, M. M. (2021).
    A local search based approach to solve continuous DCOPs.
    Proceedings of the International Joint Conference on Autonomous Agents and Multiagent Systems, AAMAS, 2, 1115–1123.
    """

    @register(CoCoAMessage.INQUIRY_MESSAGE)
    def _on_inquiry_message(self, variable_name: str, recv_msg: CoCoAMessage, t: int):
        return super()._on_inquiry_message(variable_name, recv_msg, t)

    @register(CoCoAMessage.COST_MESSAGE)
    def _on_cost_message(self, variable_name: str, recv_msg: CoCoAMessage, t: int):
        return super()._on_cost_message(variable_name, recv_msg, t)

    @register(CoCoAMessage.UPDATE_STATE_MESSAGE)
    def _on_update_state_message(self, variable_name: str, recv_msg: CoCoAMessage, t: int):
        return super()._on_update_state_message(variable_name, recv_msg, t)

    @register(CoCoAMessage.START_DCOP_MESSAGE)
    def _on_start_dcop(self, variable_name: str, recv_msg: CoCoAMessage, t: int):
        return super()._on_start_dcop(variable_name, recv_msg, t)

    def _calculate_cost(self, var_values) -> tuple:
        """
        Apply non-linear optimization using `var_values` as initial parameters.
        This optimization is used as an intermediate step to compute the cost and select a value.

        Parameters
        ----------
        var_values: dict
            The dictionary of var_name-value pairs.

        Returns
        -------
            The cost and value as a tuple
        """
        try:
            # perform optimization in each iteration
            for i in range(self.computation_def.algo.param_value("max_iter")):
                grads = defaultdict(float)
                for constraint in self.computation_def.node.constraints:
                    assert hasattr(constraint, "differentials") \
                           and constraint.differentials is not None, "Differential equations are required for C-CoCoA"

                    # calculate gradients
                    for var_name, exp in constraint.differentials.items():
                        variables = constraint.dimensions
                        equation = constraint_from_str('ddx', exp, variables)
                        grads[var_name] += equation(**{v.name: var_values[v.name] for v in variables})

                # apply gradients
                lr = self.computation_def.algo.param_value('lr')
                for var_name, grad in grads.items():
                    var_values[var_name] -= lr * grad

        except AssertionError as e:
            self.logger.error(f"{str(e)}, falling back to CoCoA")

        return super(CCoCoA, self)._calculate_cost(var_values)



