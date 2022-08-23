from typing import Iterable

from pydcop.computations_graph.constraints_hypergraph import VariableComputationNode
from pydcop.computations_graph.objects import ComputationGraph
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import Constraint, find_dependent_relations


class DynamicComputationGraph(ComputationGraph):

    """
    Dynamic Graph implementation
    """

    def __init__(self, nodes: Iterable[VariableComputationNode]):
        super(DynamicComputationGraph, self).__init__("dynamic_graph")
        self.nodes = nodes

    def density(self):
        return -1  # todo(bbrighttaer): define the density of a dynamic graph


def build_computation_graph(dcop: DCOP = None, variables: Iterable[Variable] = None,
                            constraints: Iterable[Constraint] = None) -> DynamicComputationGraph:
    computations = []
    if dcop is not None:
        if constraints or variables is not None:
            raise ValueError(
                "Cannot use both dcop and constraints / " "variables parameters"
            )
        for v in dcop.variables.values():
            var_constraints = []  # since it's a dynamic graph, no constraints are known at this stage
            computations.append(VariableComputationNode(v, var_constraints))
    else:
        if constraints is None or variables is None:
            raise ValueError(
                "Constraints AND variables parameters must be "
                "provided when not building the graph from a dcop"
            )
        for v in variables:
            var_constraints = find_dependent_relations(v, constraints)
            computations.append(VariableComputationNode(v, var_constraints))

    return DynamicComputationGraph(computations)
