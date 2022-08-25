from typing import Iterable

from pydcop.computations_graph.objects import ComputationGraph, ComputationNode
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import Constraint, find_dependent_relations


class DynamicComputationNode(ComputationNode):
    """
    Computation node for holding node creation information in dynamic graph execution case.

    Parameters
    -----------
    variable: Variable
        The variable for the node creation
    node_type: str
        Since this node type basically shadows either a `VariableComputationNode` or `PseudoTreeNode`, this
        parameter enables the actual node type to be passed down for dynamic procedures.
    var_constraints: Iterable[Constraint]
        An iterable of `Constraint` objects the variable **may be part of** in the dynamic environment. This is
        different from the actual set of constraints that may be used for computation, which are determined dynamically.
    """

    def __init__(self, variable: Variable, node_type: str, var_constraints: Iterable[Constraint]):
        super(DynamicComputationNode, self).__init__(variable.name, node_type)
        self._variable = variable
        self._constraints = []
        self._var_constraints = var_constraints

    @property
    def variable(self) -> Variable:
        return self._variable

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        self._constraints = constraints

    @property
    def var_constraints(self):
        return self._var_constraints

    def __eq__(self, other):
        if type(other) != DynamicComputationNode:
            return False
        if self.variable == other.variable and self.constraints == other.constraints:
            return True
        return False

    def __str__(self):
        return "DynamicComputationNode({})".format(self._variable.name)

    def __repr__(self):
        return "DynamicComputationNode({}, {})".format(
            self._variable, self.constraints
        )

    def __hash__(self):
        return hash(
            (self._name, self._node_type, self.variable, tuple(self.constraints))
        )


class DynamicComputationGraph(ComputationGraph):

    """
    Dynamic Graph implementation
    """

    def __init__(self, nodes: Iterable[DynamicComputationNode]):
        super(DynamicComputationGraph, self).__init__("dynamic_graph")
        self.nodes = nodes

    def density(self):
        return -1  # todo(bbrighttaer): define the density of a dynamic graph


def build_computation_graph(dcop: DCOP = None, variables: Iterable[Variable] = None,
                            constraints: Iterable[Constraint] = None, graph_module=None) -> DynamicComputationGraph:
    computations = []
    if dcop is not None:
        if constraints or variables is not None:
            raise ValueError(
                "Cannot use both dcop and constraints / " "variables parameters"
            )
        for v in dcop.variables.values():
            var_constraints = find_dependent_relations(v, dcop.constraints.values())
            setattr(v, 'constraints', var_constraints)
            computations.append(DynamicComputationNode(v, graph_module.GRAPH_NODE_TYPE, var_constraints))
    else:
        if constraints is None or variables is None:
            raise ValueError(
                "Constraints AND variables parameters must be "
                "provided when not building the graph from a dcop"
            )
        for v in variables:
            var_constraints = find_dependent_relations(v, constraints)
            computations.append(DynamicComputationNode(v, graph_module.GRAPH_NODE_TYPE, var_constraints))

    return DynamicComputationGraph(computations)
