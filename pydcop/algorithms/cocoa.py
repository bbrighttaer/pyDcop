
# Type of graph to use with COCOA
import random

import numpy as np

from pydcop.algorithms import ComputationDef, AlgoParameterDef
from pydcop.dcop.relations import Constraint
from pydcop.infrastructure.computations import VariableComputation, Message, register
from pydcop.infrastructure.message_types import ConstraintEvaluationResponse
from pydcop.infrastructure.orchestrator import DcopExecutionMessage
from pydcop.stabilization.base import DynamicDcopComputationMixin

# GRAPH_TYPE = "constraints_hypergraph"
GRAPH_TYPE = "pseudotree"
MSG_PRIORITY = 1
HOLD = "hold"
DONE = "done"
IDLE = "idle"
algo_params = [
    AlgoParameterDef("execution_order", "str", ["top-down", "bottom-up"], "bottom-up"),
    AlgoParameterDef("optimization", "str", ["min", "max"], "min"),
]


def build_computation(comp_def: ComputationDef):

    computation = CoCoA(comp_def)
    return computation


def computation_memory(*args):
    raise NotImplementedError("CoCoA has no computation memory implementation (yet)")


def communication_load(*args):
    raise NotImplementedError("CoCoA has no communication_load implementation (yet)")


class CoCoAMessage(Message):
    INQUIRY_MESSAGE = "InquiryMessage"
    COST_MESSAGE = "CostMessage"
    UPDATE_STATE_MESSAGE = "UpdateStateMessage"
    START_DCOP_MESSAGE = "StartDCOP"

    def __init__(self, msg_type, content):
        super(CoCoAMessage, self).__init__(msg_type, content)

    @property
    def size(self):
        if self.type == self.INQUIRY_MESSAGE:
            return len(self.content)
        if self.type in [self.START_DCOP_MESSAGE, self.UPDATE_STATE_MESSAGE]:
            return 1
        else:
            # COST_MESSAGE
            # in the case, `cost_map` and `min_cost_domain_vals` are of the same length hence 2 * ...
            return 2 * len(self.content["cost_map"])

    def __str__(self):
        return f"CoCoAMessage({self._msg_type}, {self._content})"


class CoCoA(VariableComputation, DynamicDcopComputationMixin):
    """
    This is an implementation of the Cooperative Constraint Approximation algorithm.

    Cornelis Jan van Leeuwen and Przemyslaw Pawelczak. 2017.
    CoCoA: A Non-Iterative Approach to a Local Search (A)DCOP Solver.
    In Proceedings of the 31st AAAIConference on Artificial Intelligence (AAAI).

    Parameters
    ----------
    comp_def: ComputationDef
    The computation definition which specifies the assigned variable, neighbors, and constraints.
    """

    def __init__(self, comp_def: ComputationDef):
        assert "cocoa" in comp_def.algo.algo

        super(CoCoA, self).__init__(comp_def.node.variable, comp_def)
        self.optimization_op = comp_def.algo.param_value('optimization')  # minimization or maximization
        self._initialize()

    def _initialize(self):
        # for determining HOLD state
        self.beta = 1
        self.status = IDLE

        # keeps track of neighbors in HOLD state
        self.hold_state_history = []

        # keeps track of neighbors in DONE state
        self.done_state_history = []

        # stores data in var: cost_map format
        self.cost_msgs = {}

        # clear value
        self.current_value = None

    def on_start(self):
        self.logger.debug(f"Starting {self.name}")
        self.start_dcop()

    def start_dcop(self, neighbor_triggered=False):
        # check if this computation can start
        parent = self.get_parent()
        if (not parent or neighbor_triggered) and self.neighbors:
            # initialize algorithm properties
            if not neighbor_triggered:
                self._initialize()

            # send inquiry messages
            msg = CoCoAMessage(CoCoAMessage.INQUIRY_MESSAGE, self.variable.domain.values)
            self.post_to_all_neighbors(msg, MSG_PRIORITY, on_error="fail")
        else:
            self.execute_neighbor_comp()

    def get_parent(self):
        for link in self.computation_def.node.links:
            if link.type == 'parent':
                return link.target

    def get_children(self):
        children = []
        for link in self.computation_def.node.links:
            if link.type == 'children':
                children.append(link.target)
        return children

    @register("dcop_execution_message")
    def _on_dcop_execution_message(self, sender: str, recv_msg: DcopExecutionMessage, t: int):
        return super(CoCoA, self)._on_dcop_execution_message(sender, recv_msg, t)

    @register("constraint_evaluation_response")
    def _on_constraint_evaluation_response(self, sender: str, recv_msg: ConstraintEvaluationResponse, t: int):
        super(CoCoA, self)._on_constraint_evaluation_response(sender, recv_msg, t)

    @register(CoCoAMessage.INQUIRY_MESSAGE)
    def _on_inquiry_message(self, variable_name: str, recv_msg: CoCoAMessage, t: int):
        """
        Handles received inquiry messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: CoCoAMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(f"{CoCoAMessage.INQUIRY_MESSAGE} from {variable_name} : {recv_msg.content} at {t}")

        # select the constraint
        constraint = None

        # find the constraint that is applicable
        for c in self.computation_def.node.constraints:
            if variable_name in c.scope_names:
                constraint = c
                break

        if constraint is None:
            self.logger.debug(f'No constraint found')
            return

        # Each constraint has two nodes. The current node is always assumed as var1.
        # This way, the domain of var1 form the rows and domain of var2 form the columns of the cost matrix
        variables = constraint.dimensions
        if self.name == variables[0].name:
            var1 = variables[0]
            var2 = variables[1]
            var2.domain.values = recv_msg.content
        else:
            var1 = variables[1]
            var2 = variables[0]
            var1.domain.values = recv_msg.content

        # Calculate the costs
        cost_map = np.zeros(constraint.shape)
        if self.current_value:
            self._calc_cost(constraint, cost_map, self.current_value, 0, var1, var2)
            cost_map_min = cost_map[0, :]
        else:
            for i, d1 in enumerate(var1.domain.values):
                self._calc_cost(constraint, cost_map, d1, i, var1, var2)

            # reduce cost map to array
            if self.optimization_op == 'min':
                cost_map_min = cost_map.min(axis=0).tolist()
            else:
                cost_map_min = cost_map.max(axis=0).tolist()

        # select the indices that yielded the min values
        if self.optimization_op == 'min':
            cost_map_indices = cost_map.argmin(axis=0)
        else:
            cost_map_indices = cost_map.argmax(axis=0)

        # map the optimal value indices in the cost map to their domain values
        domain_values = list(self.variable.domain.values)
        cost_domain_vals = [self.current_value] * len(domain_values) if self.current_value \
            else [domain_values[i] for i in cost_map_indices]

        cost_msg = {"cost_map": cost_map_min, "cost_domain_vals": cost_domain_vals}

        # send cost message
        self.post_msg(variable_name, CoCoAMessage(CoCoAMessage.COST_MESSAGE, cost_msg), MSG_PRIORITY, on_error="fail")

    def _calc_cost(self, constraint, cost_map, d1, i, var1, var2):
        for j, d2 in enumerate(var2.domain.values):
            try:
                cost_map[i, j] = constraint(**{var1.name: d1, var2.name: d2})
            except Exception as e:
                self.logger.error(f'Error calculating cost of inquiry message: {str(e)}, var1 = {var1}, var2={var2}')
                cost_map[i, j] = 0

    @register(CoCoAMessage.COST_MESSAGE)
    def _on_cost_message(self, variable_name: str, recv_msg: CoCoAMessage, t: int):
        """
        Handles received cost messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: CoCoAMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(f"{CoCoAMessage.COST_MESSAGE} from {variable_name} : {recv_msg.content} at {t}")

        # update cost map
        self.cost_msgs[variable_name] = recv_msg.content

        # check if all neighbors have responded to cost inquiries
        self.logger.debug(f'{self.neighbors}, {self.cost_msgs}, {self.computation_def.exec_mode}')
        if len(self.neighbors) == len(self.cost_msgs):
            try:
                self.select_random_value()
            except KeyError as e:
                self.logger.debug(f'Aborting, cannot find a variable: {str(e)}')

    @register(CoCoAMessage.UPDATE_STATE_MESSAGE)
    def _on_update_state_message(self, variable_name: str, recv_msg: CoCoAMessage, t: int):
        """
        Receives an update state message from a neighbor and update appropriate registers.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: CoCoAMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(f"Received state update from {variable_name}: {recv_msg}")

        if recv_msg.content == HOLD:
            self.hold_state_history.append(variable_name)
            self.start_dcop(neighbor_triggered=True)

        elif recv_msg.content == DONE:
            self.done_state_history.append(variable_name)

            # find an available to execute or complete computation
            if self.status == DONE:
                self.execute_neighbor_comp()

    @register(CoCoAMessage.START_DCOP_MESSAGE)
    def _on_start_dcop_msg(self, variable_name: str, recv_msg: CoCoAMessage, t: int):
        """
        Triggered by a neighbor to initiate the DCOP computation.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: CoCoAMessage
            received message
        t: int
            message timestamp
        """
        if self.current_value:
            self.execute_neighbor_comp()
        else:
            self.start_dcop(neighbor_triggered=True)

    def select_random_value(self):
        self.new_cycle()
        value = random.choice(self.variable.domain.values)
        self.logger.debug(f'selected value: {value}')
        self.value_selection(value, 10)
        self.execute_neighbor_comp()

    def select_value(self):
        """
        Select a value for this variable.
        CoCoA is not iterative, once we have selected our value the computation finishes.
        """
        self.logger.debug(f"Attempting value selection for {self.name}")

        # select domain indices with min/max cost
        cost_matrix = np.array([c["cost_map"] for c in self.cost_msgs.values()])
        delta = cost_matrix.sum(axis=0)
        if self.optimization_op == 'min':
            min_val = delta.min()
            opt_indices = np.asarray(delta == min_val).nonzero()[0]
            d_vals = np.array(self.variable.domain.values)
            rho = d_vals[opt_indices].tolist()
        else:
            max_val = delta.max()
            opt_indices = np.asarray(delta == max_val).nonzero()[0]
            d_vals = np.array(self.variable.domain.values)
            rho = d_vals[opt_indices].tolist()

        # determine if a HOLD state is needed
        if len(rho) > self.beta:
            if self.status == HOLD:
                self.logger.debug("Incrementing beta to release HOLD state")
                self.beta += 1
                self.select_value()
            else:
                self.logger.debug("Going into HOLD state")

                # go into HOLD state
                self.status = HOLD

                # get idle neighbors
                idle_neighbors = set(self.neighbors) - set(self.hold_state_history)
                if idle_neighbors:
                    selected_neighbor = random.choice(list(idle_neighbors))
                    self.post_msg(selected_neighbor, CoCoAMessage(CoCoAMessage.UPDATE_STATE_MESSAGE, self.status))
                else:
                    self.select_value()
        else:
            self.new_cycle()

            # construct best values of all neighbors
            val = random.choice(rho)
            min_index = opt_indices[0]
            var_values = {self.name: val}
            for agent in self.neighbors:
                var_values[agent] = self.cost_msgs[agent]["cost_domain_vals"][min_index]

            # compute cost
            cost, value = self._calculate_cost(var_values)
            self.value_selection(value, cost)
            self.logger.debug(f"Value selected at {self.name} : value = {value}, cost = {cost}")

            # update neighbors
            self.status = DONE
            self.post_to_all_neighbors(CoCoAMessage(CoCoAMessage.UPDATE_STATE_MESSAGE, self.status), on_error="fail")

            # select next neighbor to execute
            self.execute_neighbor_comp()

    def _calculate_cost(self, var_values) -> tuple:
        """
        Calculate the cost given this variable's value and that of its neighbors.

        Parameters
        ----------
        var_values: dict
            The dictionary of var_name-value pairs.

        Returns
        -------
            The cost and value as a tuple
        """
        cost = 0
        try:
            for constraint in self.computation_def.node.constraints:
                cost += constraint(**{var: var_values[var] for var in constraint.scope_names})
        except Exception as e:
            self.logger.error(f'Error calculating cost: {e} for {var_values}')
        value = var_values[self.name]
        return cost, value

    def execute_neighbor_comp(self):
        """
        Randomly select a neighbor to trigger execution
        """
        self.logger.debug("Selecting neighbor to start")
        available_neighbors = set(self.neighbors) - set(self.done_state_history)
        for neighbor in available_neighbors:
            self.logger.debug(f"Neighbor {neighbor} selected to start")
            self.post_msg(neighbor, CoCoAMessage(CoCoAMessage.START_DCOP_MESSAGE, None), on_error="fail")

        if self.computation_def.exec_mode != 'dynamic':
            self.stop()
            self.finished()

    def finished(self):
        self.done_state_history.clear()
        self.cost_msgs.clear()
        self.status = IDLE
        self.hold_state_history.clear()

        super(CoCoA, self).finished()


