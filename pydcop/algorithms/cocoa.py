
# Type of graph to use with COCOA
import random

import numpy as np

from pydcop.algorithms import ComputationDef
from pydcop.dcop.relations import Constraint
from pydcop.infrastructure.computations import VariableComputation, Message, register


GRAPH_TYPE = "constraints_hypergraph"
MSG_PRIORITY = 1
HOLD = "hold"
DONE = "done"
IDLE = "idle"


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


class CoCoA(VariableComputation):
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

        # for determining HOLD state
        self.beta = 1
        self.status = IDLE
        self.hold_state_history = []  # keeps track of neighbors in HOLD state
        self.done_state_history = []  # keeps track of neighbors in DONE state

        # stores data in var: cost_map format
        self.cost_msgs = {}

    def on_start(self):
        self.logger.debug(f"Starting {self.name}")
        self.start_dcop()

    def start_dcop(self, neighbor_triggered=False):
        # check if this computation can start
        if neighbor_triggered or self.computation_def.node.variable.kwargs.get("initiator", False):
            msg = CoCoAMessage(CoCoAMessage.INQUIRY_MESSAGE, self.variable.domain.values)
            self.post_to_all_neighbors(msg, MSG_PRIORITY, on_error="fail")

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
        constraint: Constraint = None

        # find the constraint that is applicable
        for c in self.computation_def.node.constraints:
            if variable_name in c.scope_names:
                constraint = c
                break

        # Each constraint has two nodes. The current node is always assumed as var1.
        # This way, the domain of var1 form the rows and domain of var2 form the columns of the cost matrix
        variables = constraint.dimensions
        if self.name == variables[0].name:
            var1 = variables[0]
            var2 = variables[1]
        else:
            var1 = variables[1]
            var2 = variables[0]

        # Calculate the costs
        cost_map = np.zeros(constraint.shape)
        if self.current_value:
            self._calc_cost(constraint, cost_map, self.current_value, 0, var1, var2)
            cost_map_min = cost_map[0, :]
        else:
            for i, d1 in enumerate(var1.domain.values):
                self._calc_cost(constraint, cost_map, d1, i, var1, var2)

            # reduce cost map to array
            cost_map_min = cost_map.min(axis=0).tolist()

        # select the indices that yielded the min values
        cost_map_min_indices = cost_map.argmin(axis=0)

        # map the min value indices in the cost map to their domain values
        domain_values = list(self.variable.domain.values)
        min_cost_domain_vals = [self.current_value] * len(domain_values) if self.current_value \
            else [domain_values[i] for i in cost_map_min_indices]

        cost_msg = {"cost_map": cost_map_min, "min_cost_domain_vals": min_cost_domain_vals}

        # send cost message
        self.post_msg(variable_name, CoCoAMessage(CoCoAMessage.COST_MESSAGE, cost_msg), MSG_PRIORITY, on_error="fail")

    def _calc_cost(self, constraint, cost_map, d1, i, var1, var2):
        for j, d2 in enumerate(var2.domain.values):
            cost_map[i, j] = constraint(**{var1.name: d1, var2.name: d2})

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
        if len(self.neighbors) == len(self.cost_msgs) and not self.current_value:
            self.select_value()

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
        elif recv_msg.content == DONE:
            self.done_state_history.append(variable_name)

            # find an available to execute or complete computation
            if self.status == DONE:
                self.execute_neighbor_comp()

    @register(CoCoAMessage.START_DCOP_MESSAGE)
    def _on_start_dcop(self, variable_name: str, recv_msg: CoCoAMessage, t: int):
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

    def select_value(self):
        """
        Select a value for this variable.
        CoCoA is not iterative, once we have selected our value the computation finishes.
        """
        self.logger.debug(f"Attempting value selection for {self.name}")

        # select domain indices with minimum cost
        delta = np.array([c["cost_map"] for c in self.cost_msgs.values()]).sum(axis=0, keepdims=True)
        rho = np.argmin(delta, axis=1)

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
            # construct best values of all neighbors
            min_index = random.choice(rho)
            var_values = {self.name: list(self.variable.domain.values)[min_index]}
            for agent in self.neighbors:
                var_values[agent] = self.cost_msgs[agent]["min_cost_domain_vals"][min_index]

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
        for constraint in self.computation_def.node.constraints:
            cost += constraint(**{var: var_values[var] for var in constraint.scope_names})
        value = var_values[self.name]
        return cost, value

    def execute_neighbor_comp(self):
        """
        Randomly select a neighbor to trigger execution
        """
        self.logger.debug("Selecting neighbor to start")
        available_neighbors = set(self.neighbors) - set(self.done_state_history)
        if available_neighbors:
            selected_neighbor = random.choice(list(available_neighbors))
            self.logger.debug(f"Neighbor {selected_neighbor} selected to start")
            self.post_msg(selected_neighbor, CoCoAMessage(CoCoAMessage.START_DCOP_MESSAGE, None), on_error="fail")
        else:
            self.logger.debug(f"No neighbor is available to start, done history: {self.done_state_history}")
            self.finished()
            self.stop()



