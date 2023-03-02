
# Type of graph to use with COCOA
import random

import numpy as np

from pydcop.algorithms import ComputationDef, AlgoParameterDef
from pydcop.infrastructure.computations import VariableComputation, register
from pydcop.infrastructure.message_types import Message
from pydcop.stabilization.base import DynamicDcopComputationMixin

# GRAPH_TYPE = "constraints_hypergraph"
GRAPH_TYPE = "pseudotree"
MSG_PRIORITY = 1
HOLD = "hold"
DONE = "done"
IDLE = "idle"
algo_params = [
    AlgoParameterDef("execution_order", "str", ["top-down", "bottom-up"], "bottom-up"),
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
    FORWARDED_DCOP_EXECUTION = 'ForwardedDCOPExecution'

    def __init__(self, msg_type, content):
        super(CoCoAMessage, self).__init__(msg_type, content)

    @property
    def size(self):
        if self.type == self.INQUIRY_MESSAGE:
            return len(self.content)
        if self.type in [self.START_DCOP_MESSAGE, self.UPDATE_STATE_MESSAGE, self.FORWARDED_DCOP_EXECUTION]:
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
        self._mode = comp_def.algo.mode  # minimization or maximization
        self.initialize()

    def initialize(self):
        self.logger.debug(f'DCOP initialized')

        # for determining HOLD state
        self.beta = 1
        self.status = IDLE

        # monitor if DCOP process has begun
        self._dcop_started = False

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

    def start_dcop(self, neighbor_triggered=False):
        # check if this computation can start
        # if self._dcop_started:
        #     self.logger.debug(f'DCOP process already started. Neighbor triggered: {neighbor_triggered}')

        # if parent is available, relay execution call to parent
        parent = self.get_parent()
        should_forward = parent and parent not in self.done_state_history + self.hold_state_history
        no_neighbors = len(self.neighbors) == 0

        self.logger.debug(f'root: {not parent} or neighbor triggered: {neighbor_triggered}')

        if no_neighbors:
            self.logger.debug('isolated agent')
            self.select_value()

        elif should_forward:
            self.logger.debug('forwarding to parent')
            self.post_msg(
                target=parent,
                msg=CoCoAMessage(CoCoAMessage.FORWARDED_DCOP_EXECUTION, 'forwarded'),
            )

        elif not self.current_value:
            self._dcop_started = True

            # send inquiry messages
            msg = CoCoAMessage(CoCoAMessage.INQUIRY_MESSAGE, self.variable.domain.values)
            self.post_to_all_neighbors(msg, MSG_PRIORITY, on_error="fail")

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

    @register("dcop_initialization_message")
    def _on_dcop_initialization_message(self, sender: str, recv_msg, t: int):
        super(CoCoA, self)._on_dcop_initialization_message(sender, recv_msg, t)

    @register("dcop_execution_message")
    def _on_dcop_execution_message(self, sender: str, recv_msg, t: int):
        return super(CoCoA, self)._on_dcop_execution_message(sender, recv_msg, t)

    @register("dcop_configuration_message")
    def _on_dcop_configuration_message(self, sender: str, recv_msg, t: int):
        return super(CoCoA, self)._on_dcop_configuration_message(sender, recv_msg, t)

    @register("constraint_evaluation_response")
    def _on_constraint_evaluation_response(self, sender: str, recv_msg, t: int):
        super(CoCoA, self)._on_constraint_evaluation_response(sender, recv_msg, t)

    @register('agent_moved')
    def _on_agent_moved_msg(self, sender: str, recv_msg, t: int):
        super()._on_agent_moved_msg(sender, recv_msg, t)

    @register(CoCoAMessage.FORWARDED_DCOP_EXECUTION)
    def _on_forwarded_dcop_execution_message(self, sender, msg, t):
        self.logger.debug(f'Received forwarded message from {sender}')
        parent = self.get_parent()
        if parent:  # if parent is available, relay execution call to parent
            self.logger.debug('forwarding to parent')
            self.post_msg(
                target=parent,
                msg=CoCoAMessage(CoCoAMessage.FORWARDED_DCOP_EXECUTION, 'forwarded'),
            )
        # elif self._dcop_started:
        #     self.logger.debug(f'DCOP process already started. Forwarded exec from {sender}')
        else:
            self.logger.debug('Starting DCOP execution')
            self._dcop_started = True

            # send inquiry messages
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
            # var2.domain.values = recv_msg.content
        else:
            var1 = variables[1]
            var2 = variables[0]
            # var1.domain.values = recv_msg.content

        # Calculate the costs
        cost_map = np.zeros(constraint.shape)
        if self.current_value:
            self._calc_cost(constraint, cost_map, self.current_value, 0, var1, var2)
            cost_map_opt = cost_map[0, :]
        else:
            for i, d1 in enumerate(var1.domain.values):
                self._calc_cost(constraint, cost_map, d1, i, var1, var2)

            # reduce cost map to array
            if self._mode == 'min':
                cost_map_opt = cost_map.min(axis=0).tolist()
            else:
                cost_map_opt = cost_map.max(axis=0).tolist()

        # select the indices that yielded the min values
        if self._mode == 'min':
            cost_map_indices = cost_map.argmin(axis=0)
        else:
            cost_map_indices = cost_map.argmax(axis=0)

        # map the optimal value indices in the cost map to their domain values
        domain_values = list(self.variable.domain.values)
        cost_domain_vals = [self.current_value] * len(domain_values) if self.current_value \
            else [domain_values[i] for i in cost_map_indices]

        cost_msg = {"cost_map": cost_map_opt, "cost_domain_vals": cost_domain_vals}

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
                self.select_value()
            except Exception as e:
                self.logger.error(f'Aborting: {str(e)}')
        else:
            self.logger.debug('Number of expected cost messages not met')

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
        self.logger.debug(f"Received state update from {variable_name}: {recv_msg}, self.status = {self.status}")

        if recv_msg.content == HOLD:
            self.hold_state_history.append(variable_name)
            if self.current_value is None:
                self.start_dcop(neighbor_triggered=True)
            else:
                self.execute_neighbor_comp()

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
        self.logger.debug(f'Received start DCOP msg from {variable_name}: {recv_msg}')

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
        if self._mode == 'min':
            min_val = delta.min()
            opt_indices = np.asarray(delta == min_val).nonzero()[0]
            d_vals = np.array(self.variable.domain.values)
            rho = d_vals[opt_indices].tolist()
        else:
            max_val = delta.max()
            opt_indices = np.asarray(delta == max_val).nonzero()[0]
            d_vals = np.array(self.variable.domain.values)
            opt_indices = opt_indices[:len(d_vals)]
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
                idle_neighbors = set(self.neighbors) - set(self.hold_state_history + self.done_state_history)
                if idle_neighbors:
                    selected_neighbor = random.choice(list(idle_neighbors))
                    self.post_msg(selected_neighbor, CoCoAMessage(CoCoAMessage.UPDATE_STATE_MESSAGE, self.status))
                    self._dcop_started = False
                    self.cost_msgs.clear()
                else:
                    self.select_value()
        else:
            self.new_cycle()

            val = random.choice(rho)
            min_index = opt_indices[0]
            var_values = {self.name: val}

            # construct best values of all neighbors
            try:
                for agent in self.neighbors:
                    var_values[agent] = self.cost_msgs[agent]["cost_domain_vals"][min_index]

                # compute cost
                cost, _ = self._calculate_cost(var_values)
            except Exception as e:
                self.logger.error(f'Error calculating cost: {str(e)}')
                cost = 0.

            self.value_selection(val, cost)
            self.logger.debug(f"Value selected at {self.name} : value = {val}, cost = {cost}")

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
        available_neighbors = set(self.neighbors) - set(self.done_state_history)
        self.logger.debug(f'Selecting neighbor to start, available neighbors: {available_neighbors}')
        for neighbor in available_neighbors:
            self.logger.debug(f"Sending start-DCOP msg to neighbor {neighbor}")
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


