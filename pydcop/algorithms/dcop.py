import random

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
    computation = DummyDCOP(comp_def)
    return computation


def computation_memory(*args):
    raise NotImplementedError("CoCoA has no computation memory implementation (yet)")


def communication_load(*args):
    raise NotImplementedError("CoCoA has no communication_load implementation (yet)")


class DCOPMessage(Message):
    INQUIRY_MESSAGE = "InquiryMessage"
    COST_MESSAGE = "CostMessage"
    UPDATE_STATE_MESSAGE = "UpdateStateMessage"
    START_DCOP_MESSAGE = "StartDCOP"
    FORWARDED_DCOP_EXECUTION = 'ForwardedDCOPExecution'

    def __init__(self, msg_type, content):
        super(DCOPMessage, self).__init__(msg_type, content)

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


class DummyDCOP(VariableComputation, DynamicDcopComputationMixin):
    """
    This simulates the operation of a DCOP algorithm.

    Parameters
    ----------
    comp_def: ComputationDef
    The computation definition which specifies the assigned variable, neighbors, and constraints.
    """

    def __init__(self, comp_def: ComputationDef):

        super().__init__(comp_def.node.variable, comp_def)
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
        if self.current_value is None:
            self.select_random_value()

        # select next neighbor to execute
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

    @register("dcop_initialization_message")
    def _on_dcop_initialization_message(self, sender: str, recv_msg, t: int):
        super()._on_dcop_initialization_message(sender, recv_msg, t)

    @register("dcop_execution_message")
    def _on_dcop_execution_message(self, sender: str, recv_msg, t: int):
        return super()._on_dcop_execution_message(sender, recv_msg, t)

    @register("dcop_configuration_message")
    def _on_dcop_configuration_message(self, sender: str, recv_msg, t: int):
        return super(DummyDCOP, self)._on_dcop_configuration_message(sender, recv_msg, t)

    @register("constraint_evaluation_response")
    def _on_constraint_evaluation_response(self, sender: str, recv_msg, t: int):
        super()._on_constraint_evaluation_response(sender, recv_msg, t)

    @register('agent_moved')
    def _on_agent_moved_msg(self, sender: str, recv_msg, t: int):
        super()._on_agent_moved_msg(sender, recv_msg, t)

    @register(DCOPMessage.FORWARDED_DCOP_EXECUTION)
    def _on_forwarded_dcop_execution_message(self, sender, msg, t):
        self.logger.debug(f'Received forwarded message from {sender}')
        parent = self.get_parent()
        if parent:  # if parent is available, relay execution call to parent
            self.logger.debug('forwarding to parent')
            self.post_msg(
                target=parent,
                msg=DCOPMessage(DCOPMessage.FORWARDED_DCOP_EXECUTION, 'forwarded'),
            )
        # elif self._dcop_started:
        #     self.logger.debug(f'DCOP process already started. Forwarded exec from {sender}')
        else:
            self.logger.debug('Starting DCOP execution')
            self._dcop_started = True

            # send inquiry messages
            msg = DCOPMessage(DCOPMessage.INQUIRY_MESSAGE, self.variable.domain.values)
            self.post_to_all_neighbors(msg, MSG_PRIORITY, on_error="fail")

    @register(DCOPMessage.UPDATE_STATE_MESSAGE)
    def _on_update_state_message(self, variable_name: str, recv_msg: DCOPMessage, t: int):
        """
        Receives an update state message from a neighbor and update appropriate registers.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DCOPMessage
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

    @register(DCOPMessage.START_DCOP_MESSAGE)
    def _on_start_dcop_msg(self, variable_name: str, recv_msg: DCOPMessage, t: int):
        """
        Triggered by a neighbor to initiate the DCOP computation.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DCOPMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(f'Received start DCOP msg from {variable_name}: {recv_msg}')

        self.start_dcop(neighbor_triggered=True)

    def select_random_value(self):
        self.new_cycle()
        value = random.choice(self.variable.domain.values)
        self.logger.debug(f'selected value: {value}')
        cost = 0.
        self.value_selection(value, cost)
        self.logger.debug(f"Value selected at {self.name} : value = {value}, cost = {cost}")

        # update neighbors
        self.status = DONE
        self.post_to_all_neighbors(DCOPMessage(DCOPMessage.UPDATE_STATE_MESSAGE, self.status), on_error="fail")

    def execute_neighbor_comp(self):
        """
        Randomly select a neighbor to trigger execution
        """
        available_neighbors = set(self.neighbors) - set(self.done_state_history)
        self.logger.debug(f'Selecting neighbor to start, available neighbors: {available_neighbors}')
        for neighbor in available_neighbors:
            self.logger.debug(f"Sending start-DCOP msg to neighbor {neighbor}")
            self.post_msg(neighbor, DCOPMessage(DCOPMessage.START_DCOP_MESSAGE, None), on_error="fail")

        if self.computation_def.exec_mode != 'dynamic':
            self.stop()
            self.finished()

    def finished(self):
        self.done_state_history.clear()
        self.cost_msgs.clear()
        self.status = IDLE
        self.hold_state_history.clear()

        super().finished()
