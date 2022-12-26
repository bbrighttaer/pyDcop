from pydcop.infrastructure.computations import message_type

ConstraintEvaluationRequest = message_type(
    'constraint_evaluation_request', ['constraint_name', 'var_assignments'],
)

ConstraintEvaluationResponse = message_type(
    'constraint_evaluation_response', ['constraint_name', 'value'],
)

AgentMovedMessage = message_type(
    'agent_moved', ['position']
)

# SimTimeStepChange is sent by the orchestrator to all agents when the simulation environment changes its times step
SimTimeStepChanged = message_type(
    'sim_time_step_change', ['data'],
)

# Sent from a stabilization computation to DCOP computation(s)
DcopExecutionMessage = message_type(
    'dcop_execution_message', [],
)

DcopConfigurationMessage = message_type(
    'dcop_configuration_message', ['data'],
)

DcopInitializationMessage = message_type(
    'dcop_initialization_message', [],
)
