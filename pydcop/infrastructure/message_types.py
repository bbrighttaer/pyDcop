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
