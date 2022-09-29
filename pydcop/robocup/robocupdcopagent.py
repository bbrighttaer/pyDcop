from pydcop.dcop.objects import AgentDef
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.communication import CommunicationLayer
from pydcop.infrastructure.discovery import Address
from pydcop.infrastructure.orchestratedagents import ORCHESTRATOR, ORCHESTRATOR_MGT, OrchestrationComputation


class RoboCupDCOPAgent(Agent):

    def __int__(self, name: str, comm: CommunicationLayer, agent_def: AgentDef, ui_port=None):
        super(RoboCupDCOPAgent, self).__init__(name, comm, agent_def, ui_port=ui_port)

    def _on_start(self):
        self.logger.debug('Robocup agent _on_start')
        return super(RoboCupDCOPAgent, self)._on_start()

    def _on_stop(self):
        super(RoboCupDCOPAgent, self)._on_stop()
        self.clean_shutdown()
        self.logger.info(f'Agent {self.name} stopped')


class OrchestratedRoboCupDCOPAgent(RoboCupDCOPAgent):

    def __init__(self,
                 agt_def: AgentDef,
                 comm: CommunicationLayer,
                 orchestrator_address: Address,
                 metrics_on: str = None,
                 metrics_period: float = None,
                 ui_port: int = None):
        super(OrchestratedRoboCupDCOPAgent, self).__init__(agt_def.name, comm, agt_def, ui_port)

        # set orchestrator information
        self.discovery.use_directory(ORCHESTRATOR, orchestrator_address)
        self.discovery.register_agent(ORCHESTRATOR, orchestrator_address, publish=False)
        self.discovery.register_computation(ORCHESTRATOR_MGT, ORCHESTRATOR, publish=False)

        # set management computation
        self._mgt_computation = RoboCupDCOPOrchestrationComputation(self)

        # periodic metrics
        self.metrics_on = metrics_on
        if metrics_on == 'period':
            self.set_periodic_action(metrics_period, self._mgt_computation.send_metrics)

    def _on_start(self):
        if not super()._on_start():
            return False
        self.add_computation(self._mgt_computation)
        self._mgt_computation.start()
        return True


class RoboCupDCOPOrchestrationComputation(OrchestrationComputation):

    def __init__(self, agent: RoboCupDCOPAgent):
        super(RoboCupDCOPOrchestrationComputation, self).__init__(agent)

        # add other handlers
        self._handlers.update({

        })

