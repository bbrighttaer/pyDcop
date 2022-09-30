from threading import Event

from pydcop.algorithms.fmddcop import ModelFreeDynamicDCOP
from pydcop.dcop.objects import AgentDef
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.communication import CommunicationLayer
from pydcop.infrastructure.computations import build_computation
from pydcop.infrastructure.discovery import Address
from pydcop.infrastructure.orchestratedagents import ORCHESTRATOR, ORCHESTRATOR_MGT, OrchestrationComputation
from pydcop.infrastructure.orchestrator import DeployMessage


class FMDDCOPAgent(Agent):

    def __int__(self, name: str, comm: CommunicationLayer, agent_def: AgentDef, ui_port=None):
        super(FMDDCOPAgent, self).__init__(name, comm, agent_def, ui_port=ui_port)

    def _on_start(self):
        self.logger.debug('FMDDCOP agent _on_start')
        return super(FMDDCOPAgent, self)._on_start()

    def _on_stop(self):
        super(FMDDCOPAgent, self)._on_stop()
        self.clean_shutdown()
        self.logger.info(f'Agent {self.name} stopped')


class OrchestratedFMDDCOPAgent(FMDDCOPAgent):

    def __init__(self,
                 agt_def: AgentDef,
                 comm: CommunicationLayer,
                 orchestrator_address: Address,
                 metrics_on: str = None,
                 metrics_period: float = None,
                 ui_port: int = None):
        super(OrchestratedFMDDCOPAgent, self).__init__(agt_def.name, comm, agt_def, ui_port)

        # set orchestrator information
        self.discovery.use_directory(ORCHESTRATOR, orchestrator_address)
        self.discovery.register_agent(ORCHESTRATOR, orchestrator_address, publish=False)
        self.discovery.register_computation(ORCHESTRATOR_MGT, ORCHESTRATOR, publish=False)

        # set management computation
        self._mgt_computation = FMDDCOPOrchestrationComputation(self)

        # periodic metrics
        self.metrics_on = metrics_on
        if metrics_on == 'period':
            self.set_periodic_action(metrics_period, self._mgt_computation.send_metrics)

        # Event to monitor when computation is ready to be used
        self.computation_ready_evt = Event()

    def _on_start(self):
        if not super()._on_start():
            return False
        self.add_computation(self._mgt_computation)
        self._mgt_computation.start()
        return True


class FMDDCOPOrchestrationComputation(OrchestrationComputation):

    def __init__(self, agent: FMDDCOPAgent):
        super(FMDDCOPOrchestrationComputation, self).__init__(agent)

        # add other handlers
        self._handlers.update({

        })

    def _on_deploy_computations(self, sender: str, msg: DeployMessage, t: float):
        """
        Deploys a new computation on this agent.

        Instantiate the computation, deploy it on the agent and register it
        for replication, if needed.

        Notes
        -----
        We cannot register immediately the neighbor computations as we do
        not know yet which agents are hosting them. Instead we initiate a
        discovery lookup for these computations and they will be registered
        once this lookup is completed.

        Parameters
        ----------

        comp_def: ComputationDef
            Definition of the computation
        """
        comp_def = msg.comp_def
        self.logger.info(
            "Deploying computations %s  on %s", comp_def.node, self.agent.name
        )
        computation = build_computation(comp_def)
        self.agent.add_computation(computation)

        if isinstance(computation, ModelFreeDynamicDCOP):
            if hasattr(self.agent, 'computation_ready_evt'):
                self.agent.computation_ready_evt.set()
            else:
                raise RuntimeError('computation_ready_evt attribute is required')


