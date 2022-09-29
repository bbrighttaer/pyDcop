from queue import Queue

from pydcop.algorithms import AlgorithmDef
from pydcop.computations_graph.objects import ComputationGraph
from pydcop.dcop.dcop import DCOP
from pydcop.distribution.objects import Distribution
from pydcop.infrastructure.communication import CommunicationLayer
from pydcop.infrastructure.orchestrator import Orchestrator, AgentsMgt


class RoboCupOrchestrator(Orchestrator):

    def __init__(self, algo: AlgorithmDef,
                 cg: ComputationGraph,
                 agent_mapping: Distribution,
                 comm: CommunicationLayer,
                 dcop: DCOP,
                 infinity=float('inf'),
                 collector: Queue = None,
                 collect_moment: str = 'value_change',
                 collect_period: float = None,
                 ui_port: int = None):
        super(RoboCupOrchestrator, self).__init__(
            algo=algo, cg=cg, agent_mapping=agent_mapping, comm=comm, infinity=infinity,
            collector=collector, collect_moment=collect_moment,
            ui_port=ui_port, dcop=dcop,
        )

        self.mgt = RoboCupAgentsMgt(
            algo, cg, agent_mapping, dcop, self._own_agt, self, infinity,
            collector=collector, collect_moment=collect_moment, collect_period=collect_period,
        )

    def start(self):
        super(RoboCupOrchestrator, self).start()
        self.logger.debug('RoboCupOrchestrator started')


class RoboCupAgentsMgt(AgentsMgt):

    def __init__(self, *args, **kwargs):
        super(RoboCupAgentsMgt, self).__init__(*args, **kwargs)

        # add extra msg handlers
        self._msg_handlers.update({

        })
