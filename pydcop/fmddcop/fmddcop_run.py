import logging
from multiprocessing import Process
from queue import Queue

from pydcop.algorithms import AlgorithmDef
from pydcop.computations_graph.objects import ComputationGraph
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import AgentDef
from pydcop.distribution.objects import Distribution
from pydcop.fmddcop.robocup.robocup_agent_bridge import RoboCupAgent
from pydcop.infrastructure.communication import InProcessCommunicationLayer, HttpCommunicationLayer
from pydcop.fmddcop.fmddcopagent import OrchestratedFMDDCOPAgent
from pydcop.fmddcop.fmddcoporchestrator import FMDDCOPOrchestrator


def run_local_thread_dcop(algo: AlgorithmDef,
                          cg: ComputationGraph,
                          distribution: Distribution,
                          dcop: DCOP,
                          infinity,  # FIXME : this has nothing to to here, #41
                          collector: Queue = None,
                          collect_moment: str = 'value_change',
                          period=None,
                          delay=None,
                          uiport=None) -> FMDDCOPOrchestrator:
    agents = dcop.agents
    comm = InProcessCommunicationLayer()
    orchestrator = FMDDCOPOrchestrator(algo, cg, distribution, comm, dcop, infinity,
                                       collector=collector,
                                       collect_moment=collect_moment,
                                       collect_period=period,
                                       ui_port=uiport)
    orchestrator.start()

    # Create and start all agents.
    # Each agent will register it-self on the orchestrator
    for a_name in dcop.agents:
        if uiport:
            uiport += 1
        comm = InProcessCommunicationLayer()
        agent = RoboCupAgent(agents[a_name], comm,
                             orchestrator.address,
                             metrics_on=collect_moment,
                             metrics_period=period,
                             ui_port=uiport)
        agent.start()

    # once all agents have started and registered to the orchestrator,
    # computation will be deployed on them and then run.
    return orchestrator


def run_local_process_dcop(algo: AlgorithmDef, cg: ComputationGraph,
                           distribution: Distribution, dcop: DCOP,
                           infinity,  # FIXME : this has nothing to to here, #41
                           collector: Queue = None,
                           collect_moment: str = 'value_change',
                           period=None,
                           replication=None,
                           delay=None,
                           uiport=None
                           ):
    agents = dcop.agents
    port = 9000
    comm = HttpCommunicationLayer(('127.0.0.1', port))
    orchestrator = FMDDCOPOrchestrator(algo, cg, distribution, comm, dcop, infinity,
                                       collector=collector,
                                       collect_moment=collect_moment,
                                       collect_period=period,
                                       ui_port=uiport)
    orchestrator.start()

    # Create and start all agents.
    # Each agent will register it-self on the orchestrator
    for a_name in dcop.agents:
        port += 1
        if uiport:
            uiport += 1
        p = Process(target=_build_process_agent, name='p_' + a_name,
                    args=[agents[a_name], port, orchestrator.address],
                    kwargs={'metrics_on': collect_moment,
                            'metrics_period': period,
                            'replication': replication,
                            'delay': delay,
                            'uiport': uiport},
                    daemon=True)
        p.start()

    # once all agents have started and registered to the orchestrator,
    # computation will be deployed on them and then run.
    return orchestrator


def _build_process_agent(agt_def: AgentDef, port, orchestrator_address,
                         metrics_on, metrics_period, replication,
                         delay, uiport):
    comm = HttpCommunicationLayer(('127.0.0.1', port))
    agent = RoboCupAgent(agt_def, comm, orchestrator_address,
                         metrics_on=metrics_on,
                         metrics_period=metrics_period,
                         ui_port=uiport)

    # Disable all non-error logging for agent's processes, we don't want
    # all agents trying to log in the same console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.ERROR)
    root_logger.addHandler(console_handler)

    agent.start()
