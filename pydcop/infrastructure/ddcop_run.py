# BSD-3-Clause License
#
# Copyright 2017 Orange
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import logging
from multiprocessing import Process
from queue import Queue

from pydcop.algorithms import AlgorithmDef
from pydcop.computations_graph.objects import ComputationGraph
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import AgentDef
from pydcop.distribution.objects import Distribution
from pydcop.infrastructure.communication import InProcessCommunicationLayer, \
    HttpCommunicationLayer
from pydcop.infrastructure.orchestratedagents import OrchestratedAgent, DynamicOrchestratedAgent
from pydcop.infrastructure.orchestrator import Orchestrator, DynamicOrchestrator

# FIXME : need better infinity management
INFINITY = 10000


def run_local_thread_dcop(algo: AlgorithmDef,
                          cg: ComputationGraph,
                          distribution: Distribution,
                          dcop: DCOP,
                          infinity,  # FIXME : this has nothing to to here, #41
                          collector: Queue = None,
                          collect_moment: str = 'value_change',
                          period=None,
                          replication=None,
                          delay=None,
                          uiport=None,
                          stabilization_algorithm: str = None) -> DynamicOrchestrator:
    """Build orchestrator and agents for running a dcop in threads.

    The DCOP will be run in a single process, using one thread for each agent.

    Parameters
    ----------
    algo: AlgorithmDef
        Definition of DCOP algorithm, with associated parameters
    cg: ComputationGraph
        The computation graph used to solve the DCOP with the given algorithm
    distribution: Distribution
        Distribution of the computation on the agents
    dcop: DCOP
        The DCOP instance to solve
    infinity:
        FIXME : remove this!
    collector: queue
        optional queue, used to collect metrics
    collect_moment: str
        metric collection configuration : 'cycle_change', 'value_change' or
        'period'
    period: float
        period for collecting metrics, only used we 'period' metric collection
    replication
        replication algorithm,  for resilient DCOP.
    stabilization_algorithm
        If `use_dynamic_agents` is set to True, this parameter specifies which dynamic graph stabilization algorithm
        to use.

    Returns
    -------
    orchestrator
        An orchestrator agent that bootstrap dcop agents, monitor them and
        collects metrics.

    See Also
    --------
    Orchestrator, OrchestratedAgent
    run_local_process_dcopb


    """
    agents = dcop.agents
    comm = InProcessCommunicationLayer()

    orchestrator = DynamicOrchestrator(
        algo=algo, cg=cg, agent_mapping=distribution, comm=comm, dcop=dcop, infinity=infinity, collector=collector,
        collect_moment=collect_moment, collect_period=period, ui_port=uiport,
        stabilization_algorithm=stabilization_algorithm,
    )
    orchestrator.start()

    # Create and start all agents.
    # Each agent will register it-self on the orchestrator
    for a_name in dcop.agents:
        if uiport:
            uiport += 1
        comm = InProcessCommunicationLayer()
        agent = DynamicOrchestratedAgent(
                agt_def=agents[a_name],
                comm=comm,
                orchestrator_address=orchestrator.address,
                metrics_on=collect_moment,
                metrics_period=period,
                ui_port=uiport,
                stabilization_algorithm=stabilization_algorithm,
            )
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
                           delay=None,
                           uiport=None,
                           stabilization_algorithm: str = None
                           ) -> DynamicOrchestrator:
    agents = dcop.agents
    port = 9000
    comm = HttpCommunicationLayer(('127.0.0.1', port))

    orchestrator = DynamicOrchestrator(
        algo=algo, cg=cg, agent_mapping=distribution, comm=comm, dcop=dcop, infinity=infinity, collector=collector,
        collect_moment=collect_moment, collect_period=period, ui_port=uiport,
        stabilization_algorithm=stabilization_algorithm,
    )
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
                            'delay': delay,
                            'uiport': uiport,
                            'stabilization_algorithm': stabilization_algorithm},
                    daemon=True)
        p.start()

    # once all agents have started and registered to the orchestrator,
    # computation will be deployed on them and then run.
    return orchestrator


def _build_process_agent(agt_def: AgentDef, port, orchestrator_address,
                         metrics_on, metrics_period,
                         delay, uiport, stabilization_algorithm):
    comm = HttpCommunicationLayer(('127.0.0.1', port))
    agent = DynamicOrchestratedAgent(
        agt_def=agt_def,
        comm=comm,
        orchestrator_address=orchestrator_address,
        metrics_on=metrics_on,
        metrics_period=metrics_period,
        ui_port=uiport,
        stabilization_algorithm=stabilization_algorithm,
    )

    # Disable all non-error logging for agent's processes, we don't want
    # all agents trying to log in the same console
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.ERROR)
    # root_logger = logging.getLogger('')
    # root_logger.setLevel(logging.ERROR)
    # root_logger.addHandler(console_handler)

    agent.start()
