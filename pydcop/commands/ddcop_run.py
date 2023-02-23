import json
import logging
import multiprocessing
import sys
import threading
import traceback
from queue import Queue
from threading import Thread

from pydcop.algorithms import list_available_algorithms
from pydcop.commands._utils import _error, prepare_metrics_files, _load_modules, build_algo_def, collect_tread, \
    add_csvline
from pydcop.commands.run import NumpyEncoder
from pydcop.dcop.dcop import filter_dcop
from pydcop.dcop.yamldcop import load_dcop_from_file, load_scenario_from_file
from pydcop.computations_graph import dynamic_graph
from pydcop.distribution.yamlformat import load_dist_from_file
from pydcop.envs.mobile_sensing import GridWorld
from pydcop.infrastructure.ddcop_run import run_local_process_dcop, run_local_thread_dcop

logger = logging.getLogger("pydcop.cli.ddcop_run")


def set_parser(subparsers):
    algorithms = list_available_algorithms()
    logger.debug(f"Available DCOP algorithms {algorithms}")
    parser = subparsers.add_parser("ddcop_run", help="run a DDCOP simulation")

    parser.set_defaults(func=run_cmd)
    parser.set_defaults(on_timeout=on_timeout)
    parser.set_defaults(on_force_exit=on_force_exit)

    parser.add_argument("dcop_files", type=str, nargs="+", help="dcop file")

    parser.add_argument(
        "-a",
        "--algo",
        required=True,
        choices=algorithms,
        help="algorithm for solving the dcop",
    )
    parser.add_argument(
        "-p",
        "--algo_params",
        type=str,
        action="append",
        help="Optional parameters for the algorithm, given as "
             "name:value. Use this option several times "
             "to set several parameters.",
    )

    parser.add_argument(
        "-d",
        "--distribution",
        required=True,
        help="distribution of the computations on agents, " "as a yaml file ",
    )

    parser.add_argument("-s", "--scenario", required=True, help="scenario file")

    parser.add_argument(
        "-m",
        "--mode",
        default="thread",
        choices=["thread", "process"],
        help="run agents as threads or processes",
    )

    # Statistics collection arguments:
    parser.add_argument(
        "-c",
        "--collect_on",
        choices=["value_change", "cycle_change", "period"],
        default="value_change",
        help='When should a "new" assignment be observed',
    )
    parser.add_argument(
        "--period",
        type=float,
        default=None,
        help="Period for collecting metrics. only available "
             "when using --collect_on period. Defaults to 1 "
             "second if not specified",
    )
    parser.add_argument(
        "--run_metrics",
        type=str,
        default=None,
        help="Use this option to regularly store the data " "in a csv file.",
    )
    parser.add_argument(
        "--end_metrics",
        type=str,
        default=None,
        help="Use this option to append the metrics of the "
             "end of the run to a csv file.",
    )
    parser.add_argument(
        "--infinity",
        "-i",
        default=float("inf"),
        type=float,
        help="Argument to determine the value used for "
             "infinity in case of hard constraints, "
             "for algorithms that do not use symbolic "
             "infinity. Defaults to 10 000",
    )
    parser.add_argument(
        "--stabilization_alg",
        "-b",
        default="digca",
        dest="stabilization_algorithm",
        type=str,
        help="Dynamic graph stabilization algorithm",
    )
    parser.add_argument(
        "--grid_size",
        "-g",
        default=2,
        dest="grid_size",
        type=int,
        help="Size of the GridWorld",
    )
    parser.add_argument(
        "--num_targets",
        "-k",
        default=2,
        dest="num_targets",
        type=int,
        help="Number of targets in the GridWorld",
    )


dcop = None
orchestrator = None
INFINITY = None

collect_on = None
run_metrics = None
end_metrics = None

timeout_stopped = False
output_file = None

DISTRIBUTION_METHODS = [
    "oneagent", "adhoc", "ilp_fgdp", "heur_comhost", "oilp_secp_fgdp",
    "gh_secp_fgdp", "gh_secp_cgdp", "oilp_cgdp", "gh_cgdp"
]


def run_cmd(args, timer=None, timeout=None):
    logger.debug(f'dcop command "ddcop_run" with arguments {args}')

    global INFINITY, collect_on, output_file, run_metrics, end_metrics
    INFINITY = args.infinity
    collect_on = args.collect_on
    output_file = args.output
    run_metrics = args.run_metrics
    end_metrics = args.end_metrics

    period = None
    if args.collect_on == "period":
        period = 1 if args.period is None else args.period
    elif args.period is not None:
        _error('Cannot use "period" argument when collect_on is not ' '"period"')

    csv_cb = prepare_metrics_files(run_metrics, end_metrics, collect_on)

    _, algo_module, graph_module = _load_modules(None, args.algo)

    global dcop
    logger.info(f"loading dcop from {args.dcop_files}")
    dcop = load_dcop_from_file(args.dcop_files)

    # dcop = filter_dcop(dcop)

    if args.distribution in DISTRIBUTION_METHODS:
        dist_module, algo_module, graph_module = _load_modules(
            args.distribution, args.algo
        )
    else:
        dist_module, algo_module, graph_module = _load_modules(None, args.algo)

    logger.info("loading scenario from {}".format(args.scenario))
    scenario = load_scenario_from_file(args.scenario)

    logger.info("Building computation graph")
    cg = dynamic_graph.build_computation_graph(dcop, graph_module=graph_module)

    logger.info("Distributing computation graph ")
    if dist_module is not None:
        distribution = dist_module.distribute(
            cg,
            dcop.agents.values(),
            hints=dcop.dist_hints,
            computation_memory=algo_module.computation_memory,
            communication_load=algo_module.communication_load,
        )
    else:
        distribution = load_dist_from_file(args.distribution)
    logger.debug("Distribution Computation graph: %s ", distribution)

    algo = build_algo_def(algo_module, args.algo, dcop.objective, args.algo_params)

    # Setup metrics collection
    collector_queue = Queue()
    collect_t = Thread(
        target=collect_tread, args=[collector_queue, csv_cb], daemon=True
    )
    collect_t.start()

    # D-DCOP environment setup
    simulation_environment = GridWorld(
        size=10,
        num_targets=5,
        scenario=scenario,
    )
    dcop.simulation_environment = simulation_environment

    global orchestrator
    if args.mode == "thread":
        orchestrator = run_local_thread_dcop(
            algo,
            cg,
            distribution,
            dcop,
            INFINITY,
            collector=collector_queue,
            collect_moment=args.collect_on,
            period=period,
            stabilization_algorithm=args.stabilization_algorithm,
            sim_env=simulation_environment,
        )
    elif args.mode == "process":

        # Disable logs from agents, they are in other processes anyway
        agt_logs = logging.getLogger("pydcop.agent")
        agt_logs.disabled = True

        # When using the (default) 'fork' start method, http servers on agent's
        # processes do not work (why ?)
        multiprocessing.set_start_method("spawn")
        orchestrator = run_local_process_dcop(
            algo,
            cg,
            distribution,
            dcop,
            INFINITY,
            collector=collector_queue,
            collect_moment=args.collect_on,
            period=period,
            stabilization_algorithm=args.stabilization_algorithm,
            sim_env=simulation_environment,
        )

    orchestrator.set_error_handler(_orchestrator_error)

    try:
        orchestrator.deploy_computations()
        if orchestrator.wait_ready():
            orchestrator.run(scenario, timeout=timeout)
            if timer:
                timer.cancel()
            if not timeout_stopped:
                if orchestrator.status == "TIMEOUT":
                    _results("TIMEOUT")
                    sys.exit(0)
                elif orchestrator.status != "STOPPED":
                    _results("FINISHED")
                    sys.exit(0)

    except Exception as e:
        logger.error(e, exc_info=1)
        print(e)
        for th in threading.enumerate():
            print(th)
            traceback.print_stack(sys._current_frames()[th.ident])
            print()
        orchestrator.stop_agents(5)
        orchestrator.stop()
        _results("ERROR")


def _orchestrator_error(e):
    print("Error in orchestrator: \n ", e)
    sys.exit(2)


def _results(status):
    """
    Outputs results and metrics on stdout and trace last metrics in csv
    files if requested.

    :param status:
    :return:
    """
    metrics = orchestrator.end_metrics()
    metrics["status"] = status
    global end_metrics, run_metrics
    if end_metrics is not None:
        add_csvline(end_metrics, collect_on, metrics)
    if run_metrics is not None:
        add_csvline(run_metrics, collect_on, metrics)

    if output_file:
        with open(output_file, encoding="utf-8", mode="w") as fo:
            fo.write(json.dumps(metrics, sort_keys=True, indent="  ", cls=NumpyEncoder))
    else:
        print(json.dumps(metrics, sort_keys=True, indent="  ", cls=NumpyEncoder))


def on_timeout():
    if orchestrator is None:
        return
    # Timeout should have been handled by the orchestrator, if the cli timeout
    # has been reached, something is probably wrong : dump threads.
    for th in threading.enumerate():
        print(th)
        traceback.print_stack(sys._current_frames()[th.ident])
        print()
    if orchestrator is None:
        logger.debug("cli timeout with no orchestrator ?")
        return
    global timeout_stopped
    timeout_stopped = True

    # Stopping agents can be rather long, we need a big timeout !
    orchestrator.stop_agents(20)
    orchestrator.stop()
    _results("TIMEOUT")
    sys.exit(0)


def on_force_exit(sig, frame):
    if orchestrator is None:
        return
    orchestrator.status = "STOPPED"
    orchestrator.stop_agents(5)
    orchestrator.stop()
    _results("STOPPED")