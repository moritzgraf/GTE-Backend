import time
import queue
import os
import threading

from gte_backend.settings import BASE_DIR


from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

import sequential_solver.solver.gametree
import sequential_solver.solver.solver as seq_solver
from sequential_solver.solver.status_manager import add_status, update_status, is_alive

WOLFRAM_TIMEOUT = 1 * 60
EQUATIONS_TIMEOUT = 1 * 60
WOLFRAM_QUEUE_MAXSIZE = 50
_wolfram_queue = queue.Queue(WOLFRAM_QUEUE_MAXSIZE)
_wolfram_queue_counter = 0
_worker_thread = None


def read():
    file_name = os.path.join(BASE_DIR, "sequential_solver/solver/example_input/game_to_read.ef")
    g = seq_solver.import_game(file_name)
    s = "Variable : Internal : Overwrite Name"
    for variable in g.variable_names:
        name = g.variable_names[variable]
        s += "\n" + name + " : " + variable
    s += "\n---"
    s += "\nPlayer : Payoff"
    for i in range(g.players):
        s += "\n" + str(i+1) + " : " + "P" + str(i+1) + "u"
    return s


def solve(id, config, variable_overwrites):
    stop_event = threading.Event()
    add_status(id, "Solving", "Starting solver...", stop_event)
    start_time = time.time()
    file_name = os.path.join(BASE_DIR, "sequential_solver/solver/example_input/game_to_solve.ef")
    g = seq_solver.import_game(file_name)
    lines = variable_overwrites.strip().split("\n")
    for line in lines:
        split = line.split(":", 1)
        if len(split) > 1:
            var = split[0].strip()
            name = split[1].strip()
            g.variable_names[var] = name
    args = _read_config(config)
    update_status(id, "Solving", "Calculating equilibrium equations...")
    equations = seq_solver.equilibria_equations(g,
                                                restrict_belief=args["restrict_belief"],
                                                restrict_strategy=args["restrict_strategy"],
                                                include_sequential=args["include_sequential"],
                                                filter=args["filter"],
                                                ed_method=args["ed_method"],
                                                ed_timeout=EQUATIONS_TIMEOUT)
    eq_time = time.time()
    if equations == "Timeout":
        update_status(id, "Aborted", "Calculating equilibrium equations timed out.")
    elif is_alive(id):
        update_status(id, "Queue", (_wolfram_queue.qsize(), _wolfram_queue_counter))
        job = [id, stop_event, (g, equations, args["include_sequential"], args["include_nash"])]
        _wolfram_queue.put(job)
        _ensure_worker_running()
        while not g.solutions:
            continue
        solve_time = time.time()
        if g.solutions == "Session Timeout":
            update_status(id, "Aborted", "Could not connect to WolframKernel.")
        elif g.solutions == "Calculation Timeout":
            update_status(id, "Aborted", "Solving of equations timed out.")
        elif is_alive(id):
            queue_time, connect_time = job[2]
            result = g.print_solutions(long=args["long"],
                                       include_types=args["include_types"])
            if args["time"]:
                result += "\n Time to construct equations: " + str(round(eq_time - start_time, 2)) + "s"
                result += "\n Time to solve equations: " + str(round(solve_time-connect_time, 2)) + "s"
                result += "\n Total calculation Time: " + str(round((eq_time - start_time) + (solve_time - connect_time), 2)) + "s"
                if args["time"] == "long":
                    result += "\n"
                    result += "\n Time spend waiting in Queue: " + str(round(queue_time - eq_time, 2)) + "s"
                    result += "\n Time spend connecting to Wolfram: " + str(round(connect_time - queue_time, 2)) + "s"
                    result += "\n Total time elapsed " + str(round(solve_time - start_time, 2)) + "s"
                    result += "\n"
            update_status(id, "Completed", result)


def estimate_queue_pos(queue_stamp):
    ahead_in_queue, counter = queue_stamp
    offset = 0
    if counter > _wolfram_queue_counter:
        offset = 2 * WOLFRAM_QUEUE_MAXSIZE
    remaining = 1 + ahead_in_queue - _wolfram_queue_counter + counter + offset
    return remaining


def _read_config(config):
    d = {"restrict_belief": "restrict_belief" in config,
         "restrict_strategy": "restrict_strategy" in config,
         "include_sequential": "include_sequential" in config,
         "include_nash": "include_nash" in config,
         "ed_method": "dd",
         "filter": "full",
         "time": "long",
         "long": False}
    include_types = []
    if d["include_nash"]:
        include_types.append("Nash Equilibria")
    if d["include_sequential"]:
        include_types.append("Sequential Equilibria")
    d["include_types"] = include_types
    return d


def _worker():
    session = WolframLanguageSession()
    while True:
        job = _wolfram_queue.get()
        id, stop_event, args = job
        g = args[0]
        update_status(id, "Solving", "Connecting to WolframKernel...")
        t1 = time.time()
        session = _ensure_working_session(session)
        t2 = time.time()
        if session is None:
            g.solutions = "Session Timeout"
        else:
            update_status(id, "Solving", "Solving equations using wolfram...")
            done_event = threading.Event()
            session_timeout_thread = threading.Thread(target=_session_timeout, args=(session, stop_event, done_event))
            session_timeout_thread.start()
            job[2] = (t1, t2)
            try:
                g.solutions = seq_solver.wolfram_solve_equations(*args, session)
            except Exception as e:
                print(e)
                g.solutions = "Calculation Timeout"
            done_event.set()

            global _wolfram_queue_counter
            _wolfram_queue_counter = _wolfram_queue_counter + 1 % (2 * WOLFRAM_QUEUE_MAXSIZE)
            _wolfram_queue.task_done()


def _session_timeout(session, stop_event, done_event):
    t = time.time()
    while not done_event.isSet():
        if stop_event.isSet() or time.time() - t > WOLFRAM_TIMEOUT:
            session.terminate()


def _ensure_working_session(session):
    start_time = time.time()
    while True:
        try:
            session.evaluate(wlexpr("1+1"))
            return session
        except:
            session = WolframLanguageSession()
        if time.time() - start_time > WOLFRAM_TIMEOUT:
            return None


def _ensure_worker_running():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        t = threading.Thread(target=_worker, args=())
        t.start()
        _worker_thread = t

