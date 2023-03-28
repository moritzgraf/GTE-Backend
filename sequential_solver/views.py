import os
from bimatrix_solver.solver.solve_game import execute
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny

# Create your views here.
from rest_framework.response import Response

from gte_backend.settings import BASE_DIR

import queue
import signal
import threading
from time import sleep
from wolframclient.evaluation import WolframLanguageSession

import sequential_solver.solver.gametree
import sequential_solver.solver.solver as seq_solver
WOLFRAM_TIMEOUT = 5 * 60
wolframQueue = queue.Queue()
queue_running = False
wolfram_thread = None

@api_view(["POST"])
@authentication_classes(())
@permission_classes((AllowAny, ))
def read_game(request):
    game_text = request.POST.get('game_text')
    file_name = os.path.join(BASE_DIR, "sequential_solver/solver/example_input/game_to_read.ef")
    file = open(file_name, "w+")
    file.write(str(game_text))
    file.close()
    g = seq_solver.import_game(file_name)
    s = "Variable : Internal : Overwrite Name"
    for variable in g.variable_names:
        name = g.variable_names[variable]
        s += "\n" + name + " : " + variable
    s += "\n---"
    s += "\nPlayer : Payoff"
    for i in range(g.players):
        s += "\n" + str(i+1) + " : " + "P" + str(i+1) + "u"
    return Response({"variable_names": s}, status=status.HTTP_201_CREATED)


@api_view(["POST"])
@authentication_classes(())
@permission_classes((AllowAny, ))
def solve_game(request):
    game_text = request.POST.get('game_text')
    config = request.POST.get('config')
    variable_overwrites = request.POST.get('variable_overwrites')
    print(config)
    file_name = os.path.join(BASE_DIR, "sequential_solver/solver/example_input/game_to_solve.ef")
    file = open(file_name, "w+")
    file.write(str(game_text))
    file.close()
    include_nash = "include_nash" in config
    include_sequential = "include_sequential" in config
    restrict_belief = "restrict_belief" in config
    restrict_strategy = "restrict_strategy" in config
    g = seq_solver.import_game(file_name)
    lines = variable_overwrites.strip().split("\n")
    for line in lines:
        print(line)
        split = line.split(":", 1)
        if len(split) > 1:
            var = split[0].strip()
            name = split[1].strip()
            g.variable_names[var] = name
    equations = seq_solver.equilibria_equations(g,
                                     restrict_belief=restrict_belief,
                                     restrict_strategy=restrict_strategy,
                                     onlynash = not include_sequential,
                                     weak_filter=False,
                                     ed_method="dd")

    global wolfram_thread
    if not wolfram_thread or not wolfram_thread.is_alive():
        wolfram_thread = threading.Thread(target=wolfram_queue, args=())
        wolfram_thread.start()

    element = [(g, equations, include_sequential), None, "waiting"]
    wolframQueue.put(element)
    while element[2] == "waiting":
        sleep(1)
    if element[2] == "completed":
        g.solutions = element[1]
        include_types = []
        if include_nash:
            include_types.append("Nash Equilibria")
        if include_sequential:
            include_types.append("Sequential Equilibria")
        result = g.print_solutions(long=False, include_types=include_types)
    elif element[2] == "timeout":
        result = "Request timed out performing mathematica calculations after " + str(WOLFRAM_TIMEOUT) + "s"
    else:
        result = "Something went wrong: " + str(element[1])

    return Response({"solver_output": result}, status=status.HTTP_201_CREATED)


def wolfram_queue():
    session = WolframLanguageSession()
    while True:
        element = wolframQueue.get()
        completion_event = threading.Event()
        timeout_event = threading.Event()
        thread = threading.Thread(target=worker, args=(element, session, completion_event, timeout_event))
        thread.start()
        t = 0
        while thread.is_alive() and not completion_event.isSet():
            if t > WOLFRAM_TIMEOUT:
                timeout_event.set()
                session.terminate()
                break
            sleep(1)
            t += 1
        wolframQueue.task_done()
    session.terminate()


def worker(element, session, completion_event, timeout_event):
    args = element[0]
    try:
        solutions = seq_solver.wolfram_solve_equations(*args, session)
        completion_event.set()
        element[1] = solutions
        element[2] = "completed"
    except Exception as e:
        if timeout_event.isSet():
            element[1] = ""
            element[2] = "timeout"
        else:
            element[1] = e
            element[2] = "error"