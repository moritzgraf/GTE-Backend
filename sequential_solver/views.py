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
import uuid
import time
from wolframclient.evaluation import WolframLanguageSession

import sequential_solver.solver.gametree
import sequential_solver.solver.solver as seq_solver
RESPONSE_TIMEOUT = 10
WOLFRAM_TIMEOUT = 5 * 60
WOLFRAM_QUEUE_MAXSIZE = 20
wolframQueue = queue.Queue(WOLFRAM_QUEUE_MAXSIZE)
wolframQueueCounter = 0
wolfram_thread = None
statusManager = None


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
def respond_status(request):
    global statusManager
    if not statusManager:
        statusManager = StatusManager()
    time.sleep(1)
    id = request.POST.get("id")
    _, solver_status, result = statusManager.get(id)
    active = True
    if solver_status in ["Completed", "Aborted", "Error"]:
        active = False
        statusManager.set(id, time=-1)
    elif not solver_status == "Unknown":
        statusManager.set(id, time=time.time())

    if solver_status == "Queue":
        s = result.split(";", 1)
        old_position, old_counter = int(s[0]), int(s[1])
        new_counter = wolframQueueCounter
        if old_counter > new_counter:
            new_counter += (2 * WOLFRAM_QUEUE_MAXSIZE)
        est_queue_position = old_position - (old_counter - new_counter)
        result = "Waiting for access in Queue for access to Wolfram at position " + str(est_queue_position + 1)
    return Response({"solver_output": result, "solver_status": solver_status, "solver_active": active, "expected_id": id}, status=status.HTTP_201_CREATED)


@api_view(["POST"])
@authentication_classes(())
@permission_classes((AllowAny, ))
def solve_game(request):
    global statusManager
    if not statusManager:
        statusManager = StatusManager()
    game_text = request.POST.get('game_text')
    config = request.POST.get('config')
    variable_overwrites = request.POST.get('variable_overwrites')
    id = str(uuid.uuid4())
    statusManager.set(id, time=time.time(), status="Calculating", text="Calculating equilibria equations...")
    thread = threading.Thread(target=start_solving, args=(game_text, config, variable_overwrites, id))
    thread.start()
    return Response({"id": id}, status=status.HTTP_202_ACCEPTED)


def start_solving(game_text, config, variable_overwrites, id):
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

    statusManager.set(id, status="Queue", text=str(wolframQueue.qsize()) + ";" + str(wolframQueueCounter))
    element = [(g, equations, include_sequential, include_nash), id]
    wolframQueue.put(element)


class StatusManager:

    def __init__(self):
        self.statusManager = queue.Queue()
        self.status_dict = {}
        self.worker_thread = None

    def start(self):
        if not self.worker_thread or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker, args=())
            self.worker_thread.start()

    def get(self, id):
        job = [id, "get", None]
        self.statusManager.put(job)
        self.start()
        while not job[2]:
            continue
        return job[2]

    def set(self, id, time="", status="", text=""):
        job = [id, "set", [time, status, text]]
        self.statusManager.put(job)
        self.start()

    def worker(self):
        while True:
            job = self.statusManager.get()
            id, mode, value = job
            if mode == "set":
                if id in self.status_dict:
                    t, status, text = self.status_dict[id]
                    if value[0]:
                        t = value[0]
                    if value[1]:
                        status = value[1]
                    if value[2]:
                        text = value[2]
                    self.status_dict[id] = [t, status, text]
                else:
                    self.status_dict[id] = value.copy()
            elif mode == "get":
                if id in self.status_dict:
                    job[2] = self.status_dict[id]
                else:
                    job[2] = [-1, "Unknown", "Request Lost"]
            self.statusManager.task_done()
            timed_out = []
            for id in self.status_dict:
                t = self.status_dict[id][0]
                if t == -1 or time.time() - t > RESPONSE_TIMEOUT:
                    timed_out.append(id)
            for id in timed_out:
                del self.status_dict[id]


def wolfram_queue():
    global statusManager
    session = WolframLanguageSession()
    while True:
        element = wolframQueue.get()
        id = element[1]
        wolfram_timeout_event = threading.Event()
        response_timeout_event = threading.Event()
        statusManager.set(id, status="Solving", text="Solving equations using Wolfram")
        thread = threading.Thread(target=worker, args=(element, session, wolfram_timeout_event, response_timeout_event))
        thread.start()
        t = 0
        while thread.is_alive():
            if t > WOLFRAM_TIMEOUT:
                wolfram_timeout_event.set()
                session.terminate()
                break
            timestamp, _, _ = statusManager.get(id)
            if timestamp == -1:
                response_timeout_event.set()
                session.terminate()
                break
            time.sleep(1)
            t += 1
        wolframQueue.task_done()
        global wolframQueueCounter
        wolframQueueCounter = wolframQueueCounter + 1 % (2 * WOLFRAM_QUEUE_MAXSIZE)
    session.terminate()


def worker(element, session, wolfram_timeout_event, response_timeout_event):
    global statusManager
    g, equations, include_sequential, include_nash = element[0]
    id = element[1]
    try:
        g.solutions = seq_solver.wolfram_solve_equations(g, equations, include_sequential, session)
        include_types = []
        if include_nash:
            include_types.append("Nash Equilibria")
        if include_sequential:
            include_types.append("Sequential Equilibria")
        result = g.print_solutions(long=False, include_types=include_types)
        statusManager.set(id, status="Completed", text=result)
    except Exception as e:
        if wolfram_timeout_event.isSet():
            statusManager.set(id, status="Aborted", text="Wolfram calculations timed out.")
        elif not response_timeout_event.isSet():
            statusManager.set(id, status="Error", text=str(e))
