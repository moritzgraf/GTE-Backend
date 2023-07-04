import os
from bimatrix_solver.solver.solve_game import execute
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny

# Create your views here.
from rest_framework.response import Response

from gte_backend.settings import BASE_DIR

import threading
import uuid
import time

from sequential_solver.solver.solve_game import solve, read, estimate_queue_pos
from sequential_solver.solver.status_manager import get_status, is_alive


@api_view(["POST"])
@authentication_classes(())
@permission_classes((AllowAny, ))
def read_game(request):
    game_text = request.POST.get('game_text')
    file_name = os.path.join(BASE_DIR, "sequential_solver/solver/example_input/game_to_read.ef")
    file = open(file_name, "w+")
    file.write(str(game_text))
    file.close()
    s = read()
    return Response({"variable_names": s}, status=status.HTTP_201_CREATED)


@api_view(["POST"])
@authentication_classes(())
@permission_classes((AllowAny, ))
def solve_game(request):
    game_text = request.POST.get('game_text')
    config = request.POST.get('config')
    variable_overwrites = request.POST.get('variable_overwrites')
    file_name = os.path.join(BASE_DIR, "sequential_solver/solver/example_input/game_to_solve.ef")
    file = open(file_name, "w+")
    file.write(str(game_text))
    file.close()
    id = str(uuid.uuid4())
    solve_thread = threading.Thread(target=solve, args=(id, config, variable_overwrites))
    solve_thread.start()
    return Response({"id": id}, status=status.HTTP_202_ACCEPTED)


@api_view(["POST"])
@authentication_classes(())
@permission_classes((AllowAny, ))
def respond_status(request):
    time.sleep(1)
    id = request.POST.get("id")
    state, text = get_status(id)

    if state == "Solving":
        active = True
        output = text
    elif state == "Queue":
        remaining = estimate_queue_pos(text)
        active = True
        output = "Waiting in Queue for access to Wolfram. (" + str(remaining) + " in front)"
    elif state == "Completed":
        active = False
        output = text
    elif state == "Aborted":
        active = False
        output = text
    elif state == "Unknown":
        active = False
        output = text
    d = {"solver_output": output, "solver_status": state, "solver_active": active, "expected_id": id}
    return Response(d, status=status.HTTP_201_CREATED)



