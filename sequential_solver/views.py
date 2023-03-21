import subprocess

import os
from bimatrix_solver.solver.solve_game import execute
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny

# Create your views here.
from rest_framework.response import Response

from gte_backend.settings import BASE_DIR

import sequential_solver.solver.gametree
import sequential_solver.solver.solver as seq_solver

@api_view(["POST"])
@authentication_classes(())
@permission_classes((AllowAny, ))
def solve_game(request):

    game_text = request.POST.get('game_text')

    file_name = os.path.join(BASE_DIR, "sequential_solver/solver/example_input/game_to_solve.ef")
    file = open(file_name, "w+")
    file.write(str(game_text))
    file.close()
    g = seq_solver.import_game(file_name)
    result = seq_solver.solve_from_file(file_name,
          output_file="",
          create_wls=False,
          long_output=False,
          include_header=False,
          include_time=False,

          include_nash=True,
          include_sequential=True,
          restrict_belief=False,
          restrict_strategy=False,
          weak_filter=False,
          extreme_directions="dd")
    '''
    # subprocess for solver.py
    # log = subprocess.check_output(["python3",  os.path.join(BASE_DIR, "sequential_solver/solver/solver.py"),  file_name + ".ef", "-o", file_name + ".gte"])
    solver_path = "/home/mg_linux/Master/GTE_SE/sequential-equilibria/solver/solver.py"
    log = subprocess.check_output(["python3", solver_path ,  file_name + ".ef", "-o", file_name + ".gte"])
    result_file = open(file_name + ".gte", "r")
    result = result_file.read()
    '''
    return Response({"solver_output": result}, status=status.HTTP_201_CREATED)