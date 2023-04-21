from django.conf.urls import url

from sequential_solver.views import solve_game
from sequential_solver.views import read_game
from sequential_solver.views import respond_status

urlpatterns = [
    url(r'^solve', solve_game, name="solve_game"),
    url(r'^read', read_game, name="read_game"),
    url(r'^status', respond_status, name="respond_status")
]