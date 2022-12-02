import threading
import time

from pydcop.fmddcop.robocup.pyrus.player import Player

PLAYERS = []


def start_team():
    for _ in range(11):
        threading.Thread(target=_start_player, daemon=True).start()
        time.sleep(0.01)


def _start_player():
    player = Player()
    PLAYERS.append(player)
    player.start()
