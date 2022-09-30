from pydcop.algorithms.fmddcop import ModelFreeDynamicDCOP
from pydcop.fmddcop.robocup.soccerpy.agent import Agent


class Player(Agent):

    def __init__(self, computation: ModelFreeDynamicDCOP):
        super(Player, self).__init__()
        self._computation: ModelFreeDynamicDCOP = computation

    def decision_loop(self):
        # set observation

        # plan and retrieve action

        # execute action
        ...


class Goalie(Player):
    ...


class Defender(Player):
    ...


class Attacker(Player):
    ...


PLAYER_MAPPING = {
    1: Goalie,
    2: Defender,
    3: Defender,
    4: Defender,
    5: Defender,
    6: Defender,
    7: Attacker,
    8: Attacker,
    9: Attacker,
    10: Attacker,
    11: Attacker,
}
