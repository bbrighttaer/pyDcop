from pydcop.fmddcop.robocup.pyrus.base.strategy_formation import StrategyFormation
from pydcop.fmddcop.robocup.pyrus.base.set_play.bhv_set_play import Bhv_SetPlay
from pydcop.fmddcop.robocup.pyrus.pyruslib.player.templates import *
from pydcop.fmddcop.robocup.pyrus.base.bhv_kick import BhvKick
from pydcop.fmddcop.robocup.pyrus.base.bhv_move import BhvMove


def get_decision(agent, computation):
    wm: WorldModel = agent.world()
    st = StrategyFormation().i()
    st.update(wm)

    # handle set-piece
    if wm.game_mode().type() != GameModeType.PlayOn:
        return Bhv_SetPlay().execute(agent)

    # handle hooting and passing
    if wm.self().is_kickable():
        return BhvKick().execute(agent)

    # handle player movement
    return BhvMove().execute(agent)
