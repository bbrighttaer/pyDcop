from pydcop.fmddcop.robocup.pyrus.pyruslib.debug.level import Level
from pydcop.fmddcop.robocup.pyrus.pyruslib.debug.logger import dlog
from pydcop.fmddcop.robocup.pyrus.pyruslib.math.angle_deg import AngleDeg
from pydcop.fmddcop.robocup.pyrus.base.strategy_formation import StrategyFormation

class Bhv_BeforeKickOff:
    def __init__(self):
        pass

    def execute(self, agent):
        unum = agent.world().self().unum()
        st = StrategyFormation.i()
        target = st.get_pos(unum)
        agent.do_move(target.x(), target.y())
        return True
