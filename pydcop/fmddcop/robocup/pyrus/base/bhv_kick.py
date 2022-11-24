from pydcop.algorithms.fmddcop import ModelFreeDynamicDCOP
from pydcop.fmddcop.robocup.pyrus.pyruslib.player.templates import *
from pydcop.fmddcop.robocup.pyrus.pyruslib.action.smart_kick import SmartKick
from typing import List
from pydcop.fmddcop.robocup.pyrus.base.generator_action import KickAction, ShootAction
from pydcop.fmddcop.robocup.pyrus.base.generator_dribble import BhvDribbleGen
from pydcop.fmddcop.robocup.pyrus.base.generator_pass import BhvPassGen
from pydcop.fmddcop.robocup.pyrus.base.generator_shoot import BhvShhotGen


class BhvKick:
    def __init__(self):
        pass

    def execute(self, agent, computation: ModelFreeDynamicDCOP):
        wm: WorldModel = agent.world()
        shoot_candidate: ShootAction = BhvShhotGen().generator(wm)
        if shoot_candidate:
            agent.debug_client().set_target(shoot_candidate.target_point)
            agent.debug_client().add_message(
                'shoot' + 'to ' + shoot_candidate.target_point.__str__() + ' ' + str(shoot_candidate.first_ball_speed))
            SmartKick(shoot_candidate.target_point, shoot_candidate.first_ball_speed,
                      shoot_candidate.first_ball_speed - 1, 3).execute(agent)
            return True
        else:
            action_candidates: List[KickAction] = []
            action_candidates += BhvPassGen().generator(wm)
            action_candidates += BhvDribbleGen().generator(wm)
            if len(action_candidates) == 0:
                return True

            observation = wm.get_observation(wm.self_unum())
            val_index = computation.resolve_decision_variable(
                observation,
                [c.get_feature_vector() for c in action_candidates],
                [c.eval for c in action_candidates]
            )

            # selected_action: KickAction = max(action_candidates)
            selected_action = action_candidates[val_index]

            target = selected_action.target_ball_pos
            # print(selected_action)
            agent.debug_client().set_target(target)
            agent.debug_client().add_message(
                selected_action.type.value + 'to ' + selected_action.target_ball_pos.__str__()
                + ' ' + str(selected_action.start_ball_speed)
            )
            SmartKick(target, selected_action.start_ball_speed, selected_action.start_ball_speed - 1, 3).execute(agent)
            return True
