import logging

from pydcop.algorithms.fmddcop import ModelFreeDynamicDCOP
from pydcop.fmddcop.robocup.soccerpy.agent import Agent
from pydcop.fmddcop.robocup.soccerpy.world_model import WorldModel

INFINITY = 100.0

class Player(Agent):

    def __init__(self, computation: ModelFreeDynamicDCOP):
        super(Player, self).__init__()
        self._computation: ModelFreeDynamicDCOP = computation

        self.logger = logging.getLogger(f'player-{self._computation.name}')

        # set constraint callbacks of the computation/algorithm
        self._computation.coordination_constraint_cb = self.coordination_constraint
        self._computation.unary_constraint_cb = self.unary_constraint

    def think(self):
        """
        Performs a single step of thinking for our agent.  Gets called on every
        iteration of our think loop.
        """

        # DEBUG:  tells us if a thread dies
        if not self._think_thread.is_alive() or not self._msg_thread.is_alive():
            raise Exception("A thread died.")

        # take places on the field by uniform number
        if not self.in_kick_off_formation:
            print("the side is", self.wm.side)

            # used to flip x coords for other side
            side_mod = 1
            if self.wm.side == WorldModel.SIDE_R:
                side_mod = -1

            self.set_433_formation(side_mod)

            self.in_kick_off_formation = True

            return

        # determine the enemy goal position
        if self.wm.side == WorldModel.SIDE_R:
            self.enemy_goal_pos = (-55, 0)
            self.own_goal_pos = (55, 0)
        else:
            self.enemy_goal_pos = (55, 0)
            self.own_goal_pos = (-55, 0)

        if not self.wm.is_before_kick_off() or self.wm.is_kick_off_us() or self.wm.is_playon():
            # The main decision loop
            return self.decision_loop()

    def set_433_formation(self, side_mod):
        # goalie
        if self.wm.uniform_number == 1:
            self.wm.teleport_to_point((-50 * side_mod, 0))

        # defenders
        elif self.wm.uniform_number == 2:
            self.wm.teleport_to_point((-40 * side_mod, 25))
        elif self.wm.uniform_number == 3:
            self.wm.teleport_to_point((-40 * side_mod, -25))
        elif self.wm.uniform_number == 4:
            self.wm.teleport_to_point((-40 * side_mod, 12))
        elif self.wm.uniform_number == 5:
            self.wm.teleport_to_point((-40 * side_mod, -12))

        # midfielder
        elif self.wm.uniform_number == 6:
            self.wm.teleport_to_point((-30 * side_mod, 0))
        # midfielders
        elif self.wm.uniform_number == 10:
            self.wm.teleport_to_point((-20 * side_mod, -15))
        elif self.wm.uniform_number == 8:
            self.wm.teleport_to_point((-20 * side_mod, 15))

        # forwards
        elif self.wm.uniform_number == 7:
            self.wm.teleport_to_point((-10 * side_mod, -25))
        elif self.wm.uniform_number == 9:
            self.wm.teleport_to_point((-5 * side_mod, 0))
        elif self.wm.uniform_number == 11:
            self.wm.teleport_to_point((-10 * side_mod, 25))

    def decision_loop(self):
        self.logger.debug('In decision loop')

        # set observation
        observation = self.get_current_observation()
        self._computation.set_observation(observation)

        # plan and retrieve action
        self.default_action()  # TODO: temporary action

        # execute action

    def get_current_observation(self):
        nearest_teammate = self.wm.get_nearest_teammate()
        nearest_opponent = self.wm.get_nearest_enemy()

        return {
            'dist_to_own_goal_post': self.wm.get_distance_to_point(self.own_goal_pos),

            'angle_of_own_goal_post': self.wm.get_angle_to_point(self.own_goal_pos),

            'dist_to_ball': self.wm.get_distance_to_point(
                self.wm.get_object_absolute_coords(self.wm.ball)
            ) if self.wm.ball else 100,

            'angle_of_ball': self.wm.get_angle_to_point(
                self.wm.get_object_absolute_coords(self.wm.ball)
            ) if self.wm.ball else 180,

            'dist_to_nearest_mate': self.wm.get_distance_to_point(
                self.wm.get_object_absolute_coords(nearest_teammate)
            ) if nearest_teammate else 100,

            'angle_to_nearest_mate': self.wm.get_angle_to_point(
                self.wm.get_object_absolute_coords(nearest_teammate)
            ) if nearest_teammate else 180,

            'dist_to_nearest_opp': self.wm.get_distance_to_point(
                self.wm.get_object_absolute_coords(nearest_opponent)
            ) if nearest_opponent else 100,

            'angle_of_nearest_opp': self.wm.get_angle_to_point(
                self.wm.get_object_absolute_coords(nearest_opponent)
            ) if nearest_opponent else 180,

            'is_path_clear_for_ball': self.is_clear(
                self.wm.get_object_absolute_coords(self.wm.ball)
            ) if self.wm.ball else False,

            'is_ball_owned': self.wm.is_ball_owned_by_us() if self.wm.ball else False,

            'is_ball_opp_owned': self.wm.is_ball_owned_by_enemy() if self.wm.ball else False,

            'is_ball_kickable': self.wm.is_ball_kickable(),

            'dist_to_opp_goal_post': self.wm.get_distance_to_point(self.enemy_goal_pos),

            'angle_to_opp_goal_post': self.wm.get_angle_to_point(self.enemy_goal_pos),

            'stamina': self.wm.get_stamina(),

            'speed': self.wm.speed_amount,

            'speed_direction': self.wm.speed_direction,
        }

    def coordination_constraint(self, *args, **kwargs):
        ...

    def unary_constraint(self, *args, **kwargs):
        ...

    # check if ball is close to self
    def ball_close(self):
        return self.wm.ball.distance < 10

    # check if enemy goalpost is close enough
    def goal_post_close(self):
        return self.wm.get_distance_to_point(self.enemy_goal_pos) < 20

    # check if path to target's coordinate is clear, by direction
    def is_clear(self, target_coords):
        q = self.wm.get_nearest_enemy()
        if q == None:
            return False
        q_coords = self.wm.get_object_absolute_coords(q)
        qDir = self.wm.get_angle_to_point(q_coords)
        qDist = self.wm.get_distance_to_point(q_coords)

        tDir = self.wm.get_angle_to_point(target_coords)
        tDist = self.wm.get_distance_to_point(target_coords)

        # the closest teammate is closer, or angle is clear
        return tDist < qDist or abs(qDir - tDir) > 20

    # Action decisions start
    # find the ball by rotating if ball not found
    def find_ball(self):
        # find the ball
        if self.wm.ball is None or self.wm.ball.direction is None:
            self.wm.ah.turn(30)
            if not -7 <= self.wm.ball.direction <= 7:
                self.wm.ah.turn(self.wm.ball.direction / 2)

            return

    # condition for shooting to the goal
    def shall_shoot(self):
        return self.wm.is_ball_kickable() and self.goalpos_close() and self.is_clear(self.enemy_goal_pos)

    # do shoot
    def shoot(self):
        print("shoot")
        return self.wm.kick_to(self.enemy_goal_pos, 1.0)

    # condition for passing to the closest teammate
    # if can kick ball, teammate is closer to goal, path clear
    def shall_pass(self):
        # self.defaultaction()
        p = self.wm.get_nearest_teammate()
        if p == None:
            return False
        p_coords = self.wm.get_object_absolute_coords(p)
        pDistToGoal = self.wm.euclidean_distance(p_coords, self.enemy_goal_pos)
        myDistToGoal = self.wm.get_distance_to_point(self.enemy_goal_pos)
        # kickable, pass closer to goal, path is clear
        return self.wm.is_ball_kickable() and pDistToGoal < myDistToGoal and self.is_clear(p_coords)

    # do passes
    def passes(self):
        print("pass")
        p = self.wm.get_nearest_teammate()
        if p == None:
            return False
        p_coords = self.wm.get_object_absolute_coords(p)
        dist = self.wm.get_distance_to_point(p_coords)
        power_ratio = 2 * dist / 55.0
        # kick to closest teammate, power is scaled
        return self.wm.kick_to(p_coords, power_ratio)

    # condition for dribbling, if can't shoot or pass
    def shall_dribble(self):
        # find the ball
        # self.find_ball()
        # if self.wm.ball is None or self.wm.ball.direction is None:
        # self.wm.ah.turn(30)
        return self.wm.is_ball_kickable()

    # dribble: turn body, kick, then run towards ball
    def dribble(self):
        print("dribbling")
        self.wm.kick_to(self.enemy_goal_pos, 1.0)
        self.wm.turn_body_to_point(self.enemy_goal_pos)
        self.wm.align_neck_with_body()
        if self.wm.get_distance_to_point(self.own_goal_pos) < 40:
            self.wm.ah.dash(50)
        else:
            self.wm.turn_body_to_point(self.own_goal_pos)
            self.wm.ah.dash(50)
        return

    # if enemy has the ball, and not too far move towards it
    def shall_move_to_ball(self):
        # while self.wm.ball is None:
        # self.find_ball()
        # self.wm.align_neck_with_body()
        return self.wm.is_ball_owned_by_enemy() and self.wm.ball.distance < 30

    # move to ball, if enemy owns it
    def move_to_ball(self):
        print("move_to_ball")
        if self.wm.get_distance_to_point(self.own_goal_pos) < 40:
            self.wm.ah.dash(60)
        else:
            self.wm.turn_body_to_point(self.own_goal_pos)
            self.wm.ah.dash(50)
        return

        # defensive, when ball isn't ours, and has entered our side of the field

    def shall_move_to_defend(self):
        # self.defaultaction()
        if self.wm.ball is not None or self.wm.ball.direction is not None:
            b_coords = self.wm.get_object_absolute_coords(self.wm.ball)
            return self.wm.is_ball_owned_by_enemy() and self.wm.euclidean_distance(self.own_goal_pos,
                                                                                   b_coords) < 55.0
        return False

    # defend
    def move_to_defend(self):
        print("move_to_defend")
        q = self.wm.get_nearest_enemy()
        if q == None:
            return False
        q_coords = self.wm.get_object_absolute_coords(q)
        qDir = self.wm.get_angle_to_point(q_coords)
        qDistToOurGoal = self.wm.euclidean_distance(self.own_goal_pos, q_coords)
        # if close to the goal, aim at it
        if qDistToOurGoal < 55:
            self.wm.turn_body_to_point(q_coords)
        # otherwise aim at own goalpos, run there to defend
        else:
            self.wm.turn_body_to_point(self.own_goal_pos)

        self.wm.align_neck_with_body()
        if self.wm.get_distance_to_point(self.own_goal_pos) < 40:
            self.wm.ah.dash(80)
        else:
            self.wm.turn_body_to_point(self.own_goal_pos)
            self.wm.ah.dash(50)
        return

    # when our team has ball, and self is not close enough to goalpos. advance to enemy goalpos
    def shall_move_to_enemy_goalpos(self):
        return self.wm.is_ball_owned_by_us() and not self.goalpos_close()

    # if our team has the ball n u r striker
    def move_to_enemy_goalpos(self):
        print("move_to_enemy_goalpos")
        if self.wm.is_ball_kickable():
            # kick with 100% extra effort at enemy goal
            self.wm.kick_to(self.enemy_goal_pos, 1.0)
        self.wm.turn_body_to_point(self.enemy_goal_pos)
        self.wm.align_neck_with_body()
        if self.wm.get_distance_to_point(self.own_goal_pos) < 40:
            self.wm.ah.dash(70)
        else:
            self.wm.turn_body_to_point(self.own_goal_pos)
            self.wm.ah.dash(50)
        return

    def goalpos_close(self):
        return self.wm.get_distance_to_point(self.enemy_goal_pos) < 20


class Goalie(Player):

    def unary_constraint(self, *args, **kwargs):
        ...

    # if enemy has the ball, and not too far move towards it
    def shall_move_to_ball(self):
        # while self.wm.ball is None:
        # self.find_ball()
        # self.wm.align_neck_with_body()
        return self.wm.is_ball_owned_by_enemy() and self.wm.ball.distance < 10 and self.wm.get_distance_to_point(
            self.own_goal_pos) < 10


class Defender(Player):

    def unary_constraint(self, *args, **kwargs):
        ...


class Attacker(Player):

    def unary_constraint(self, *args, **kwargs):
        ...

    # dribble: turn body, kick, then run towards ball
    def dribble(self):
        print("dribbling")
        self.wm.kick_to(self.enemy_goal_pos, 1.0)
        self.wm.turn_body_to_point(self.enemy_goal_pos)
        self.wm.align_neck_with_body()
        self.wm.ah.dash(50)
        return

    # move to ball, if enemy owns it
    def move_to_ball(self):
        print("move_to_ball")
        self.wm.ah.dash(60)
        return

    # defend
    def move_to_defend(self):
        print("move_to_defend")
        q = self.wm.get_nearest_enemy()
        if q == None:
            return False
        q_coords = self.wm.get_object_absolute_coords(q)
        qDir = self.wm.get_angle_to_point(q_coords)
        qDistToOurGoal = self.wm.euclidean_distance(self.own_goal_pos, q_coords)
        # if close to the goal, aim at it
        if qDistToOurGoal < 55:
            self.wm.turn_body_to_point(q_coords)
        # otherwise aim at own goalpos, run there to defend
        else:
            self.wm.turn_body_to_point(self.own_goal_pos)

        self.wm.align_neck_with_body()
        self.wm.ah.dash(80)
        return

    # if our team has the ball n u r striker
    def move_to_enemy_goalpos(self):
        print("move_to_enemy_goalpos")
        if self.wm.is_ball_kickable():
            # kick with 100% extra effort at enemy goal
            self.wm.kick_to(self.enemy_goal_pos, 1.0)
        self.wm.turn_body_to_point(self.enemy_goal_pos)
        self.wm.align_neck_with_body()
        self.wm.ah.dash(70)
        return


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
