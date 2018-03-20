from __future__ import division
from rl.core import Env
import random
from rl.core import Processor
import numpy as np
import pickle
from example.smart_vacuum_system.svs_gui import SVSVisualizer


class CFMProcessor(Processor):
    def process_observation(self, observation):
        p_obs = np.array(observation.serialize())
        return p_obs


def clean_dust_reward(state):
    if not state.dirt_positions:
        return 1
    return 0


def avoid_obstacle_reward(state):
    if point_in_rect(state.obstacle_left_bottom, state.obstacle_width, state.obstacle_height, state.robot_position):
        return -1
    return clean_dust_reward(state)


def point_in_rect(rect_left_bottom, rect_width, rect_height, point):
    rect_right = rect_left_bottom[0] + rect_width
    rect_top = rect_left_bottom[1] + rect_height
    return rect_left_bottom[0] <= point[0] < rect_right and \
        rect_left_bottom[1] <= point[1] < rect_top


class ActionSpace(object):
    def __init__(self, action_count):
        self.n = action_count


class SVSCtx(Env):
    """
        TODO Describe reset-modes and other parameters.
    """
    # Limit actions on driving Left, Right, Ahead, Backward
    action_space = ActionSpace(4)

    def __init__(self, source_state, objective, reset_mode=0):
        self.state = source_state.copy()
        self.source_state = source_state.copy()
        self.objective = objective
        self.visualizer = None
        self.reset_mode = reset_mode

    def step(self, action):
        robot_pos = self.state.robot_position
        room_width = self.state.room_width
        room_height = self.state.room_height

        # Action Right
        if action == 0 and robot_pos[0] < room_width - 1:
            self.state.robot_position = (robot_pos[0] + 1, robot_pos[1])
        # Action Left
        elif action == 1 and robot_pos[0] > 0:
            self.state.robot_position = (robot_pos[0] - 1, robot_pos[1])
        # Action Ahead
        elif action == 2 and robot_pos[1] < room_height - 1:
            self.state.robot_position = (robot_pos[0], robot_pos[1] + 1)
        # Action Backward
        elif action == 3 and robot_pos[1] > 0:
            self.state.robot_position = (robot_pos[0], robot_pos[1] - 1)

        # Assert that robot is staying in room.
        assert (self.state.robot_position[0] < room_width and self.state.robot_position[1] < room_height)

        # Clear vacuumed dust
        dirt_positions = [dirt for dirt in self.state.dirt_positions if dirt != self.state.robot_position]
        self.state.dirt_positions = dirt_positions

        r = self.objective(self.state)

        within_obstacle = point_in_rect(self.state.obstacle_left_bottom, self.state.obstacle_width, self.state.obstacle_height,
                                        self.state.robot_position)

        done = r == 1 or within_obstacle

        return self.state, r, done, {}

    def reset(self):
        # TODO reuse State.random_state method!
        if self.reset_mode < 0:
            return self.state

        if self.reset_mode == 3:
            self.state = self.source_state.copy()
            return self.state

        self.state = State.sample()
        if self.reset_mode == 2:
            left_bottom_x = random.randint(0, self.state.room_width - 2)
            left_bottom_y = random.randint(0, self.state.room_height - 2)
            self.state.obstacle_left_bottom = (left_bottom_x, left_bottom_y)
            self.state.obstacle_width = random.randint(1, (self.state.room_width - 1 - left_bottom_x))
            self.state.obstacle_height = random.randint(1, (self.state.room_height - 1 - left_bottom_y))

        free_positions = self.state.free_positions()
        self.state.dirt_positions = [random.choice(free_positions)]
        free_positions.remove(self.state.dirt_positions[0])

        if self.reset_mode >= 1:
            self.state.robot_position = random.choice(
                [p for p in free_positions if p not in self.state.dirt_positions])

        return self.state

    def render(self, mode='human', close=False):
        if self.visualizer is None:
            print("new instance made")
            self.visualizer = SVSVisualizer()
            self.visualizer.start()
        self.visualizer.put_state(self.state.copy())
        if close and self.visualizer is not None:
            self.visualizer.close()
            self.join()

    def close(self):
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer.join()

    def seed(self, seed=None):
        raise NotImplementedError

    def configure(self, *args, **kwargs):
        raise NotImplementedError

    def save_to_file(self, file_name):
        with open(file_name, 'wb') as output:
            pickle.dump(self, output, -1)

    @staticmethod
    def concrete_states(ctx_state):
        free_positions = ctx_state.free_positions()
        concrete_states = []
        for free_position in free_positions:
            concrete_states.append(State(ctx_state.room_height, ctx_state.room_width,
                                         ctx_state.obstacle_left_bottom, ctx_state.obstacle_width,
                                         ctx_state.obstacle_height, ctx_state.robot_position,
                                         free_position))
        return concrete_states

    @staticmethod
    def load_from_file(file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)


class State(object):
    # TODO Categorize features: fixed (to be initialized with a static value), variable (to be initialized randomly)
    def __init__(self, room_height, room_width, obstacle_left_bottom, obstacle_width, obstacle_height, robot_position,
                 dirt_position=None, room_segmentation=None):
        """

        :param room_height:
        :param room_width:
        :param obstacle_left_bottom:
        :param obstacle_width:
        :param obstacle_height:
        :param robot_position:
        :param dirt_position:
        :param room_segmentation: RoomSegmentation object.
        """

        self.room_height = room_height
        self.room_width = room_width
        self.obstacle_left_bottom = obstacle_left_bottom
        self.obstacle_width = obstacle_width
        self.obstacle_height = obstacle_height
        self.dirt_positions = [dirt_position] if dirt_position else []
        self.robot_position = robot_position
        self.room_segmentation = room_segmentation

    def serialize(self):
        """
        Serialize state:(Ctx, TestInput) as input for neural network.
        :return:
        """
        assert (len(self.dirt_positions) <= 1)  # at least for the time being.
        # init with 'default' value of 0.25
        dirt_position_x = self.dirt_positions[0][0] if self.dirt_positions else -1
        dirt_position_y = self.dirt_positions[0][1] if self.dirt_positions else -1
        return [self.robot_position[0] / self.room_width, self.robot_position[1] / self.room_height,
                dirt_position_x / self.room_width, dirt_position_y / self.room_height,
                self.obstacle_left_bottom[0] / self.room_width, self.obstacle_left_bottom[1] / self.room_height,
                self.obstacle_width / self.room_width, self.obstacle_height / self.room_height]

    def copy(self):
        return State(room_height=self.room_height, room_width=self.room_width,
                     obstacle_left_bottom=self.obstacle_left_bottom, obstacle_width=self.obstacle_width,
                     obstacle_height=self.obstacle_height,
                     dirt_position=self.dirt_positions[0] if len(self.dirt_positions) == 1 else None,
                     robot_position=self.robot_position, room_segmentation=self.room_segmentation)

    def free_positions(self):
        room_height = self.room_height
        room_width = self.room_width
        free_positions = []
        for x in range(0, room_width):
            for y in range(0, room_height):
                p = (x,y)
                if not point_in_rect(point=p, rect_left_bottom=self.obstacle_left_bottom,
                                     rect_height=self.obstacle_height, rect_width=self.obstacle_width):
                    free_positions.append(p)
        return free_positions

    @staticmethod
    def deserialize(features, room_width, room_height):
        features = features
        assert (len(features) == State.dimensionality())
        return State(room_width=room_width, room_height=room_height,
                     robot_position=(features[0] * room_width, features[1] * room_height),
                     dirt_position=(features[2] * room_width, features[3] * room_height),
                     obstacle_left_bottom=(features[4] * room_width, features[5] * room_height),
                     obstacle_width=features[6] * room_width, obstacle_height=features[7] * room_height)

    @staticmethod
    def sample():
        return State(room_height=10, room_width=10, obstacle_left_bottom=(3, 4), obstacle_width=1,
                     obstacle_height=1, dirt_position=(1, 2), robot_position=(9, 9))

    @staticmethod
    def dimensionality():
        return len(State.sample().serialize())

    @staticmethod
    def random_state(room_height, room_width):
        state = State.sample()
        state.room_width = room_width
        state.room_height = room_height
        state.left_bottom_x = random.randint(0, room_width - 2)
        state.left_bottom_y = random.randint(0, room_height - 2)
        state.obstacle_left_bottom = (state.left_bottom_x, state.left_bottom_y)
        state.obstacle_width = random.randint(1, (room_width - 1 - state.left_bottom_x))
        state.obstacle_height = random.randint(1, (room_height - 1 - state.left_bottom_y))
        free_positions = state.free_positions()
        state.dirt_positions = [random.choice(free_positions)]
        free_positions.remove(state.dirt_positions[0])
        state.robot_position = random.choice(
            [p for p in free_positions if p not in state.dirt_positions])
        return state

    def fuzzy_relative(self):
        state = self.copy()
        obstacle_bottom_candidates = []
        olb = state.obstacle_left_bottom
        for i in range(-2, 2):
            for j in range(-2, 2):
                if olb[0] + i >= 0 and olb[0] + i + state.obstacle_width < state.room_width and \
                                        olb[1] + j >= 0 and olb[1] + j + state.obstacle_height < state.room_height:
                    obstacle_bottom_candidates.append((olb[0], olb[1]))
        state.obstacle_left_bottom = random.choice(obstacle_bottom_candidates)
        free_positions = state.free_positions()
        dirt_position_candidates = []
        dirt_pos = self.dirt_positions[0]
        for i in range(-4, 4):
            for j in range(-4, 4):
                fuzz_pos = (dirt_pos[0] + i, dirt_pos[1] + j)
                if fuzz_pos in free_positions:
                    dirt_position_candidates.append(fuzz_pos)
        state.dirt_positions = [random.choice(dirt_position_candidates)]
        return state

    @staticmethod
    def random_states(room_height, room_width, no_states):
        return [State.random_state(room_height, room_width) for _ in range(no_states)]

    def __key(self):
        return (self.room_height, self.room_width, self.obstacle_left_bottom, self.obstacle_width, self.obstacle_height,
                self.dirt_positions[0], self.robot_position)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(self, other)