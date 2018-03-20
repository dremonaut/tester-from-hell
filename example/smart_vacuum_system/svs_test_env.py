from tfh.environment import TestEnvironment, EnvConfig
from example.smart_vacuum_system.svs_trainer import load_agent
from example.smart_vacuum_system.svs_environment import point_in_rect, State
from ap.action_provider import ActionProvider, Action
from rl.core import Env
from example.smart_vacuum_system.svs_environment import SVSCtx, clean_dust_reward
from rl.callbacks import Callback


class OpenAiGymEnvProxy(Env):

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def configure(self, *args, **kwargs):
        pass

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


class SVSStateObserver(Callback):

    def __init__(self):
        self.history = []

    def on_step_begin(self, step, logs={}):
        self.history.append(self.env.state)


class SVSActionProvider(ActionProvider):

    def actions(self, state):
        semantic_state = State.deserialize(state, 10, 10)
        free_positions = semantic_state.free_positions()
        actions = []
        for position in free_positions:
            actions.append(Action([position[0]/semantic_state.room_width, position[1]/semantic_state.room_height]))
        return actions


class SVSTestEnv(TestEnvironment):

    def __init__(self, weights_filename):
        self.env = SVSCtx(source_state=State.sample(), objective=clean_dust_reward, reset_mode=2)
        self.svs = load_agent(self.env)
        self.svs.load_weights(weights_filename)
        self.env_config = EnvConfig(test_action_shape=(2,), observation_shape=(10,), system_actions=2,
                                    test_action_provider=SVSActionProvider())
        self.count_broken_invs = 0
        self.state_observer = SVSStateObserver()

    def step(self, test_action, visualize=False):
        """
        :param visualize:
        :param test_action:
        :return:
        """

        self.env.reset_mode = -1

        # Apply test action
        manipulated_state = self.env.state.copy()
        manipulated_state.dirt_positions = [(test_action.params[0]*self.env.state.room_width,
                                             test_action.params[1]*self.env.state.room_height)]
        self.env.state = manipulated_state

        # Apply system actions
        self.svs.test(env=self.env, nb_episodes=1, nb_max_episode_steps=5, verbose=0, visualize=visualize,
                      callbacks=[self.state_observer])

        resulting_state = self.env.state

        measurements = resulting_state.robot_position
        done = len(resulting_state.dirt_positions) == 0 or self.svs_collided(resulting_state)

        return measurements, resulting_state, done

    def distance(self, observation_one, observation_two):
        raise NotImplementedError

    def reset(self):
        self.env.reset_mode = 2
        state = self.env.reset()
        measurements = state.robot_position
        history = self.state_observer.history[:]
        self.state_observer.history.clear()
        return history, measurements, state, False

    @staticmethod
    def svs_collided(state):
        return point_in_rect(state.obstacle_left_bottom, state.obstacle_width, state.obstacle_height,
                             state.robot_position)
