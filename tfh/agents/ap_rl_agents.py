from tfh.test_agent import TesterFromHell
from ap.action_provider import ActionProvider
from learning_agents.adfp import ADFPAgent
from learning_agents.adqn import ADQNAgent
from learning_agents.remote.remote_adfp import RemoteAdfp
from learning_agents.remote.remote_dql import RemoteAdqn
from tfh.environment import TestEnvironment
from example.smart_vacuum_system.svs_test_env import SVSTestEnv
from example.smart_vacuum_system.svs_environment import point_in_rect


class Failure(object):

    def __init__(self, type, step, episode, record):
        self.type = type
        self.step = step
        self.episode = episode
        self.record = record


class AdfpTester(TesterFromHell):

    def __init__(self, env_config, policy, model, memory, temporal_offsets, log_interval, goal, processor, optimizer,
                 folder_path, metrics):
        self.folder_path = folder_path
        self.log_interval = log_interval
        self.env_config = env_config
        self.adfp_agent = ADFPAgent(policy=policy, model=model, action_provider=env_config.test_action_provider,
                                    memory=memory, goal=goal,
                                    temporal_offsets=temporal_offsets, target_model_update=100)
        self.adfp_agent.compile(optimizer=optimizer, metrics=metrics)
        self.processor = processor

    def fit(self, test_env: TestEnvironment, nb_steps, goal_param_func, nb_max_episode_steps=None):

        assert test_env.env_config == self.env_config

        failures = []

        tester = RemoteAdfp(agent=self.adfp_agent, training_steps=nb_steps, log_interval=self.log_interval,
                            processor=self.processor, folder_path=self.folder_path)

        _, measurements, observation, done = test_env.reset()

        for i in range(nb_steps):

            test_action = tester.train_move(raw_observation=observation, measurement=measurements,
                                            goal_params=goal_param_func(test_env), done=done)
            if done:
                history, measurements, observation, done = test_env.reset()
                for failure in failures:
                    if failure.episode == tester.episode - 1:
                        failure.record = history
                i -= 1
                continue

            measurements, observation, done = test_env.step(test_action)

            # Analyse resulting state
            if point_in_rect(observation.obstacle_left_bottom, observation.obstacle_width,
                             observation.obstacle_height, observation.robot_position):
                # TODO Generalize failure definitions
                failures.append(Failure(episode=tester.episode, step=tester.step, record=None,
                                        type='Crash'))

            if nb_max_episode_steps and tester.episode_step >= nb_max_episode_steps - 1:
                done = True

        return failures

    def test(self, test_env: TestEnvironment, goal_param_func, nb_episodes):

        assert test_env.env_config == self.env_config

        tester = RemoteAdfp(agent=self.adfp_agent, training_steps=0, log_interval=self.log_interval,
                            processor=self.processor, folder_path=self.folder_path)

        _, measurements, observation, done = test_env.reset()

        episode_nb = 0
        while episode_nb < nb_episodes:
            test_action = tester.test_move(raw_observation=observation, measurement=measurements,
                                           goal_params=goal_param_func(test_env))
            measurements, observation, done = test_env.step(test_action, visualize=True)
            if done:
                _, measurements, observation, done = test_env.reset()
                episode_nb += 1


class QLearningTester(TesterFromHell):

    def __init__(self, env_config, policy, model, memory, temporal_offsets, log_interval, goal, processor, optimizer,
                 folder_path, metrics):
        self.folder_path = folder_path
        self.log_interval = log_interval
        self.env_config = env_config
        self.goal = goal
        self.temporal_offsets = temporal_offsets
        self.agent = ADQNAgent(policy=policy, model=model, action_provider=env_config.test_action_provider,
                               memory=memory, nb_steps_warmup=3, target_model_update=100, processor=processor)

        self.agent.compile(optimizer=optimizer, metrics=metrics)
        self.processor = processor

    def fit(self, test_env: TestEnvironment, nb_steps, goal_param_func, nb_max_episode_steps=None):

        assert test_env.env_config == self.env_config

        failures = []

        tester = RemoteAdqn(agent=self.agent, training_steps=nb_steps, log_interval=self.log_interval,
                            folder_path=self.folder_path)

        _, measurements, observation, done = test_env.reset()

        for i in range(nb_steps):
            goal_params = goal_param_func(test_env)
            eval_goal_params = goal_params if not isinstance(goal_params[0], list) else goal_params[-1]
            reward = self.goal.immediate_reward_function(measurement=measurements, goal_params=eval_goal_params)
            test_action = tester.train_move(observation=observation, reward=reward, done=done)

            if done:
                history, measurements, observation, done = test_env.reset()
                for failure in failures:
                    if failure.episode == tester.episode - 1:
                        failure.record = history
                i -= 1
                continue

            measurements, observation, done = test_env.step(test_action)

            # Analyse resulting state
            if point_in_rect(observation.obstacle_left_bottom, observation.obstacle_width,
                             observation.obstacle_height, observation.robot_position):
                # TODO Generalize failure definitions
                failures.append(Failure(episode=tester.episode, step=tester.agent.step, record=None,
                                        type='Crash'))

            if nb_max_episode_steps and tester.episode_step >= nb_max_episode_steps - 1:
                done = True

        return failures

    def test(self, test_env: TestEnvironment, goal_param_func, nb_episodes):

        assert test_env.env_config == self.env_config

        tester = RemoteAdqn(agent=self.agent, training_steps=0, log_interval=self.log_interval,
                            folder_path=self.folder_path)

        _, measurements, observation, done = test_env.reset()

        episode_nb = 0
        while episode_nb < nb_episodes:
            test_action = tester.test_move(observation=observation)
            measurements, observation, done = test_env.step(test_action, visualize=True)
            if done:
                _, measurements, observation, done = test_env.reset()
                episode_nb += 1
