from example.smart_vacuum_system.svs_test_env import SVSTestEnv
from tfh.agents.ap_rl_agents import AdfpTester, QLearningTester
from learning_agents.adfp import DecreasingEpsilonGreedyPolicy, Policy, DefaultMemory, Goal
from keras.layers import Dense, Flatten, Input, concatenate
from keras.models import Model
from example.smart_vacuum_system.svs_environment import point_in_rect
from learning_agents.remote.remote_adfp import Processor
from keras.optimizers import Adam
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


class RandomPolicy(Policy):
    def select_action(self, expected_action_qualities):
        return np.random.random_integers(0, len(expected_action_qualities) - 1)


TEMPORAL_OFFSETS = [1, 2, 5]

# Goals #

GOAL_PARAM_SHAPE = (len(TEMPORAL_OFFSETS), 5,)


def collision_goal_param_func(test_env):
    current_state = test_env.env.state
    general_params = [current_state.obstacle_left_bottom[0], current_state.obstacle_left_bottom[1],
                      current_state.obstacle_width, current_state.obstacle_height]
    factors = [[1], [1], [1]]
    for factor in factors:
        factor.extend(general_params)

    return factors


def lead_to_obstacle(measurement, goal_params):
    """
    :param measurement: [ robot position x , robot position y ]
    :param goal_params: position (left lower point) as well as width and height of the obstacle.
    :return:
    """
    robot_position = np.array([measurement[0], measurement[1]])
    obstacle_center = np.array([goal_params[1]+(goal_params[3]/2), goal_params[2]+(goal_params[4]/2)])
    dist_to_center = np.linalg.norm(robot_position - obstacle_center)
    max_dist = 3

    return max(0, max_dist - dist_to_center)


def collision(measurement, goal_params):
    if point_in_rect((goal_params[1], goal_params[2]), goal_params[3], goal_params[4],
                     (measurement[0], measurement[1])):
        return 10
    else:
        return 0


goal = Goal(nb_temporal_offsets=len(TEMPORAL_OFFSETS), immediate_reward_function=lead_to_obstacle,
            future_measurement_processor=lambda pres_obs, fut_meas: fut_meas)

collision_goal = Goal(nb_temporal_offsets=len(TEMPORAL_OFFSETS), immediate_reward_function=collision,
                      future_measurement_processor=lambda pres_obs, fut_meas: fut_meas)


# Define tester's model

class SVSProcessor(Processor):

    def process_observation(self, observation):
        return observation.serialize()

    def process_measurement(self, measurement):
        return list(measurement)

    def process_reward(self, reward):
        return reward

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        """
        return batch


def load_model(env_config):
    inputs_observation = Input(shape=env_config.observation_shape)
    inputs_action = Input(shape=env_config.test_action_shape)
    inputs_goal = Input(shape=GOAL_PARAM_SHAPE)
    flatten_goal = Flatten()(inputs_goal)
    # hidden layers
    merged = concatenate([inputs_observation, inputs_action, flatten_goal])
    hidden_1 = Dense(256, activation='relu')(merged)
    hidden_2 = Dense(128, activation='relu')(hidden_1)
    hidden_3 = Dense(64, activation='relu')(hidden_2)
    # output layer
    output = Dense(len(TEMPORAL_OFFSETS) * env_config.system_actions, activation='linear')(hidden_3)
    model = Model(inputs=[inputs_observation, inputs_action, inputs_goal], outputs=output)

    return model


def plot(steps, values_1, values_2, values_3):

    plt.plot(steps, values_1)
    plt.plot(steps, values_2)
    plt.plot(steps, values_3)

    plt.show()


def process_failures(steps, failures):
    avgs = []
    stddev = []
    maxima = []
    minima = []

    transf_failures = []
    for failure_list in failures:
        transf = []
        for step in steps:
            nb_failures = 0
            for failure in failure_list:
                if failure.step <= step:
                    nb_failures += 1
            transf.append(nb_failures)
        transf_failures.append(transf)

    for idx, step in enumerate(steps):
        values = [f[idx] for f in transf_failures]
        avgs.append(np.mean(values))
        stddev.append(np.std(values))
        maxima.append(max(values))
        minima.append(min(values))

    return {'avgs': avgs, 'stddev': stddev, 'maxima': maxima, 'minima' : minima}


def test_q_learner(goal):
    failures = []
    for i in range(epochs_per_tester):
        inputs_observation = Input(shape=(1, 8))
        flatten_inputs_observations = Flatten()(inputs_observation)
        inputs_action = Input(shape=test_env.env_config.test_action_shape)
        # hidden layers
        merged = concatenate([flatten_inputs_observations, inputs_action])
        hidden_1 = Dense(256, activation='relu')(merged)
        hidden_2 = Dense(128, activation='relu')(hidden_1)
        hidden_3 = Dense(64, activation='relu')(hidden_2)
        # output layer
        output = Dense(1, activation='linear')(hidden_3)
        model = Model(inputs=[inputs_observation, inputs_action], outputs=output)

        q_policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.05, value_test=.05,
                                        nb_steps=5000)

        q_learning_tester = QLearningTester(env_config=test_env.env_config, policy=q_policy,
                                            model=model, memory=SequentialMemory(limit=50000, window_length=1),
                                            temporal_offsets=[1, 2, 5], log_interval=100, processor=SVSProcessor(),
                                            goal=goal, optimizer=optimizer,
                                            folder_path=q_learning_folder, metrics=metrics)

        failures.append(q_learning_tester.fit(goal_param_func=collision_goal_param_func, nb_steps=steps_per_epoch,
                                              test_env=test_env, nb_max_episode_steps=10))
    return failures


def test_dfp():
    failures = []
    for i in range(epochs_per_tester):
        model = load_model(test_env.env_config)
        standard_tester = AdfpTester(env_config=test_env.env_config,
                                     policy=DecreasingEpsilonGreedyPolicy(steps=5000, start_eps=1, end_eps=0.05),
                                     model=model, memory=DefaultMemory(max_length=50000), temporal_offsets=[1, 2, 5],
                                     log_interval=100, processor=SVSProcessor(), goal=collision_goal,
                                     optimizer=optimizer, folder_path=dfp_folder, metrics=metrics)

        failures.append(standard_tester.fit(goal_param_func=collision_goal_param_func, nb_steps=steps_per_epoch,
                                            test_env=test_env, nb_max_episode_steps=10))
    return failures


def test_random():
    failures = []
    for i in range(epochs_per_tester):
        model = load_model(test_env.env_config)
        random_tester = AdfpTester(env_config=test_env.env_config, policy=RandomPolicy(), model=model,
                                   memory=DefaultMemory(max_length=50000), temporal_offsets=[1, 2, 5],
                                   log_interval=100, processor=SVSProcessor(), goal=goal, optimizer=optimizer,
                                   folder_path=random_folder, metrics=metrics)

        failures.append(
            random_tester.fit(goal_param_func=collision_goal_param_func, nb_steps=steps_per_epoch, test_env=test_env,
                              nb_max_episode_steps=10))
    return failures


def test_the_testers(step):
    # QLearning Tester
    q_learner_failures = test_q_learner(goal=collision_goal)
    logs_q_learner = process_failures(steps=steps, failures=q_learner_failures)
    with open('q_learning_tester_step_' + str(step) + '.json', 'w') as outfile:
        json.dump(logs_q_learner, outfile)

    # Standard Tester
    standard_failures = test_dfp()
    logs_dfp = process_failures(steps=steps, failures=standard_failures)
    with open('dfp_tester_step_'+ str(step) + '.json', 'w') as outfile:
        json.dump(logs_dfp, outfile)

    # Random Tester
    random_failures = test_random()
    logs_random_tester = process_failures(steps=steps, failures=random_failures)
    with open('random_tester_step_' + str(step) + '.json', 'w') as outfile:
        json.dump(logs_random_tester, outfile)


if __name__ == '__main__':

    # goals: collision_goal and goal

    steps_per_epoch = 1000
    epochs_per_tester = 10

    steps = np.linspace(0, steps_per_epoch, 100)

    optimizer = Adam(lr=1e-3)
    metrics = ['mae']

    q_learning_folder = 'SVS_q_learning_tester_weights'
    dfp_folder = 'SVS_standard_tester_weights'
    random_folder = 'SVS_random_tester_weights'

    # Load test environment for 500k steps
    weights_filename = '../dqn_SVS_weights_500k.h5f'
    test_env = SVSTestEnv(weights_filename=weights_filename)

    # test with the 3 testers
    test_the_testers(1)

    # Load test environment for 1000k steps
    weights_filename = '../dqn_SVS_weights_1000k.h5f'
    test_env = SVSTestEnv(weights_filename=weights_filename)

    # test with the 3 testers again
    test_the_testers(2)


    # AdfpTester(env_config=test_env.env_config,
            #                         policy=DecreasingEpsilonGreedyPolicy(steps=5000, start_eps=1, end_eps=0.05),
            #                         model=model, memory=DefaultMemory(max_length=50000), temporal_offsets=[1, 2, 5],
            #                         log_interval=100, processor=SVSProcessor(), goal=goal, optimizer=optimizer,
            #                         folder_path=tester_weights_folder, metrics=metrics)