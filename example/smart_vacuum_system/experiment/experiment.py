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
import os
import pickle


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


def collision_goal_param_func_v2(test_env):
    current_state = test_env.env.state
    general_params = [0.0, 1.0, 1, 1]
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


def load_dfp_model(env_config, policy):
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
    #print(model.summary())

    tester = AdfpTester(env_config=test_env.env_config,
                        policy=policy,
                        model=model, memory=DefaultMemory(max_length=50000), temporal_offsets=[1, 2, 5],
                        log_interval=steps_per_epoch, processor=SVSProcessor(), goal=collision_goal,
                        optimizer=optimizer, folder_path=dfp_folder, metrics=metrics, failure=failure_1)

    return tester


def load_q_model(env_config, policy):
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
    #print(model.summary())

    q_learning_tester = QLearningTester(env_config=env_config, policy=policy,
                                        model=model, memory=SequentialMemory(limit=50000, window_length=1),
                                        temporal_offsets=[1, 2, 5], log_interval=steps_per_epoch, processor=SVSProcessor(),
                                        goal=collision_goal, optimizer=optimizer,
                                        folder_path=q_learning_folder, metrics=metrics, failure=failure_1)

    return q_learning_tester


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


def evaluate(testers, env, goal_param_func, nb_steps):
    failures = []
    for tester in testers:
        failures.append(tester.fit(goal_param_func=goal_param_func, nb_steps=nb_steps,
                                       test_env=env, nb_max_episode_steps=10))
    logs = process_failures(steps=steps, failures=failures)
    return logs, failures


def failure_1(observation):
    if point_in_rect(observation.obstacle_left_bottom, observation.obstacle_width,
                  observation.obstacle_height, observation.robot_position):
        return True
    return False


def failure_2(observation):
    if point_in_rect((0, 1), 1, 1, observation.robot_position):
        return True
    return False


if __name__ == '__main__':

    # goals: collision_goal and goal

    steps_per_epoch = 10000
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
    policy_steps = 5000

    q_learning_testers = [load_q_model(env_config=test_env.env_config, policy=LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.05, value_test=.05,
                                    nb_steps=policy_steps)) for i in range(epochs_per_tester)]
    dfp_testers = [load_dfp_model(env_config=test_env.env_config, policy=DecreasingEpsilonGreedyPolicy(steps=policy_steps, start_eps=1, end_eps=0.05)) for i in range(epochs_per_tester)]
    random_testers = [load_dfp_model(env_config=test_env.env_config, policy=RandomPolicy()) for i in range(epochs_per_tester)]

    if True:
        if False:
            print("===== Starting with Q-Learning Tester =====")
            logs_q_learner, _ = evaluate(q_learning_testers, test_env, collision_goal_param_func, steps_per_epoch)
            with open('logs/q_learning_tester_step_1.json', 'w') as outfile:
                json.dump(logs_q_learner, outfile)

        print("===== Starting with DFP Tester =====")
        logs_dfp, failures = evaluate(dfp_testers, test_env, collision_goal_param_func, steps_per_epoch)
        with open('logs/dfp_tester_step_1.json', 'w') as outfile:
            json.dump(logs_dfp, outfile)
        # Save reached failures for analysis
        with open('logs/dfp_failure_data.pkl', 'wb') as output:
            pickle.dump(failures, output, pickle.HIGHEST_PROTOCOL)

        if False:
            print("===== Starting with Random Tester =====")
            logs_random_tester, _ = evaluate(random_testers, test_env, collision_goal_param_func, steps_per_epoch)
            with open('logs/random_tester_step_1.json', 'w') as outfile:
                json.dump(logs_random_tester, outfile)

        # Save weights of dfp_testers
        #for i in range(len(dfp_testers)):
        #    path = os.path.abspath(dfp_folder + '/agent_' + str(i))
        #    print(path)
        #    dfp_testers[i].adfp_agent.save(path)

    if False:
        print("Starting to evaluate the goal generalization property of DFP and DQN")
        for tester in dfp_testers:
            tester.failure = failure_2
        logs_dfp, _ = evaluate(dfp_testers, test_env, collision_goal_param_func_v2, steps_per_epoch)
        with open('logs/dfp_tester_step_generalization.json', 'w') as outfile:
            json.dump(logs_dfp, outfile)
        for tester in q_learning_testers:
            tester.failure = failure_2
        logs_q_learner, _ = evaluate(q_learning_testers, test_env, collision_goal_param_func_v2, steps_per_epoch)
        with open('logs/q_learning_tester_step_generalization.json', 'w') as outfile:
            json.dump(logs_q_learner, outfile)

    if True:
        print("Starting to evaluate the interstep-generalization property of DFP")
        # Load test environment for 1000k steps
        weights_filename = '../dqn_SVS_weights_1000k.h5f'
        prev_config = test_env.env_config
        test_env = SVSTestEnv(weights_filename=weights_filename)
        test_env.env_config = prev_config  # TODO do this by appropriate equals method in EnvConfig

        free_dfp_testers = [load_dfp_model(env_config=test_env.env_config,
                            policy=DecreasingEpsilonGreedyPolicy(steps=policy_steps, start_eps=1, end_eps=0.05))
                            for i in range(epochs_per_tester)]

        print("===== Starting with pre-learned DFP Tester =====")
        # Load weights of dfp_testers
        for i in range(len(dfp_testers)):
            tester = dfp_testers[i]
            tester.adfp_agent.policy = DecreasingEpsilonGreedyPolicy(steps=policy_steps, start_eps=0.5, end_eps=0.05)
            tester.failure = failure_2
        logs_dfp, _ = evaluate(dfp_testers, test_env, collision_goal_param_func_v2, steps_per_epoch)
        with open('logs/dfp_tester_step_2.json', 'w') as outfile:
            json.dump(logs_dfp, outfile)

        print("===== Starting with free DFP Tester =====")
        for tester in free_dfp_testers:
            tester.failure = failure_2
        logs_dfp, failures = evaluate(free_dfp_testers, test_env, collision_goal_param_func_v2, steps_per_epoch)
        with open('logs/dfp_tester_step_2_free.json', 'w') as outfile:
            json.dump(logs_dfp, outfile)
        with open('logs/dfp_failure_data_test.pkl', 'wb') as output:
            pickle.dump(failures, output, pickle.HIGHEST_PROTOCOL)




    # AdfpTester(env_config=test_env.env_config,
            #                         policy=DecreasingEpsilonGreedyPolicy(steps=5000, start_eps=1, end_eps=0.05),
            #                         model=model, memory=DefaultMemory(max_length=50000), temporal_offsets=[1, 2, 5],
            #                         log_interval=100, processor=SVSProcessor(), goal=goal, optimizer=optimizer,
            #                         folder_path=tester_weights_folder, metrics=metrics)