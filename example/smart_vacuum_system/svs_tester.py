from example.smart_vacuum_system.svs_test_env import SVSTestEnv
from tfh.agents.ap_rl_agents import AdfpTester
from learning_agents.adfp import DecreasingEpsilonGreedyPolicy, Policy, DefaultMemory, Goal
from keras.layers import Dense, Flatten, Input, concatenate
from keras.models import Model
from example.smart_vacuum_system.svs_environment import point_in_rect
from learning_agents.remote.remote_adfp import Processor
from keras.optimizers import Adam
import argparse
import numpy as np
import pickle


TEMPORAL_OFFSETS = [1, 2, 5]

# Define Goal #

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

goal = Goal(nb_temporal_offsets=len(TEMPORAL_OFFSETS), immediate_reward_function=lead_to_obstacle,
            future_measurement_processor=lambda pres_obs, fut_meas: fut_meas)


# Define tester's model

class SVSProcessor(Processor):

    def process_observation(self, observation):
        return observation.serialize()

    def process_measurement(self, measurement):
        return list(measurement)


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


if __name__ == '__main__':

    # Load test environment
    weights_filename = 'dqn_SVS_weights_500k.h5f'
    test_env = SVSTestEnv(weights_filename=weights_filename)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    args = parser.parse_args()

    # Construct tester
    model = load_model(test_env.env_config)

    optimizer = Adam(lr=1e-3)
    metrics = ['mae']

    tester_weights_folder = 'SVS_tester_weights'

    tester = AdfpTester(env_config=test_env.env_config,
                        policy=DecreasingEpsilonGreedyPolicy(steps=5000, start_eps=1, end_eps=0.05), model=model,
                        memory=DefaultMemory(max_length=50000), temporal_offsets=[1, 2, 5], log_interval=10000,
                        processor=SVSProcessor(), goal=goal, optimizer=optimizer,
                        folder_path=tester_weights_folder, metrics=metrics)
    # Apply
    if args.mode == 'train':
        failures = tester.fit(goal_param_func=collision_goal_param_func, nb_steps=300000, test_env=test_env,
                              nb_max_episode_steps=10)

        with open('failure_data.pkl', 'wb') as output:
            pickle.dump(failures, output, pickle.HIGHEST_PROTOCOL)

        tester.adfp_agent.save(tester_weights_folder)
        print("Nb of revealed failures: " + str(len(failures)))

    elif args.mode == 'test':
        tester.adfp_agent.load_weights(tester_weights_folder + '/model.h5f')
        tester.test(goal_param_func=collision_goal_param_func, nb_episodes=10, test_env=test_env)
