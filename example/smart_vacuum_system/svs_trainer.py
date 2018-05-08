import os
#os.environ['KERAS_BACKEND'] = "theano"
from rl.agents import DQNAgent
from example.smart_vacuum_system.svs_environment import SVSCtx, State, avoid_obstacle_reward, CFMProcessor, point_in_rect
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
import argparse
from rl.callbacks import FileLogger


def load_agent(environment):
    nb_actions = environment.action_space.n
    state_dim = State.dimensionality()
    window_length = 1

    model = Sequential()
    model.add(Flatten(input_shape=(window_length,) + (state_dim,)))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=200000, window_length=window_length)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=50000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=5000, policy=policy, processor=CFMProcessor())
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    return dqn


if __name__ == '__main__':

    # Get the environment and extract the number of actions.
    env = SVSCtx(source_state=State.sample(), objective=avoid_obstacle_reward, reset_mode=2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    args = parser.parse_args()

    weights_filename_1 = 'dqn_SVS_weights_500k.h5f'
    weights_filename_2 = 'dqn_SVS_weights_1000k.h5f'
    dqn = load_agent(environment=env)

    if args.mode == 'train':
        log_filename = 'dqn_SVS_log.json'
        callbacks = [FileLogger(log_filename, interval=100)]
        dqn.fit(env, nb_steps=500000, visualize=False, nb_max_episode_steps=20)
        dqn.save_weights(weights_filename_1, overwrite=True)
        dqn.fit(env, nb_steps=500000, visualize=False, nb_max_episode_steps=20)
        dqn.save_weights(weights_filename_2, overwrite=True)

    elif args.mode == 'test':
        dqn.load_weights(weights_filename_2)
        dqn.test(env, nb_episodes=10, nb_max_episode_steps=50, visualize=True)
