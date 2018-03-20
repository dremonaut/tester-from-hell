class EnvConfig(object):
    def __init__(self, observation_shape, test_action_shape, test_action_provider, system_actions):
        self.observation_shape = observation_shape
        self.test_action_shape = test_action_shape
        self.test_action_provider = test_action_provider
        self.system_actions = system_actions

    def __eq__(self, other):
        return self.observation_shape == other.observation_shape and \
               self.test_action_shape == other.test_action_shape and \
               self.test_action_provider == other.test_action_provider and\
               self.system_actions == other.system_actions


class TestEnvironment(object):

    env_config = None

    def step(self, test_action, visualize=False):
        """

        :return: measurement, observation, done
        """
        raise NotImplementedError

    def reset(self):
        """

        :return: measurements, observation, done
        """
        raise NotImplementedError

    def distance(self, observation_one, observation_two):
        """

        :param observation_one:
        :param observation_two:
        :return:
        """
        raise NotImplementedError
