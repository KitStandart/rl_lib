import gym
import numpy as np


class ImageNormWrapper(gym.Wrapper):
    """Обертка нормализации наблюдений среды,
    если это изображения

    Args:
        gym (_type_): _description_
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=40, options={}):
        observation, info = self.env.reset(seed=40, options={})
        return self.preprocess(observation), info

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return self.preprocess(observation), reward, done, tr, info

    def preprocess(self, observation):
        observation = (observation - 255/2)/(255/2)
        return observation.astype(np.float16)
