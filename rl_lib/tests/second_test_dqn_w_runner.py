from pprint import pprint

import gym
import tensorflow as tf
from tensorflow.keras import layers

from rl_lib import DQN
from rl_lib import load_default_config
from rl_lib import Base_Env_Runner

env = gym.make('CartPole-v0')


def create_model():
    """Создает модель tf.keras.Model, архитектура DQN"""
    input_layer = layers.Input(shape=env.observation_space.shape, )
    dence_layer1 = layers.Dense(32, activation='relu')(input_layer)
    dence_layer2 = layers.Dense(32, activation='relu')(dence_layer1)
    dence_out = layers.Dense(env.action_space.n, activation=None)(dence_layer2)

    return tf.keras.Model(inputs=input_layer, outputs=dence_out)


config = load_default_config("./rl_lib/tests/dqn_config.yaml")
pprint(config)
config['model_config']['model'] = create_model()
config['model_config']['input_shape'] = env.observation_space.shape
config['model_config']['action_space'] = env.action_space.n
algo = DQN(config)

pprint(algo.config)

runner = Base_Env_Runner(env=env,
                         algo=algo,
                         episodes=250,
                         env_steps=200,
                         env_test_steps=200,
                         pre_train_steps=2000,
                         test_counts=4,
                         train_frequency=4,
                         test_frequency=10,
                         copy_weigths_frequency=100,
                         new_step_api=True)

runner.run()
