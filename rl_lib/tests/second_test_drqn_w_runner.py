from pprint import pprint
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"'
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from rl_lib import DRQN, load_default_config, Base_Env_Runner

env = gym.make('CartPole-v0')


def create_model(lstm_size=32):
    """Создает модель tf.keras.Model, архитектура DRQN"""

    input_layer = layers.Input(shape=(None, *env.observation_space.shape), )
    h_t_input = layers.Input(shape=(lstm_size, ), )
    c_t_input = layers.Input(shape=(lstm_size, ), )

    lstm = layers.LSTM(lstm_size, activation='tanh', recurrent_activation='sigmoid', return_sequences=True,
                       return_state=True, stateful=False)(input_layer, initial_state=[h_t_input, c_t_input])
    dence_layer1 = layers.Dense(32, activation='relu')(lstm[0])
    dence_out = layers.Dense(env.action_space.n, activation=None)(dence_layer1)

    return tf.keras.Model(inputs=[input_layer, h_t_input, c_t_input], outputs=[dence_out, lstm[1], lstm[2]])


config = load_default_config("./rl_lib/tests/drqn_config.yaml")
config['model_config']['model'] = create_model(
    lstm_size=config['model_config']['lstm_size'])
config['model_config']['input_shape'] = env.observation_space.shape
config['model_config']['action_space'] = env.action_space.n
algo = DRQN(config)

pprint(algo.config)
runner = Base_Env_Runner(env=env,
                         algo=algo,
                         episodes=250,
                         env_steps=200,
                         env_test_steps=200,
                         pre_train_steps=2000,
                         test_counts=1,
                         train_frequency=1,
                         test_frequency=10,
                         copy_weigths_frequency=100,
                         new_step_api=True)

runner.run()
