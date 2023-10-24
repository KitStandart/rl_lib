import os.path as os_path
import time
import traceback
from pprint import pprint
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"'
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from rl_lib.src.algoritms.model_free.value_based import DRQN
from rl_lib.src.data_saver.utils import load_default_config

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


def run(algo):
    epidodes = 250
    steps = 200
    train_frequency = 1
    test_frequency = 10
    test_steps = 200
    pre_train_steps = 500   
    copy_weigths_frequency = 100

    # history data
    rewards = []
    episode_reward = 0
    episode_test_reward = 0
    episode_loss = []
    count = 0

    for episode in range(1, epidodes):
        start_time = time.time()

        observation, info = env.reset()
        algo.initial_state()
        episode_reward = 0
        episode_loss = []
        for step in range(1, steps):
            action = algo.get_action(observation)
            new_observation, reward, done, _, info = env.step(action)
            algo.add((observation, action, reward, done, new_observation))
            episode_reward += reward
            count += 1
            if count % train_frequency == 0 and count > pre_train_steps:
                td_error = algo.train_step()
                episode_loss.append(td_error)
                if count % copy_weigths_frequency == 0:
                    res = algo.copy_weights()
            observation = new_observation
            if done:
                break

        algo.save()
        rewards.append(episode_reward)
        # testing algoritm perfomans
        if episode % test_frequency == 0:
            observation, info = env.reset()
            algo.initial_state()
            episode_test_reward = 0
            for test_step in range(1, test_steps):
                action = algo.get_test_action(observation)
                observation, test_reward, done, _, info = env.step(action)
                episode_test_reward += test_reward
                if done:
                    break

        # print info
        print("   Episode %d - Reward = %.3f, episode reward = %.3f, test reward %.3f, Loss = %.6f, Time = %.f sec, Total steps = %.f" %
              (
                  episode,
                  np.asarray(rewards[-10:]).mean() if len(rewards) != 0 else 0,
                  episode_reward,
                  episode_test_reward,
                  np.asarray(episode_loss).mean() if len(
                      episode_loss) != 0 else 0,
                  time.time()-start_time,
                  count
              )
              )


if __name__ == "__main__":
    try:
        run(algo=algo)

    except Exception:
        print(traceback.format_exc())
        input("Press enter to exit: ")
