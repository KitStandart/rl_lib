import gym
import numpy as np
import time
import os.path as os_path
from tensorflow.keras import layers
import tensorflow as tf
from pprint import pprint

from rl_lib.src.algoritms.dqn.dqn import DQN
from rl_lib.src.data_saver.utils import load_default_config

env = gym.make('CartPole-v0')

def create_model():
    """Создает модель tf.keras.Model, архитектура DQN"""
    input_layer = layers.Input(shape=env.observation_space.shape, )
    dence_layer1 = layers.Dense(32, activation='relu')(input_layer)
    dence_layer2 = layers.Dense(32, activation='relu')(dence_layer1)
    dence_out = layers.Dense(env.action_space.n, activation=None)(dence_layer2)
    
    return tf.keras.Model(inputs=input_layer, outputs=dence_out)

config = load_default_config("..\\rl_lib\\rl_lib\\examples\\dqn\\cart_pole/")
config['model_config']['model'] = create_model()
config['model_config']['input_shape'] = env.observation_space.shape
config['model_config']['action_space'] = env.action_space.n
algo = DQN(config)

pprint(algo.config)

def run(algo):
    epidodes = 250
    steps = 200
    train_frequency = 1
    test_frequency = 10
    test_steps = 200
    pre_train_steps = 2000
    copy_weigths_frequency = 100

    #history data
    rewards = []
    episode_reward = 0
    episode_test_reward = 0
    episode_loss = []
    count = 0

    for episode in range(1, epidodes):
        start_time = time.time()

        observation, info = env.reset()
        episode_reward = 0
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
        #testing algoritm perfomans
        if episode%test_frequency == 0:
            observation, info = env.reset()
            episode_test_reward = 0
            for test_step in range(1, test_steps):
                action = algo.get_test_action(observation)
                observation, test_reward, done, _, info = env.step(action)
                episode_test_reward += test_reward
                if done:
                    break
        

        #print info
        print("   Episode %d - Reward = %.3f, episode reward = %.3f, test reward %.3f, Loss = %.6f, Time = %.f sec, Total steps = %.f" %
                (
                episode,
                np.asarray(rewards[-10:]).mean() if len(rewards) != 0 else 0,
                episode_reward,
                episode_test_reward,
                np.asarray(episode_loss).mean() if len(episode_loss) != 0 else 0,
                time.time()-start_time,
                count
                )
                )

if __name__ == "__main__":
    try:
        run(algo=algo)
    
    except Exception as e:
        print(e)
        input("Press enter to exit: ")

