import gym
import numpy as np
import time
import os.path as os_path
from tensorflow.keras import layers
import tensorflow as tf
from pprint import pprint
import traceback

from rl_lib.src.algoritms.ddpg.ddpg import DDPG
from rl_lib.src.data_saver.utils import load_default_config

env = gym.make('CarRacing-v2')

def create_conv():
    input_layer = layers.Input(shape=env.observation_space.shape, )
    cov_layer1 = layers.Conv2D(16, 7, activation='relu')(input_layer)
    cov_layer2 = layers.Conv2D(32, 5, activation='relu')(cov_layer1)
    conv_out = layers.Flatten()(cov_layer2)
    return tf.keras.Model(inputs=input_layer, outputs=conv_out)  

def create_model():
    """Создает модель tf.keras.Model, архитектура DQN"""
    input_layer = layers.Input(shape=env.observation_space.shape, )
    # conv_out = create_conv()(input_layer)
    dence_layer1 = layers.Dense(256, activation='relu')(input_layer)
    dence_layer2 = layers.Dense(256, activation='relu')(dence_layer1)
    dence_out = layers.Dense(env.action_space.shape[0], activation='tanh')(dence_layer2)

    dence_out = dence_out*2
    
    return tf.keras.Model(inputs=input_layer, outputs=dence_out)

def create_critic_model():
    """Создает модель tf.keras.Model, архитектура DQN, начальные слои - сверточные"""
    input_layer = layers.Input(shape=env.observation_space.shape, )
    obsv_layer = layers.Dense(16, activation='relu')(input_layer)
    obsv_layer = layers.Dense(32, activation='relu')(obsv_layer)
    input_action_layer = layers.Input(shape=env.action_space.shape, )
    action_layer = layers.Dense(32, activation='relu')(input_action_layer)
    
    # conv_out = create_conv()(input_layer)
    concat = layers.Concatenate()((obsv_layer, action_layer))
    flatten = layers.Flatten()(concat)
    dence_layer1 = layers.Dense(256, activation='relu')(flatten)
    dence_layer2 = layers.Dense(256, activation='relu')(dence_layer1)
    dence_out = layers.Dense(env.action_space.shape[0], activation=None)(dence_layer2)
    
    return tf.keras.Model(inputs=[input_layer, input_action_layer], outputs=dence_out)   

config = load_default_config(__file__)

config['actor_model_config']['model_config']['model'] = create_model()
config['critic_model_config']['model_config']['model'] = create_critic_model()
config['model_config']['input_shape'] = env.observation_space.shape
config['model_config']['action_space'] = env.action_space.shape[0]

config['exploration_config']['strategy_config']['upper_bound'] = env.action_space.high
config['exploration_config']['strategy_config']['lower_bound'] = env.action_space.low

algo = DDPG(config)

pprint(algo.config)

def run(algo):
    epidodes = 250
    steps = 100
    train_frequency = 1
    test_frequency = 10
    test_steps = 100
    pre_train_steps = 1000
    copy_weigths_frequency = 1

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
        for step in range(1, steps+1):
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
            for test_step in range(1, test_steps+1):
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
    
    except Exception:
        print(traceback.format_exc())
        input("Press enter to exit: ")

