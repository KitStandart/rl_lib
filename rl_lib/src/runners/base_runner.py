import abc
import time
import traceback

import numpy as np
from gym import Env


class Abc_Base_Env_Runner(abc.ABC):
    def __init__(self, env: Env, algo) -> None:
        self.env = env
        self.algo = algo

    @abc.abstractclassmethod
    def _run():
        "Основная логика обучения"
        pass

    @abc.abstractclassmethod
    def train():
        "Запускает обучения алгоритма в среде"
        pass

    @abc.abstractclassmethod
    def test():
        "Запускает тестирование алгоритма в среде"
        pass

    def save(self):
        "Сохраняет текущие прааметры обучения"
        self.algo.save()

    def load(self):
        "Загружает текущие прааметры обучения"
        self.algo.load()

    def run(self):
        "Запуск процесса обучения нейронной сети в текущей среде"
        try:
            self._run()
        except Exception as e:
            print(traceback.format_exc())
            input("Press enter to exit: ")


def run_episode(func):
    "Обертка выполнения эпизода в среде. Работает только с self."

    def wrapper(self, *args, **kwargs):
        tr = False
        observation, _ = self.env.reset()
        self.algo.initial_state()
        episode_reward = 0
        for env_step in range(1, kwargs.get("steps")):
            env_step_result, other_info = func(self, observation)
            observation, reward, done = env_step_result[:3]
            if self.new_step_api:
                tr = env_step_result[3]
            episode_reward += reward
            if done or tr:
                break
        return episode_reward, other_info
    return wrapper


class Base_Env_Runner(Abc_Base_Env_Runner):
    """Этот класс реализует в себе все методы для обучения нейронной сети.
    Для запуска обучения просто нужно передать
    все параметры обучения алгоритма, среду и алгоритм.

    Args:
        env: gym.Env
        algo: Any, какой либо алгоритм из rl_lib.src.algoritms
    """

    def __init__(self, env: Env, algo,
                 episodes: int = None,
                 env_steps: int = None,
                 env_test_steps: int = None,
                 pre_train_steps: int = None,
                 test_counts: int = 1,
                 train_frequency: int = None,
                 test_frequency: int = None,
                 copy_weigths_frequency: int = 1,
                 new_step_api: bool = False,
                 save_frequency: int = -1,
                 *args, **kwargs) -> None:
        super().__init__(env, algo)
        self.episodes = episodes
        self.env_steps = env_steps
        self.env_test_steps = env_test_steps
        self.pre_train_steps = pre_train_steps
        self.test_counts = test_counts

        self.train_frequency = train_frequency
        self.test_frequency = test_frequency
        self.copy_weigths_frequency = copy_weigths_frequency
        self.save_frequency = save_frequency

        self.new_step_api = new_step_api

        self.counter = 0
        self.episode_num = 0
        self._check_params()

    def _check_params(self):
        "Проверяет все параметры обучения"
        assert isinstance(
            self.episodes, int), "Кол-во эпизодов должно быть int"
        assert isinstance(
            self.env_steps, int), "Кол-во шагов в среде должно быть int"
        assert isinstance(self.env_test_steps,
                          int), "Кол-во тестовых шагов в среде должно быть int"
        assert isinstance(self.pre_train_steps,
                          int), "Кол-во претреин шагов должно быть int"

        assert isinstance(self.train_frequency,
                          int), "Частота обучения должна быть int"
        assert isinstance(self.test_frequency, int) \
            or self.test_frequency is None, "Частота тестов должна быть int"
        assert isinstance(self.copy_weigths_frequency,
                          int), "Частота копирования весов должна быть int"
        assert isinstance(self.save_frequency,
                          int), "Частота сохранения должна быть int"

    def _single_explore_step(self, observation):
        action = self.algo.get_action(observation)
        env_step_result = self.env.step(action)
        self.algo.add(
            (observation, action, *env_step_result[1:3], env_step_result[0]))
        self.counter += 1
        return env_step_result

    def _train_step(self):
        td_error = self.algo.train_step()
        if self.counter % self.copy_weigths_frequency == 0:
            res = self.algo.copy_weights()
            assert res, "Ошибка копирования весов"
        return td_error

    @run_episode
    def _train_episode(self, observation=None):
        td_error = None
        env_step_result = self._single_explore_step(observation=observation)
        if self.counter % self.train_frequency == 0 \
                and self.counter > self.pre_train_steps:
            td_error = self._train_step()
        return env_step_result, td_error

    def _single_test_step(self, observation):
        action = self.algo.get_test_action(observation)
        env_step_result = self.env.step(action)
        return env_step_result

    @run_episode
    def _single_test_episode(self, observation=None) -> float:
        episode_test_reward = self._single_test_step(observation=observation)
        return episode_test_reward, None

    def _print_info(self,
                    episode: int,
                    all_rewards: list,
                    episode_reward: float,
                    avg_test_reward: float,
                    all_td_error: list,
                    start_time: float,
                    counter: int):
        print("   Episode %d - Reward = %.3f, episode reward = %.3f, test reward %.3f, Loss = %.6f, Time = %.f sec, Total steps = %.f" %
              (
                  episode,
                  np.asarray(
                    all_rewards[-10:]).mean() if len(all_rewards) != 0 else 0,
                  episode_reward,
                  avg_test_reward,
                  np.asarray(all_td_error).mean() if len(
                      all_td_error) != 0 else 0,
                  time.time()-start_time,
                  counter
              )
              )

    def _run(self):
        all_td_error = []
        all_rewards = []
        avg_test_reward = 0
        for episode in range(self.episodes):
            start_time = time.time()
            episode_reward, td_error = self.train()
            if episode % self.test_frequency == 0:
                avg_test_reward = self.test()
            if self.save_frequency > 0 and \
                    episode % self.save_frequency == 0:
                self.save()
            all_rewards.append(episode_reward)
            if td_error:
                all_td_error.append(td_error)
            self._print_info(
                episode=episode,
                all_rewards=all_rewards,
                episode_reward=episode_reward,
                avg_test_reward=avg_test_reward,
                all_td_error=all_td_error,
                start_time=start_time,
                counter=self.counter
            )

    def train(self):
        return self._train_episode(self,
                                   observation=None,
                                   steps=self.env_steps)

    def test(self):
        test_reward = []
        for _ in range(self.test_counts):
            test_reward.append(self._single_test_episode(
                self, observation=None, steps=self.env_test_steps)[0])
        return sum(test_reward)/len(test_reward)
