import numpy as np 
from tensorflow.math import argmax
from tensorflow.dtypes import int32

from ..data_saver.utils import save_data, load_data
from .base_explore import Base_Explore

class Epsilon_Greedy(Base_Explore):
    """Эпсилон-жадная стратегия исследования

    Kwargs:
        eps_decay_steps: int, Количество внутренних шагов исследований до установки минимального эпсилон
        eps_max: float, Максимальный эпсилон
        eps_min: float, Минимальный эпсилон
        eps_test: float, Тестовый эпсилон
        action_spase: int, Размер пространтства действий
        axis: int, Ось вычислений
    """
    def __init__(self, eps_decay_steps=1e6, eps_max=1.0, eps_min=1e-1, eps_test=1e-3, action_space=None, axis=-1):
        self.eps_desay_steps = eps_decay_steps
        self.eps_min = eps_max
        self.eps_max = eps_min
        self.eps_test = eps_test
        assert type(action_space) == int, "Пространство действий должно быть int"
        self.action_space = action_space 
        self.axis = axis
        self._name = "epsilon_greedy_strategy"
        self.reset()

    def __call__(self, Q):
        self.eps = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * self.count/self.eps_desay_steps)
        self.count += 1
        return self.get_action(self.eps, Q)

    def get_action(self, eps, action):
        if np.random.random() < eps: return  np.random.randint(self.action_space)
        else: return argmax(Q, axis=self.axis, output_type=int32)

    def load(self, path):
        self.__dict__ = load_data(path+self.name)
        
    @property
    def name(self):
        return self._name
        
    def reset(self, ):
        self.count = 0
        self.eps = self.eps_max
    
    def save(self, path):
        save_data(path+self.name, self.__dict__)

    def test(self, Q):
        return self.get_action(self.eps_test, Q)

    
