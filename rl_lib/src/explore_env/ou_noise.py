import numpy as np 

from ..data_saver.utils import save_data, load_data
from .base_explore import Base_Explore

class OU_Noise(Base_Explore):
    """Шум Орнштейна — Уленбека стратегия исследования, применяется к предсказанным непрерывным действиям.

    Kwargs:        
        action_spase: int, Размер пространтства действий
        alpha: int, Количество внутренних шагов исследований до установки минимального эпсилон        
        axis: int, ось вычислений
        sigma: float, Максимальный эпсилон
    """
    def __init__(self, action_space = None, axis=-1, alpha = 0.9, sigma=1.0, **kwargs):        
        self.action_space = action_space 
        self.alpha = alpha
        self.axis = axis
        self.eps = np.random.normal(size=self.action_space, scale = sigma)
        self.sigma = sigma        
        self._name = "ou_noise"                
        
    def __call__(self, action):
        action += self.eps
        self.eps = self.alpha*self.eps + np.random.normal(size=self.action_space, scale = self.sigma)
        return action

    def load(self, path):
        self.__dict__ = load_data(path+self.name)
        
    @property
    def name(self):
        return self._name
        
    def reset(self, ):
        self.eps = np.random.normal(size=self.action_space, scale = self.sigma)
    
    def save(self, path):
        save_data(path+self.name, self.__dict__)

    def test(self, action):
        return action