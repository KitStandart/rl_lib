from tensorflow.keras.activations import softmax
from tensorflow.math import argmax
from tensorflow.dtypes import int32

from .base_explore import Base_Explore
from ..data_saver.utils import save_data, load_data

class Soft_Q(Base_Explore):
  """Больцмановская стратегия исследования 
  a = softmax(Q/tau)

  Kwargs:
    tau: float, Больцмановская температура
    axis: int, Ось вычислений
  """
  def __init__(self, tau=1.0, axis=-1):
    self.tau = 1.0
    self.axis = axis
  
  def reset(self, ) -> None:
    """Выполняет внутренний сброс"""
    pass
  
  def save(self, path) -> None:
    """Сохраняет какие либо внутренние переменные"""
    save_data(path ,{
                'count': self.count,
                    })
 
  def load(self, path) -> None:
    """Загружает какие либо внутренние переменные"""
    data = load_data(path)
    self.count = data['count']
  
  def __call__(self, Q) -> int:
    """Возвращает действие в соответствии с стратегией исследования"""
    return softmax(Q/self.tau, axis=self.axis)
  
  def test(self, Q) -> int:
    """Возвращает действие в соответствии с стратегией тестирования"""
    return argmax(Q, axis=self.axis, output_type=int32)
