from rl_lib.rl_lib.src.model.model import Model
from .base_algo import Base_Algo

import abc

class Algoritm(Model, Base_Algo):
  """Абстрактный класс алгоритма, 
  реализует основную логику вычислений
  """
  
  def __init__(self, **config):
    super().__init__(**config)
    self.action_model = Model(config = config, name = "DQN_action" + config.get("name", ""))
    self.target_model = Model(config = config, name = "DQN_target" + config.get("name", ""))

  def _initial_model(self):
    if len(self._config["input_shape"]) <= 1:
        return self.create_model(self._config["input_shape"], self._config["action_space"])
    else:
      return self.create_model_with_conv(self._config["input_shape"], self._config["action_space"])

  def initial_model(self):
      model = self._initial_model()
      self.action_model.set_new_model(model, )
      self.target_model.set_new_model(model, )
      self.target_model.set_weights(self.action_model.get_weights())

  @staticmethod
  @abstractmethod
  def create_model(input_shape: tuple, action_space: int) -> tf.keras.Model:
     """Создает модель и возвращает tf.keras.Model, архитектура в соответствии с алгоритмом, начальные слои - полносвязные"""
    
  @staticmethod
  @abstractmethod
  def create_model_with_conv(input_shape: tuple, action_space: int) -> tf.keras.Model:
    """Создает модель и возвращает tf.keras.Model, архитектура в соответствии с алгоритмом, начальные слои - сверточные"""
   

  @abstractmethod
  def train_step(self):
    """Вычисляет полный обучающий шаг"""

  @abstractmethod
  def calculate_new_best_action(self):
    """Вычислеят новое лучшее действие для получения таргета"""

  @abstractmethod
  def calculate_target(self):
    """Вычисляет таргет для обучения"""
