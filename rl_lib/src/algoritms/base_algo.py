import tensorflow as tf
import abc

from ..data_saver.utils import load_default_config
from .utils import update_config
from rl_lib.src.data_saver.saver import Saver

class Base_Algo(Saver, abc.ABC):
  """Базовый абстрактный класс алгоритма.
  Хранит все методы, необходимые для вычислений в каком либо алгоритме.
  """
  def __init__(self, action_model: object, target_model: object, config: dict, default_config_path: str, *args, **kwargs):
    self._config = load_default_config(default_config_path)
    update_config(self._config, config)
    self.action_model = action_model(self._config, algo_name = kwargs.get("algo_name", "unkown"), name = kwargs.get("name", "unkown_name") + "_action_" + config.get("model_config", {}).get("name", ""))
    self.target_model = target_model(self._config, algo_name = kwargs.get("algo_name", "unkown"), name = kwargs.get("name", "unkown_name") + "_target_" + config.get("model_config", {}).get("name", ""))
    
    super().__init__(**self.config.get('data_saver', {}), **kwargs)
    self.target_model.set_weights(self.action_model.get_weights())

  @property
  def config(self):
      return self._config

  @abc.abstractclassmethod
  def calculate_new_best_action(self) -> tf.Tensor:
    """Вычислеят новое лучшее действие для получения таргета"""

  @abc.abstractclassmethod
  def calculate_target(self) -> dict:
    """Вычисляет таргет для обучения"""

  @abc.abstractclassmethod
  def get_action(self, observation) -> float:
    """Возвращает действие на основе наблюдения с учетом исследования"""

  @abc.abstractclassmethod
  def get_test_action(self, observation) -> float:
    """Возвращает действие на основе наблюдения без исследования"""
     
  @abc.abstractclassmethod
  def get_gradients(self) -> tf.Tensor:
    """Вычисляет градиенты и возвращает их"""

  @abc.abstractclassmethod
  def load(self, path) -> None:
    """Загружает алгоритм"""
  
  @abc.abstractclassmethod
  def reset(self) -> None:
    """Сбрасывает внутренние данные модели"""  
    
  @abc.abstractclassmethod
  def _train_step(self) -> dict:
    """Вспомогательная train_step"""
    
  @abc.abstractclassmethod
  def train_step(self) -> dict:
    """Вычисляет полный обучающий шаг"""

  @abc.abstractclassmethod
  def save(self, path) -> None:
    """Сохраняет алгоритм"""
    
  @abc.abstractclassmethod
  def summary(self) -> None:
    """Выводит архитектуру модели"""
    
  @tf.function(reduce_retracing=None, jit_compile=None, experimental_autograph_options=None)
  def copy_weights(self,) -> tf.constant:
      """Копирует веса из модели действия в целевую модель"""
      for a_w, t_w in zip(self.action_model.weights, self.target_model.weights):
          new_weights = tf.add(tf.multiply(self.tau, a_w), tf.multiply((1-self.tau), t_w))
          t_w.assign(tf.identity(new_weights))
      return tf.constant(1)
    
  @tf.function(reduce_retracing=True,
                jit_compile=True,
                experimental_autograph_options = tf.autograph.experimental.Feature.ALL)
  def sample_action(self, state: tf.Tensor | tuple) -> tf.Tensor | list:
      """Возвращает предсказания модели на основе текущих наблюдений"""
      predict = self.action_model(state)
      if isinstance(predict, list): 
        return self.squeeze_predict(predict[0]), *predict[1:]
      return self.squeeze_predict(predict)

  @tf.function(reduce_retracing=None, jit_compile=None, experimental_autograph_options=None)
  def set_weights(self, target_weights: list) -> tf.constant:
      """Устанавливает переданные как аргумент веса в основную сеть"""
      for a_w, t_w in zip(self.action_model.weights, target_weights):
        a_w.assign(tf.identity(t_w))
      return tf.constant(1)
  
  @staticmethod
  def squeeze_predict(predict) -> tf.Tensor:
    """Удаляет единичные измерения из предсказаний"""
    while predict.shape[0] == 1:
          predict = tf.squeeze(predict)
    return predict
    
