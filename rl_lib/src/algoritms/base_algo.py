import tensorflow as tf
import abc

from rl_lib.rl_lib.src.optimizers.optimizers import get_optimizer

class Base_Algo(abc.ABC):
  """Базовый абстрактный класс алгоритма.
  Хранит все методы, необходимые для вычислений в каком либо алгоритме.
  """
  def __init__(self, action_model: object, target_model: object, **config):
    self.action_model = action_model
    self.target_model = target_model
    self._config = self.action_model.config
    super().__init__()

  def _initial_model(self):
    if len(self._config["input_shape"]) == 1:
        return self.create_model(self._config["input_shape"], self._config["action_space"])
    else:
      return self.create_model_with_conv(self._config["input_shape"], self._config["action_space"])

  def initial_model(self):
    """Инициализирует модель в соответствии с типом алгоритма"""
    model = self._initial_model()
    optimizer = self.config.get("optimizer")
    optimizer_name = optimizer.get("optimizer_name", "adam")
    optimizer_params = optimizer.get("optimizer_params", {})
    cutom_optimizer = optimizer.get("cutom_optimizer", None)
    optimizer = get_optimizer(optimizer_name, optimizer_params, cutom_optimizer)
    self.action_model.set_new_model(model, optimizer)
    self.target_model.set_new_model(model, optimizer)
    self.target_model.set_weights(self.action_model.get_weights())

  @property
  def config(self):
      return self._config
    
  @staticmethod
  @abc.abstractclassmethod
  def create_model(input_shape: tuple, action_space: int) -> tf.keras.Model:
     """Создает модель по умолчанию и возвращает tf.keras.Model, архитектура в соответствии с алгоритмом, начальные слои - полносвязные"""
    
  @staticmethod
  @abc.abstractclassmethod
  def create_model_with_conv(input_shape: tuple, action_space: int) -> tf.keras.Model:
    """Создает модель по умолчанию  и возвращает tf.keras.Model, архитектура в соответствии с алгоритмом, начальные слои - сверточные"""

  @abc.abstractclassmethod
  def calculate_new_best_action(self) -> tf.Tensor:
    """Вычислеят новое лучшее действие для получения таргета"""

  @abc.abstractclassmethod
  def calculate_target(self) -> dict:
    """Вычисляет таргет для обучения"""
  
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
      for a_w, t_w in zip(self.action_model.weights, self.target_model.weights):
          new_weights = tf.add(tf.multiply(self.tau, a_w), tf.multiply((1-self.tau), t_w))
          t_w.assign(tf.identity(new_weights))
      return tf.constant(1)
    
  @tf.function(reduce_retracing=True,
                jit_compile=True,
                experimental_autograph_options = tf.autograph.experimental.Feature.ALL)
  def sample_action(self, state) -> tf.Tensor or list:
      predict = self.action_model(state)
      if isistance(predict) == list: 
        return self.squeeze_predict(predict[0]), *predict[1:]
      return self.squeeze_predict(predict)

  @tf.function(reduce_retracing=None, jit_compile=None, experimental_autograph_options=None)
  def set_weights(self, target_weights) -> tf.constant:
      for a_w, t_w in zip(self.action_model.weights, target_weights):
        a_w.assign(tf.identity(t_w))
      return tf.constant(1)
  
  @staticmethod
  def squeeze_predict(predict) -> tf.Tensor:
    while predict.shape[0] == 1:
          predict = tf.squeeze(predict)
    return predict
    
