from ..data_saver.data_saver import Saver

import abc
import tensorflow as tf


class BaseModel(abc.ABC):
  """Абстрактный базовый класс,
  представляющий общий интерфейс для всех алгоритмов и моделей в RL-Lib.

  Model определяет общие методы для ввода, вывода и базовых вычислений,
  которые должны быть реализованы в каждом конкретном алгоритме или модели.

  Этот класс служит в качестве основы для всех других классов в RL-Lib
  и обеспечивает единый интерфейс для работы с различными моделями.
  """

  def __init__(self, **kwargs):
    pass

  @property
  @abc.abstractmethod
  def input_spec(self) -> tuple:
    """Возвращает кортеж размера входных данных Модели"""

  @property
  @abc.abstractmethod
  def output_spec(self) -> tuple:
    """Возвращает кортеж размера выходных данных Модели"""

  @abc.abstractmethod
  def initial_state(self) -> None:
    """Инициализирует внутреннее состояние реккурентной Модели"""

  @abc.abstractmethod
  def _update_next_state(self) -> None:
    """Обновляет внутреннее состояние реккурентной Модели"""

class ModelIO(Saver, abc.ABC):
  def __init__(self, config: dict, **kwargs):
    super().__init__(**kwargs)
    self._config = {}
    self._config.update(config)

  @property
  def config(self) -> dict:
    """Возвращает конфигурацию алгоритма"""
    return self._config

  @abc.abstractmethod
  def save(self) -> None:
    """Сохраняет модель в директории"""

  @abc.abstractmethod
  def load(self) -> None:
    """Загружает модель из директории"""

class ModelNN(abc.ABC):
  """Абстрактрный класс, представляющий модель нейронной сети для вычисления градиента,
   обновления весов и извлечения слоев, весов, компиляции модели.

   Kwargs:
    model: tf.keras.Model
    name: str Необязательно, название модели   
  """

  def __init__(self, model_config: dict, **kwargs):
    super().__init__(**kwargs)
    self.model = model_config.get('model', None)
    self.name = model_config.get('name', 'None')
    # self.validate_args()
  
  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    return self.model(inputs)
  
  @abc.abstractmethod
  def _prediction_processing(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Обрабатывает выходы модели перед вычислением лоссов
    Args:
      inputs: tf.Tensor(dtype=tf.float32)
      mask: tf.Tensor(dtype=tf.float32)
    Returns
      outputs: tf.Tensor(dtype=tf.float32
    """
    return inputs

  def prediction_processing(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Обрабатывает выходы модели перед вычислением лоссов
    Args:
      inputs: tf.Tensor(dtype=tf.float32)
      mask: tf.Tensor(dtype=tf.float32)
    Returns
      outputs: tf.Tensor(dtype=tf.float32
    """
    inputs = inputs[0] if isinstance(inputs, list) else inputs
    if inputs.shape != len(mask.shape): mask = tf.expand_dims(mask, -1)
    return self._prediction_processing(inputs)

  def set_new_model(self, model: tf.keras.Model, optimizer: tf.keras.optimizers, jit_compile=True) -> None:
    self.model = model
    self.model.compile(optimizer=optimizer, jit_compile=jit_compile)

  def validate_args(self):
      assert isinstance(self.model, tf.keras.Model), "Передан неверный аргумент, должно быть tf.keras.Model"

  @property
  def layers(self, ) -> list:
      return self.model.layers

  @property
  def weights(self, ) -> list:
      return self.model.weights
  
  @property
  def summary(self, ) -> None:
      print(self.name)
      self.model.summary()

  def get_weights(self, ) -> list:
      return self.model.get_weights()

  def set_weights(self, weights: list) -> None:
      self.model.set_weights(weights)

  @tf.function(reduce_retracing=True,
                 jit_compile=False,
                 experimental_autograph_options = tf.autograph.experimental.Feature.ALL)
  def calculate_gradients(self, **kwargs) -> dict:
      """
      Вычисляет градиенты, лосс, td-ошибку

      Kwargs:
          dict содержащий батч, таргет, маску, опционально приоритетные веса

      Returns: 
          dict содержащий лоссы и td-ошибку
      """
      with tf.GradientTape(persistent=False) as tape:
          Q = self.model(kwargs['state'], training=True)
          Q = self.prediction_processing(Q, kwargs['mask'])
          if Q.shape != len(kwargs['Qtarget'].shape): Q = tf.expand_dims(Q, -1)

          td_error = kwargs['Qtarget'] - Q
          loss = self.loss(kwargs['Qtarget'], Q)*kwargs.get('weights', 1.0)
      gradients = tape.gradient(loss, self.model.trainable_variables)
      return {'gradients': gradients, 'loss': loss, 'td_error': td_error}

  @tf.function(reduce_retracing=True,
                 jit_compile=False,
                 experimental_autograph_options = tf.autograph.experimental.Feature.ALL)
  def update_weights(self, **kwargs) -> dict:
      """
      Выполняет шаг отимизатора

      Kwargs:
          dict содержащий батч, таргет, маску, опционально приоритетные веса

      Returns: 
          dict содержащий лоссы и td-ошибку
      """
      gradients, loss, td_error = self.calculate_gradients(**kwargs).values()
      self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
      return {'loss': loss, 'td_error': td_error}
