import tensorflow as tf
import numpy as np

from .base_algo import Base_Algo
from rl_lib.rl_lib.src.replay_buffers.replay_buffer import ReplayBuffer
from rl_lib.rl_lib.src.explore_env.exploration_manager import ExplorationManager

class SimpleQ(Base_Algo):
  """Произовдит все вычисления необходимые для Q-learning
  """
  def __init__(self, action_model: object, target_model: object, **config: dict):
    super().__init__(action_model, target_model, **config)
    
    self.buffer = ReplayBuffer(config.get("replay_buffer", {}))
    self.exploration = ExplorationManager(config.get("exploration", {}))
    
    self.discount_factor = self.buffer.discount_factor
    self.n_step = self.buffer.n_step

    self.batch_size = kwargs.get("batch_size", 32)
    self.double_network = kwargs.get("double_network", True)
    self.priority = kwargs.get("priority", False)
    self.tau = kwargs.get("tau", 0.01)

    self.batch_dims = -1
    self.ind_axis = -1

    self.path = self.action_model.path

  def add(self, data, priority = None):
    """Добавляет переходы в буфер"""
    self.buffer.add(data, priority)
  
  def calculate_new_best_action(self) -> tf.Tensor:
    """Вычислеят новое лучшее действие для получения таргета"""

  @tf.function(reduce_retracing=True,
                 jit_compile=False,
                 experimental_autograph_options = tf.autograph.experimental.Feature.ALL)
  def calculate_Qtarget(self, **kwargs):
    Qtarget = self.calculate_best_acitons(**kwargs)
    dones = tf.ones((self.batch_size,), dtype=tf.dtypes.float32) if not self.recurrent else tf.ones((self.batch_size, kwargs.get('trace_length', 10)), dtype=tf.dtypes.float32)
    dones = dones - kwargs['done']
    Qtarget = kwargs['reward'] + (self.discount_factor**self.n_step) * Qtarget * dones
    if self.recurrent:
        Qtarget = Qtarget[:, kwargs.get('recurrent_skip', 10):]
    return Qtarget

  def get_batch(self):
    """Получает батч из буфера"""
    return self.buffer.sample(self.batch_size)
    
  def get_best_action(self, Qaction, Qtarget):
    ind = tf.argmax(Qaction, axis=self.ind_axis)
    Qtarget = tf.gather(Qtarget, ind, batch_dims=self.batch_dims)
    return Qtarget
    
  def get_gradients(self) -> tf.Tensor:
    """Вычисляет градиенты и возвращает их"""

  def load(self, path) -> None:
    """Загружает алгоритм"""
    self.action_model.load(path)
    self.target_model.load(path)
    self.buffer.load(path)
    self.exploration.load(path)

  def reset(self) -> None:
    """Сбрасывает внутренние данные модели"""  
    self.buffer.reset()
    self.exploration.reset()
    self.initial_model()
    
  def _train_step(self) -> dict:
    """Вспомогательная train_step"""

  def train_step(self) -> np.array:
    """Вычисляет полный обучающий шаг"""
    batch = self.get_batch()
    result = self._train_step(**batch)
    td_error = result['td_error'].numpy()
    loss = result['loss'].numpy()
    assert not np.all(np.isnan(td_error)), "td_error не может быть nan"
    if self.priority: self.buffer.update_priorities(batch['data_idxs'], loss if not self.recurrent else loss[:, -1])
    if self.tau != 1:
      _ = self.copy_weights()
    return np.mean(td_error)
      
  def save(self, path) -> None:
    """Сохраняет алгоритм"""
    self.action_model.save(path)
    self.target_model.save(path)
    self.buffer.save(path)
    self.exploration.save(path)

  def summary(self) -> None:
    """Выводит архитектуру модели"""
    self.action_model.summary()
