import tensorflow as tf
import numpy as np

from .base_algo import Base_Algo
from rl_lib.src.replay_buffers.replay_buffer import ReplayBuffer
from rl_lib.src.explore_env.exploration_manager import ExplorationManager

class SimpleQ(Base_Algo, ):
  """Произовдит все вычисления необходимые для Q-learning
  """
  def __init__(self, action_model: object, target_model: object, config: dict, **kwargs):
    
    Base_Algo.__init__(self, action_model, target_model, config, **kwargs)
    config = link_data_inside_the_config(self.config)
    self.buffer = ReplayBuffer(**config.get("buffer_config", {}))
    self.exploration = ExplorationManager(**config.get("exploration_config", {}))
    
    self.discount_factor = self.config['model_config']['discount_factor']
    self.n_step = self.config['model_config']['n_step']

    self.batch_size = self.config['model_config'].get("batch_size")
    self.double_network = self.config['model_config'].get("double_network")
    self.priority = self.config['model_config'].get("priority")
    self.tau = self.config['model_config'].get("tau")

    self.recurrent = False
    self.batch_dims = 1
    self.ind_axis = -1
  
  def add(self, data: tuple, priority = None) -> None:
    """
    Добавляет переходы в буфер
    Аргументы:
      data: tuple(state, action, reward, done, next_state)
      priority: np.array (только для приоритетных буферов)
    """
    self.buffer.add(data, priority)

  def calculate_double_q(self, **kwargs):
    Qaction = self.action_model(kwargs['next_state'])
    Qtarget = self.target_model(kwargs['next_state'])
    Qaction = Qaction[0] if isinstance(Qaction, list) else Qaction
    Qtarget = Qtarget[0] if isinstance(Qtarget, list) else Qtarget
    if kwargs["p_double"] < 0.5 : Qtarget = self.get_best_action(Qtarget, Qaction)
    else: Qtarget = self.get_best_action(Qaction, Qtarget)
    return Qtarget
    
  def calculate_gradients(self, batch = None):
    if batch == None: batch = self.choice_model_for_double_calculates(**batch)
    batch = self.choice_model_for_double_calculates(**batch)
    return self.action_model.calculate_gradients(**batch) if batch['p_double'] > 0.5 else self.target_model.calculate_gradients(**batch)
    
  def calculate_new_best_action(self, **kwargs) -> tf.Tensor:
    """Вычислеят новое лучшее действие для получения таргета"""
    if self.double_network:
        Qtarget = self.calculate_double_q(**kwargs)
    else:
        Qtarget = self.target_model(kwargs['next_state'])
        Qtarget = Qtarget[0] if isinstance(Qtarget, list) else Qtarget
        Qtarget = self.get_best_action(Qtarget, Qtarget)
    return Qtarget
    
  @tf.function(reduce_retracing=True,
                 jit_compile=False,
                 experimental_autograph_options = tf.autograph.experimental.Feature.ALL)
  def calculate_target(self, **kwargs):
    Qtarget = self.calculate_new_best_action(**kwargs)
    dones = tf.ones(kwargs['done'].shape, dtype=tf.dtypes.float32) 
    dones = dones - kwargs['done']
    Qtarget = kwargs['reward'] + (self.discount_factor**self.n_step) * Qtarget * dones
    if self.recurrent:
        Qtarget = Qtarget[:, kwargs.get('recurrent_skip', 10):]
    return Qtarget

  def check_fullness_buffer(self):
    """Проверяет наполненость буфера, возвращает true если в буфере элементов больше батча"""
    if self.buffer.real_size > self.batch_size: return True
    else: return False

  def choice_model_for_double_calculates(self, **batch):
    batch['p_double'] = tf.random.uniform((1,), minval = 0.0, maxval = 1.0) if self.double_network else 1.
    batch['Qtarget'] = self.calculate_target(**batch)
    return batch
  
  def _get_action(self, observation: tf.Tensor) -> tf.Tensor:
    """Возвращает ценность дейтсвий Q(s,a) всех действий на основе наблюдения"""
    return self.sample_action(self.action_model.check_input_shape(observation))
  
  def get_action(self, observation: tf.Tensor) -> float:
    """Возвращает действие на основе наблюдения с учетом исследования"""
    action = self.exploration(self._get_action(observation))
    if isinstance(action, int) or action.shape == 1: return int(action)
    else: return action.numpy()

  def get_test_action(self, observation: tf.Tensor) -> float:
    """Возвращает действие на основе наблюдения без исследования"""
    action = self.exploration.test(self._get_action(observation))
    if isinstance(action, int) or action.shape == 1: return int(action)
    else: return action.numpy()
  
  def get_batch(self):
    """Получает батч из буфера"""
    return self.buffer.sample(self.batch_size)

  def get_batch_and_td_error(self):
    batch = self.get_batch()
    batch['batch_dims'] = self.batch_dims
    batch['p_double'] = 1.
    td_error = self.calculate_gradients(batch)['td_error']
    return {'td_error': td_error.numpy(), 'batch': batch}
    
  def get_best_action(self, Qaction, Qtarget):
    ind = tf.argmax(Qaction, axis=self.ind_axis)
    Qtarget = tf.gather(Qtarget, ind, batch_dims=self.batch_dims)
    return Qtarget
    
  def get_gradients(self) -> tf.Tensor:
    """Вычисляет градиенты и возвращает их"""
    batch = self.get_batch()
    batch['batch_dims'] = self.batch_dims
    batch['p_double'] = 1.
    return self.calculate_gradients(batch)['gradients']
    
  def load(self, ) -> None:
    """Загружает алгоритм"""
    self.action_model.load(self.path)
    self.target_model.load(self.path)
    self.buffer.load(self.path)
    self.exploration.load(self.path)  
  
  def reset(self) -> None:
    """Сбрасывает внутренние данные модели"""  
    self.buffer.reset()
    self.exploration.reset()
    self.initial_model()
    
  def _train_step(self, **batch) -> dict:
    """Вспомогательная train_step"""
    batch = self.choice_model_for_double_calculates(**batch)
    batch['batch_dims'] = self.batch_dims
    return self.action_model.update_weights(**batch) if batch['p_double'] > 0.5 else self.target_model.update_weights(**batch)
    
  def train_step(self) -> np.array:
    """Вычисляет полный обучающий шаг"""
    if not self.check_fullness_buffer(): return 0
    batch = self.get_batch()
    result = self._train_step(**batch)
    td_error = result['td_error'].numpy()
    loss = result['loss'].numpy()
    assert not np.all(np.isnan(td_error)), "td_error не может быть nan"
    if self.priority: self.buffer.update_priorities(batch['data_idxs'], loss if not self.recurrent else loss[:, -1])
    if self.tau != 1:
      _ = self.copy_weights()
    return np.mean(td_error)
  
  def save(self) -> None:
    """Сохраняет алгоритм"""
    self.action_model.save(self.path)
    self.target_model.save(self.path)
    self.buffer.save(self.path)
    self.exploration.save(self.path)

  def summary(self) -> None:
    """Выводит архитектуру модели"""
    self.action_model.summary



def link_data_inside_the_config(config):
  discount_factor = config['model_config']['discount_factor']
  n_step = config['model_config']['n_step']
  action_space = config['model_config']['action_space']
  priority = config['model_config']['priority']

  config['buffer_config']['priority'] = priority
  config['buffer_config']['discount_factor'] = discount_factor
  config['buffer_config']['n_step'] = n_step
  config['exploration_config']['strategy_config']['action_space'] = action_space
  return config
