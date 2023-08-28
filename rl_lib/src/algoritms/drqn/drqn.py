import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from rl_lib.src.models.model import Model
from rl_lib.src.algoritms.simple_q import SimpleQ

class DRQN_Model(Model):
  def __init__(self, config = {},**kwargs):
    super().__init__(model_config = config.get('model_config', {}), config = config,  default_config_path=__file__, **kwargs)
    self.h_t, self.c_t, self.new_h_t, self.new_c_t = None, None, None, None
    self.lstm_size = config['model_config'].get("lstm_size", 64)
  
  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    return super().__call__([inputs, self.h_t, self.c_t] if not isinstance(inputs, list) else inputs)

  def _initial_model(self):
    input_shape = self._config['model_config']["input_shape"]
    action_space =  self._config['model_config']["action_space"]
    if len(input_shape) == 1:
      return self.create_model(input_shape, action_space, self.lstm_size)
    else:
      return self.create_model_with_conv(input_shape, action_space, self.lstm_size)
    
  def initial_state(self):
    """Инициализирует внутреннее состояние рекуррентной сети"""
    self.h_t = tf.zeros((1, self.lstm_size),dtype=tf.float32)
    self.c_t = self.h_t
  
  def get_states(self) -> tuple:
    """Возвращает кортеж внутренних состояний реккурентной сети"""
    return  tf.squeeze(self.h_t.numpy()), tf.squeeze(self.c_t.numpy())
  
  def loss(self, target: tf.Tensor, predict: tf.Tensor) -> tf.Tensor:
    """Вычисляет и возвращает потери в соответствии с функцией потерь"""
    return tf.math.squared_difference(target, predict)

  def make_mask(self, action) -> tf.Tensor:
    """Создает маску по действиям """
    return tf.one_hot(action, self.output_spec()[-1])
  
  def _prediction_processing(self, inputs: tf.Tensor, **kwargs):
    mask = self.make_mask(kwargs['action'])
    while len(inputs.shape) < len(mask.shape): mask = tf.expand_dims(mask, -1)
    return tf.reduce_sum(tf.multiply(inputs, mask), axis=kwargs['batch_dims'])[:, kwargs['recurrent_skip']:]


  def _update_next_state(self):
    """Обновляет внутреннее состояние рекуррентной сети"""
    self.h_t, self.c_t = self.new_h_t, self.new_c_t

  @staticmethod
  def create_model(input_shape: tuple, action_space: int, lstm_size: int) -> tf.keras.Model:
    """Создает модель tf.keras.Model, архитектура DRQN"""
    input_layer = layers.Input(shape=input_shape, )
    h_t_input = layers.Input(shape=(lstm_size, ), ) 
    c_t_input = layers.Input(shape=(lstm_size, ), ) 

    lstm = layers.LSTM(lstm_size, activation='tanh', recurrent_activation='sigmoid', return_sequences = True, 
                          return_state=True, stateful = False)(input_layer, initial_state = [h_t_input, c_t_input])
    dence_layer1 = layers.Dense(256, activation='relu')(input_layer)
    dence_layer2 = layers.Dense(128, activation='relu')(dence_layer1)
    dence_out = layers.Dense(action_space, activation=None)(dence_layer2)
    
    return tf.keras.Model(inputs=[input_layer, h_t_input, c_t_input], outputs=[dence_out, lstm[1], lstm[2]])

  @staticmethod
  def create_model_with_conv(input_shape: tuple, action_space: int, lstm_size: int) -> tf.keras.Model:
    """Создает модель tf.keras.Model, архитектура DRQN, начальные слои - сверточные"""
    input_layer = layers.Input(shape=input_shape, )
    h_t_input = layers.Input(shape=(lstm_size, ), ) 
    c_t_input = layers.Input(shape=(lstm_size, ), ) 

    cov_layer1 = layers.Conv2D(32, 7, activation='relu')(input_layer)
    cov_layer2 = layers.Conv2D(64, 5, activation='relu')(cov_layer1)
    cov_layer3 = layers.Conv2D(64, 3, activation='relu')(cov_layer2)
    conv_out = layers.Flatten()(cov_layer3)
    lstm = layers.LSTM(lstm_size, activation='tanh', recurrent_activation='sigmoid', return_sequences = True, 
                          return_state=True, stateful = False)(conv_out, initial_state = [h_t_input, c_t_input])
    dence_layer1 = layers.Dense(256, activation='relu')(lstm[0])
    dence_layer2 = layers.Dense(128, activation='relu')(dence_layer1)
    dence_out = layers.Dense(action_space, activation=None)(dence_layer2)
    
    return tf.keras.Model(inputs=[input_layer, h_t_input, c_t_input], outputs=[dence_out, lstm[1], lstm[2]])
    
class DRQN(SimpleQ):
  def __init__(self, config):
    super().__init__(DRQN_Model, DRQN_Model, config,  default_config_path=__file__, algo_name = "DRQN", name = "DRQN_Model_" + config.get('model_config','').get('name',''))
    
    self.initial_state()
    self.recurrent_skip = self.config['buffer_config']['recurrent_skip']
    self.trace_length = self.config['buffer_config']['trace_length']
    self.recurrent = True
    self.batch_dims = 2  
  
  def add(self, data: tuple, priority = None) -> None:
    """
    Добавляет переходы в буфер
    Аргументы:
      data: tuple(state, action, reward, done, next_state)
      priority: np.array (только для приоритетных буферов)
    """
    super().add((*data, *self.action_model.get_states()), priority)
    self._update_next_state()

  def initial_state(self):
    """Сбравсывает внутренне состояние lstm"""
    self.action_model.initial_state()
  
  def _get_action(self, observation: tf.Tensor) -> tf.Tensor:
    """Возвращает ценность дейтсвий Q(s,a) всех действий на основе наблюдения"""
    predict = super()._get_action(observation)
    action, self.action_model.new_h_t, self.action_model.new_c_t = predict
    return action

  def get_test_action(self, observation: tf.Tensor) -> float:
    action = super().get_test_action(observation)
    self._update_next_state()
    return action
  
  def get_batch(self, ):
    batch = super().get_batch()

    new_h_t, new_c_t = tf.squeeze(batch['h_t'][:, 1:],axis=1), tf.squeeze(batch['c_t'][:, 1:],axis=1)
    h_t, c_t = tf.squeeze(batch['h_t'][:, :-1],axis=1), tf.squeeze(batch['c_t'][:, :-1],axis=1)
    batch['state'] = [batch['state'], h_t, c_t]
    batch['next_state'] = [batch['next_state'], new_h_t, new_c_t]
    batch['recurrent_skip'] = self.recurrent_skip
    batch['trace_length'] = self.trace_length

    if self.priority: batch['weights'] = np.repeat(np.expand_dims(batch['weights'], -1), self.trace_length-self.recurrent_skip, axis=1)
    return batch

  def _update_next_state(self):
    """Обновляет внутреннее состояние lstm новым состоянием lstm"""
    self.action_model._update_next_state()

  
