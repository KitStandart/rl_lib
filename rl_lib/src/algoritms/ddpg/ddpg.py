from typing import Any
import tensorflow as tf
from tensorflow.keras import layers

from rl_lib.src.models.model import Model
from rl_lib.src.algoritms.simple_q import SimpleQ
from rl_lib.src.algoritms.a2c.actor_critic import Actor_Critic_Model



class DDPG_Model(Actor_Critic_Model):
  def __init__(self, config = {},**kwargs):
    super().__init__(config=config, **kwargs)

  def update_weights(self, **kwargs) -> dict:
      """
      Выполняет обновление весов по алгоритму DDPG отимизатора

      Kwargs:
          dict содержащий батч, таргет, маску, опционально приоритетные веса

      Returns: 
          dict содержащий лоссы и td-ошибку
      """

      kwargs['action'] = self.actor_model(kwargs['next_state']) 
      _ = self.actor_model.update_weights(**kwargs)
      loss = self.critic_model.update_weights(**kwargs)
      return {'loss': loss['loss'], 'td_error': loss['td_error']}
  
class DDPG(SimpleQ):
  def __init__(self, config):
    super().__init__(DDPG_Model, DDPG_Model, config,  default_config_path=__file__, algo_name = "DDPG_Model", name = "DDPG_Model_" + config.get('model_config','').get('name',''))

  def _prediction_processing(self, input_data):
    pass

  def _update_next_state(self, state, action):
    pass

  def initial_state(self):
    pass
  
  def get_batch(self, ):
    batch = super().get_batch()
    batch['reward'] = tf.reshape(batch['reward'], (self.batch_size, 1))
    batch['done'] = tf.reshape(batch['done'], (self.batch_size, 1))
    return batch
  
  def get_best_action(self, Qaction, Qtarget):
    return Qtarget 

  @tf.function(reduce_retracing=None, jit_compile=None, experimental_autograph_options=None)
  def _copy_weights(self, action_model_weights: list, target_model_weights: list) -> tf.constant:
      """Копирует веса из модели действия в целевую модель"""
      for a_w, t_w in zip(action_model_weights, target_model_weights):
          new_weights = tf.add(tf.multiply(self.tau, a_w), tf.multiply((1-self.tau), t_w))
          t_w.assign(tf.identity(new_weights))
      return tf.constant(1)
  
  def copy_weights(self) -> tf.constant:
     """Копирует веса из модели действия в целевую модель"""
     _ = self._copy_weights(self.action_model.actor_model.weights, self.target_model.actor_model.weights)
     _ = self._copy_weights(self.action_model.critic_model.weights, self.target_model.critic_model.weights)
     return tf.constant(1)
  
  @tf.function(reduce_retracing=True,
                jit_compile=True,
                experimental_autograph_options = tf.autograph.experimental.Feature.ALL)
  def sample_action(self, state: tf.Tensor) -> tf.Tensor:
      """Возвращает предсказания модели на основе текущих наблюдений"""
      predict = self.action_model.actor_model(state)
      return self.squeeze_predict(predict)