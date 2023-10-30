from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from ...policy_gradient.a2c.actor_critic import Actor_Critic_Model
from ...value_based.simple_q import SimpleQ


class DDPG_Model(Actor_Critic_Model):
    def __init__(self, config={}, **kwargs):
        super().__init__(config=config, **kwargs)


class DDPG(SimpleQ):
    def __init__(self, config):
        self.actor_tau = config['actor_model_config']['model_config']['tau']
        self.critic_tau = config['critic_model_config']['model_config']['tau']
        super().__init__(DDPG_Model, DDPG_Model,
                         config, default_config_path=__file__,
                         algo_name="DDPG_Model",
                         name=("DDPG_Model_" +
                               config.get('model_config', '').get('name', '')))

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

    def _train_step(self, **batch) -> dict:
        """Вспомогательная train_step"""
        batch = self.choice_model_for_double_calculates(**batch)
        batch['batch_dims'] = self.BATCH_DIMS
        if self.priority:
            batch['weights'] = tf.expand_dims(batch['weights'], -1)
        if batch['p_double'] > 0.5:
            self.action_model.update_weights_actor(**batch)
            return self.action_model.update_weights_critic(**batch)
        else:
            self.target_model.update_weights_actor(**batch)
            return self.target_model.update_weights_critic(**batch)

    def copy_weights(self) -> tf.constant:
        """Копирует веса из модели действия в целевую модель"""
        _ = self._copy_weights(self.action_model.actor_model.weights,
                               self.target_model.actor_model.weights,
                               self.actor_tau)
        _ = self._copy_weights(self.action_model.critic_model.weights,
                               self.target_model.critic_model.weights,
                               self.critic_tau)
        return tf.constant(1)

    @tf.function(reduce_retracing=True,
                 jit_compile=True,
                 experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def sample_action(self, state: tf.Tensor) -> tf.Tensor:
        """Возвращает предсказания модели на основе текущих наблюдений"""
        predict = self.action_model.actor_model(state)
        return self.squeeze_predict(predict)
