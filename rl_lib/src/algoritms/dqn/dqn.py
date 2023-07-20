import tensorflow as tf
from tensorflow.keras import layers

from rl_lib.rl_lib.src.models.model import Model
from rl_lib.rl_lib.src.algoritms.simple_q import SimpleQ
from rl_lib.rl_lib.src.data_saver.utils import load_default_config

class DQN_Model(Model):
  def __init__(self, config = {},**kwargs):
    super().__init__(model_config = config.get('model_config', {}), config = config,  default_config_path=__file__, **kwargs)
  
  def _prediction_processing(self, input_data):
    pass

  def _update_next_state(self, state, action):
    pass

  def initial_state(self):
    pass
    
  @staticmethod
  def create_model(input_shape: tuple, action_space: int) -> tf.keras.Model:
    """Создает модель tf.keras.Model, архитектура DQN"""
    input_layer = layers.Input(shape=input_shape, )
    dence_layer1 = layers.Dense(256, activation='relu')(input_layer)
    dence_layer2 = layers.Dense(256, activation='relu')(dence_layer1)
    dence_out = layers.Dense(action_space, activation=None)(dence_layer2)
    
    return tf.keras.Model(inputs=input_layer, outputs=dence_out)

  @staticmethod
  def create_model_with_conv(input_shape: tuple, action_space: int) -> tf.keras.Model:
    """Создает модель tf.keras.Model, архитектура DQN, начальные слои - сверточные"""
    input_layer = layers.Input(shape=input_shape, )
    cov_layer1 = layers.Conv2D(32, 7, activation='relu')(input)
    cov_layer2 = layers.Conv1D(64, 5, activation='relu')(cov_layer1)
    cov_layer3 = layers.Conv1D(64, 3, activation='relu')(cov_layer2)
    conv_out = layers.Flatten()(cov_layer3)

    dence_layer1 = layers.Dense(256, activation='relu')(conv_out)
    dence_layer2 = layers.Dense(256, activation='relu')(dence_layer1)
    dence_out = layers.Dense(action_space, activation=None)(dence_layer2)
    
    return tf.keras.Model(inputs=input_layer, outputs=dence_out)
    
class DQN(SimpleQ):
  def __init__(self, config):
    action_model = DQN_Model(config, name = "DQN_action" + config['model_config'].get("name", ""))
    target_model = DQN_Model(config, name = "DQN_target" + config['model_config'].get("name", ""))
    config.update(action_model.config)
    print(config)
    print(action_model.config)
    super().__init__(action_model, target_model, **config)

  def _prediction_processing(self, input_data):
    pass

  def _update_next_state(self, state, action):
    pass

  def initial_state(self):
    pass

  
