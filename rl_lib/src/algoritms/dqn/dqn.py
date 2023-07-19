from tensorflow.keras import layers

from ..models.model import Model
from ..algoritms.simple_q import SimpleQ

class DQN_Model(ModelNN, ModelIO, Model):
  def __init__(self, config = None, model = None, **kwargs):
    super().__init__(model = model, config = config, **kwargs)
  
  def _prediction_processing(self, input_data):
    pass

  def _update_next_state(self, state, action):
    pass

  def initial_state(self):
    pass

class DQN_Algo(SampleQ):
  def __init__(self, config, model):
    action_model = DQN_Model(model = model, config = config, name = "DQN_action" + config.get("name", ""))
    target_model = DQN_Model(model = model, config = config, name = "DQN_target" + config.get("name", ""))
    super().__init__(model = model, config = config)

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
