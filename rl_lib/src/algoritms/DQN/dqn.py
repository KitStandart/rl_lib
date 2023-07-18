from tensorflow.keras import layers

from ..models.base_model import *
from ..algoritms.base_algo import *

class DQN_Model(ModelNN, ModelIO, Model):
  def __init__(self, config = None, model = None, **kwargs):
    super().__init__(model = model, config = config, **kwargs)
  
  def _prediction_processing(self, input_data):
    pass

  def _update_next_state(self, state, action):
    pass

  def initial_state(self):
    pass

  def input_spec(self):
    return self.model.layers[0].input_shape[0]

  def output_spec(self):
    return self.model.layers[-1].input_shape[0]

  def load(self):
    """Args: ModelIO.path, ModelNN.name"""
    self.model = tf.keras.models.load_model(self.path+self.name)

  def save(self):
    """Args: ModelIO.path, ModelNN.name"""
    self.model.save(self.path+self.name)

class DQN_Algo(DQN_Model, Base_Algo):
  def __init__(self, config, model):
    super().__init__(model = model, config = config)
    self.action_model = DQN_Model(model = model, config = config, name = "DQN_action" + config.get("name", ""))
    self.target_model = DQN_Model(model = model, config = config, name = "DQN_target" + config.get("name", ""))
  
  def _initial_model(self):
    if len(self._config["input_shape"]) <= 1:
        return self.create_model(self._config["input_shape"], self._config["action_space"])
    else:
      return self.create_model_with_conv(self._config["input_shape"], self._config["action_space"])
  
  def initial_model(self):
      model = self._initial_model()
      self.action_model.set_new_model(model, tf.keras.optimizers.Adam())
      self.target_model.set_new_model(model, tf.keras.optimizers.Adam())
      self.target_model.set_weights(self.action_model.get_weights())

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
