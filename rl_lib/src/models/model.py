from .base_models import *

class Model(ModelNN, ModelIO, BaseModel, abc.ABC):
  """Абстрактный класс модели, который соединяет все методы классов ModelNN, ModelIO, BaseModel"""
  def __init__(self, **config: dict):
    super().__init__(**config)

  def input_spec(self):
    return self.model.layers[0].input_shape[0]

  def output_spec(self):
    return self.model.layers[-1].input_shape[0]

  def load(self):
    """Args: ModelIO.path, ModelNN.name"""
    self.model = tf.keras.models.load_model(self.path+self.name+'.h5')

  def save(self):
    """Args: ModelIO.path, ModelNN.name"""
    print(self.name)
    self.model.save(self.path+self.name+'.h5')
