import tensorflow.keras.optimizers as optimizers
import tensorflow_addons as tfa

def get_optimizer(optimizer_name: str = "adam", optimizer_params: dict = {}, custom_optimizer: object = None) -> object:
  """Возврщает настроенный оптимизатор.
  Доступные оптимизаторы tensorflow:
    Adam
    LAMB
    Adadelta
    RMSprop
  Args:
    optimizer: str: Название оптимизатора
    optimizer_params: dict: Параметры оптимизатора
    cutom_optimizer: object: Класс кастомного потимизатора
  """
  if optimizer_name.lower() == 'adam':
    return optimizers.Adam(**optimizer_params)
    
  elif optimizer_name.lower() == 'lamb':
    return tfa.optimizers.LAMB(**optimizer_params)

  elif optimizer_name.lower() == 'cutom' and type(custom_optimizer) != None:
      return custom_optimizer(**optimizer_params)

  elif optimizer_name.lower() == 'adadelta':
    return optimizers.Adam(**optimizer_params)

  elif optimizer_name.lower() == 'rmsprop':
    return optimizers.Adam(**optimizer_params)
