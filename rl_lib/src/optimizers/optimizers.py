import tensorflow.keras.optimizers as optimizers
import tensorflow_addons as tfa

def set_optimizer(self, optimizer: str = "adam", optimizer_params: dict = {}, cutom_optimizer: object = None) -> object:
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
  if optimizer.lower() == 'adam':
    return optimizers.Adam(**optimizer_params)
    
  elif optimizer.lower() == 'lamb':
    return tfa.optimizers.LAMB(**optimizer_params)

  elif optimizer.lower() == 'cutom' and type(cutom_optimizer) != None:
      return cutom_optimizer(**optimizer_params)

  elif optimizer.lower() == 'adadelta':
    return optimizers.Adam(**optimizer_params)

  elif optimizer.lower() == 'rmsprop':
    return optimizers.Adam(**optimizer_params)
