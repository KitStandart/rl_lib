import abc


class Model(abc.ABC):
  """Абстрактный базовый класс,
  представляющий общий интерфейс для всех алгоритмов и моделей в RL-Lib.

  Model определяет общие методы для ввода, вывода и базовых вычислений,
  которые должны быть реализованы в каждом конкретном алгоритме или модели.

  Этот класс служит в качестве основы для всех других классов в RL-Lib
  и обеспечивает единый интерфейс для работы с различными моделями.
  """

  def __init__(self):
    pass

  @property
  @abc.abstractmethod
  def input_spec(self) -> tuple:
    """Возвращает кортеж размера входных данных Модели"""

  @property
  @abc.abstractmethod
  def output_spec(self) -> tuple:
    """Возвращает кортеж размера выходных данных Модели"""

  @abc.abstractmethod
  def initial_state(self) -> None:
    """Инициализирует внутреннее состояние реккурентной Модели"""

  @abc.abstractmethod
  def _update_next_state(self) -> None:
    """Обновляет внутреннее состояние реккурентной Модели"""

class ModelIO(abc.ABC):
  def __init__(self, config):
    self._config = config

  @abc.abstractmethod
  def config(self) -> dict:
    """Возвращает конфигурацию алгоритма"""
    return self._config

  @abc.abstractmethod
  def save(self) -> None:
    """Сохраняет модель в директории"""

  @abc.abstractmethod
  def load(self) -> None:
    """Загружает модель из директории"""
