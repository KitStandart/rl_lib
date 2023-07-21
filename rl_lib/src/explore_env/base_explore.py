import abc

class Base_Explore(abc.ABC):
  """Абстрактный класс представляющий общий интерфейс для всех классов исследования
  нейронной сетью среды обучения
  
  """
  def __init__():
    pass

  @property
  @abc.abstractmethod
  def name(self):
    """Возвращает имя стратегии"""
  
  @abc.abstractmethod
  def reset(self, ) -> None:
    """Выполняет внутренний сброс"""
  
  @abc.abstractmethod
  def save(self, path) -> None:
    """Сохраняет какие либо внутренние переменные"""
    
  @abc.abstractmethod
  def load(self, path) -> None:
    """Загружает какие либо внутренние переменные"""

  @abc.abstractmethod
  def __call__(self, action) -> int:
    """Возвращает действие в соответствии с стратегией исследования"""

  @abc.abstractmethod
  def test(self, action) -> int:
    """Возвращает действие в соответствии с стратегией тестирования"""
