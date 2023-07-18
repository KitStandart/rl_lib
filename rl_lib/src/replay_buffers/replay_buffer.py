from .random_buffers import *
from .priority_buffers import *

class Replay_Buffer:
  
  def __init__(self, **kwargs):
    """Сохраняет переходы и выполняет сэмплирование батчей
    Kwargs: 
        priority: bool True если приоритетный
        recurrent: bool True если рекуррентный
        size: int Размер буфера
        n_step: int 
        discount_factor: float
        num_var: int (Кол-во сохраянемых переменных, по умполчанию 5 (s, a, r, d, s'))
        eps: float
        alpha: float
        beta: float
        beta_changing: float
        beta_changing_curve: str
        max_priority: float Максимальный приоритет при добавлении новых данных
        trace_length: int. Длина возращаемой последовательности
    """
    if kwargs.get("priority", 0) :
      if kwargs.get("recurrent", 0):Prioritized_Replay_Recurrent_Buffer(**kwargs)
      else: self.buffer = Prioritized_Replay_Buffer(**kwargs)
    else:
      if kwargs.get("recurrent", 0):Random_Recurrent_Buffer(**kwargs)
      else: self.buffer = Random_Buffer(**kwargs)

  def clear():
    self.buffer.clear()
    
  def add(*args):
    self.buffer.add(*args)

  def sample(*args):
    return self.buffer.sample(*args)

  def update_priorities(*args):
    self.buffer.update_priorities(*args)

  def save(*args):
    self.buffer.save(*args)
    
  def load(*args):
    self.buffer.load(*args)
