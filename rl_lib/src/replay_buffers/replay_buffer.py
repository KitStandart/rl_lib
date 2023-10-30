import threading

from .priority_buffers import (Prioritized_Replay_Buffer,
                               Prioritized_Replay_Recurrent_Buffer)
from .random_buffers import Random_Buffer, Random_Recurrent_Buffer


class ReplayBuffer:
    """Сохраняет переходы и выполняет сэмплирование батчей

      Kwargs:
          priority: bool True если приоритетный
          recurrent: bool True если рекуррентный
          size: int Размер буфера
          n_step: int
          discount_factor: float
          num_var: int, Кол-во сохраянемых переменных,
                        по умполчанию 5 (s, a, r, d, s')
          eps: float
          alpha: float
          beta: float
          beta_changing: float
          beta_changing_curve: str
          max_priority: float,
                      Максимальный приоритет при добавлении новых данных
          trace_length: int. Длина возращаемой последовательности
      """

    def __init__(self, **kwargs):
        self._config = kwargs
        if kwargs.get("priority", 0):
            if kwargs.get("recurrent", 0):
                self.buffer = Prioritized_Replay_Recurrent_Buffer(**kwargs)
            else:
                self.buffer = Prioritized_Replay_Buffer(**kwargs)
        else:
            if kwargs.get("recurrent", 0):
                self.buffer = Random_Recurrent_Buffer(**kwargs)
            else:
                self.buffer = Random_Buffer(**kwargs)
        self.lock = threading.Lock()

    def add(self, *args):
        with self.lock: self.buffer.add(*args)

    @property
    def config(self):
        return self._config

    def clear(self):
        self.buffer.clear()

    def load(self, *args):
        self.buffer.load(*args)

    @property
    def name(self):
        return self.buffer.name

    @property
    def real_size(self):
        with self.lock: return self.buffer.real_size

    def sample(self, *args):
        with self.lock: return self.buffer.sample(*args)

    def save(self, *args):
        self.buffer.save(*args)

    def update_priorities(self, *args):
        with self.lock: self.buffer.update_priorities(*args)
