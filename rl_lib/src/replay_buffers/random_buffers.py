import numpy as np

from ..data_saver.utils import load_data, save_data
from .dict_array import DictArray


class _n_step_buffer:
    def __init__(self, **kwargs):
        self.buffer = [[]]
        self.steps = kwargs.get("n_step", 1)
        self.discount_factor = kwargs.get("discount_factor", 0.99)

    def clear(self, ):
        self.buffer = [[]]

    def add(self, memory_tuplet):
        state, action, reward, done, next_state, *another_data = memory_tuplet

        if len(self.buffer[0]) == 0:
            self.buffer[-1] = [state, action, [reward], None,
                               None, *[None for _ in range(len(another_data))]]

        if len(
            self.buffer[0][2]
            ) == self.steps-1 or (
                len(self.buffer[0][2]) == self.steps and self.steps == 1
                ):
            self.buffer[0][3] = int(done)
            self.buffer[0][4] = next_state
            for i in range(len(another_data)):
                self.buffer[0][i+5] = another_data[i]

        for j in range(len(self.buffer)-1):
            self.buffer[j][2].append(
                reward * self.discount_factor**len(self.buffer[j][2]))

        if len(self.buffer) != 1:
            self.buffer[-1] = [state, action, [reward], None,
                               None, *[None for _ in range(len(another_data))]]

        self.buffer.append([])

        if len(self.buffer) > self.steps:
            self.buffer[0][2] = sum(self.buffer[0][2])
            return_data = self.buffer[0]
            self.buffer = self.buffer[1:]
            return return_data
        return None


class Random_Buffer:
    '''Сохраняет переходы (s,a,r,d,s') и возвращает батчи.

    Аргументы:
        size: int. Размер буфера
        n_step: int. N-step алгоритм
        discount_factor: float
        num_var: int, Кол-во сохраянемых переменных,
                                по умполчанию 5 (s, a, r, d, s')
    '''

    def __init__(self, **kwargs):
        n_step = kwargs.get("n_step", 1)
        self.size = kwargs.get("size", 100000)
        # discount_factor = kwargs.get("discount_factor", 0.99)
        num_var = kwargs.get("num_var", 5)
        self.hash_table = kwargs.get("var_names",
                                     {"state": 0,
                                      "action": 1,
                                      "reward": 2,
                                      "done": 3,
                                      "next_state": 4}
                                     )

        # буфер для хранения перехода
        self.data = DictArray((self.size, num_var), dtype=object)
        self.name = "Random_Buffer"
 
        # размер буфера
        self.count = 0
        self.real_size = 0

        self.n_step_buffer = _n_step_buffer(**kwargs) if n_step > 1 else None

    def clear(self, ):
        self.data = DictArray(self.data.shape, dtype=object)
        self.count = 0
        self.real_size = 0
        if self.n_step_buffer is not None:
            self.n_step_buffer.clear()

    def add(self, samples: tuple, args=None):
        """Добавляет данные в буфер s,a,r,n_s,d,
        индексы данных должны быть равны индексам в hash_table.
        Автоматической проверки нет"""
        if self.n_step_buffer is not None:
            result = self.n_step_buffer.add(samples)
            if result is not None:
                return self._add_data(result)
            return False
        else:
            return self._add_data(samples)

    def sample(self, batch_size, idx=None):
        """Возвращает батч: dict"""
        if np.any(idx) is None:
            idx = self._get_idx(batch_size)
        data = self.data[idx]
        return {key: data[val]
                for key, val in self.hash_table.items()}

    def save(self, path):
        path += self.name
        save_data(path, {
            'data': self.data,
            'count': self.count,
            'size': self.size,
            'real_size': self.real_size
        })

    def load(self, path):
        path += self.name
        data = load_data(path)
        self.data = data['data']
        self.count = data['count']
        self.size = data['size']
        self.real_size = data['real_size']

    def _add_data(self, samples):
        self.data[self.count] = samples
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)
        return True

    def _get_idx(self, batch_size):
        return np.random.choice(self.real_size, size=batch_size, replace=False)


class Random_Recurrent_Buffer(Random_Buffer):
    '''
    Аргументы:
        size: int. Размер буфера
        n_step: int. N-step алгоритм
        discount_factor: float
        num_var: int, Кол-во сохраняемых переменных,
                        по умполчанию 7 (s, a, r, d, s', h, c)
        trace_length: int. Длина возращаемой последовательности
    '''

    def __init__(self, **kwargs):
        kwargs["num_var"] = 7
        kwargs["var_names"] = {"state": 0,
                               "action": 1,
                               "reward": 2,
                               "done": 3,
                               "next_state": 4,
                               "h_t": 5,
                               "c_t": 6}
        Random_Buffer.__init__(self, **kwargs)
        self.name = "Random_Recurrent_Buffer"
        self.trace_length = kwargs.get("trace_length", 10)

    def _make_linspace(self, idx):
        idx = np.linspace(start=idx - self.trace_length,
                        stop=idx, num=self.trace_length+1,
                        dtype=int, axis=1)[:, :-1]
        return idx

    def _get_idx(self, batch_size, *args, **kwargs):
        if self.real_size != self.size:
            return self._make_linspace(np.random.randint(low=self.trace_length,
                                    high=self.real_size, size=(batch_size,)))
        else:
            return self._make_linspace(np.random.randint(
                low=-self.size + self.count + self.trace_length,
                high=self.count, size=(batch_size,)
                ))

    def sample(self, batch_size, idx=None):
        if idx is not None:
            idx = self._make_linspace(idx)
        return Random_Buffer.sample(self, batch_size, idx)

    def stack(self, data, batch_size):
        return np.asarray([np.stack(data[i]) for i in range(batch_size)])
