from typing import Any
import numpy as np

class StructArray:
    """Структурированный массив"""
    def __init__(self, shape, dict_keys, dtype=object) -> None:
        self.data = np.zeros(shape=shape, 
                        dtype=(
                            [
                                (key, dtype) for key in sorted(dict_keys)
                            ]
                        )
                        )
        self.dict_keys = sorted(dict_keys)
        self.dtype = dtype
    
    def __getitem__(self, index):
        data = self.data[index]
        return {key: np.asarray(data[key]).astype(np.float32)
                if isinstance(index, int) else StructArray.stack(data[key], axis=0).astype(np.float32) 
                for key in self.dict_keys}
    
    def __setitem__(self, index, values):
        "values = (state, action, reward, next_state, done, *other_data)"
        self.data[index] = tuple(values[key] for key in self.dict_keys)

    @staticmethod
    def stack(array, axis=0):
        return np.stack(array, axis=axis)

class NonStructArray:
    """Не структурированный массив"""
    def __init__(self, shape, dtype=object) -> None:
        self.data = np.zeros(shape=shape, dtype=dtype)
        self.dtype = dtype
    
    def __getitem__(self, index):
        return StructArray.stack(self.data[index])
    
    def __setitem__(self, index, values):
        "values = (state, action, reward, next_state, done, *other_data)"
        self.data[index] = values
    
class DictArray:
    """
    Класс реализующий сохранение/ извлечение данных в структурированные массивы numpy
    """
    def __init__(self, shape, dtype=object) -> None:
        self.dtype = dtype
        self.initialized = False
        self.shape = shape
        self.data = np.zeros((shape[1], ), dtype=object) #В этом массиве мы будем хранить вложенные массивы (s,a,r,s',d)

    def __getitem__(self, index):
        return tuple(self.data[i][index] for i in range(self.shape[1]))
    
    def __setitem__(self, index, values):
        "values = (state, action, reward, next_state, done, *other_data)"
        if not self.initialized: self.init_array(values)
        for i in range(self.shape[1]):
            self.data[i][index] = values[i]

    def choose_array_type(self, data):
        if isinstance(data, dict): return self.init_struct_array((self.shape[0], ), data.keys())
        else: return self.init_non_struct_array((self.shape[0], ))

    def init_array(self, data):
        for i, d in zip(range(self.shape[0]), data):
            self.data[i] = self.choose_array_type(d)
        self.initialized=True    

    def init_struct_array(self, shape, dict_keys):
        return StructArray(shape, dict_keys, dtype=self.dtype)
    
    def init_non_struct_array(self, shape):
        return NonStructArray(shape=shape, dtype=self.dtype)

    
    