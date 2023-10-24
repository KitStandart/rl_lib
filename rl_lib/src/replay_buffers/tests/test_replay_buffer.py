import os
from copy import deepcopy
from shutil import rmtree

from rl_lib.src.replay_buffers.replay_buffer import ReplayBuffer


class Test_Replay_Buffer:
    """
    Производит тестирование буфера

    buffer_args:
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
          max_priority: float Максимальный приоритет
                              при добавлении новых данных
          trace_length: int. Длина возращаемой последовательности
    """

    def __init__(self, buffer_args):
        self.buffer = ReplayBuffer(**buffer_args)
        self.path = os.getcwd() + '/test_replay_buffer/'
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def __exit__(self):
        """Удаляет созданную папку с файлами, если есть"""
        if os.path.isdir(self.path):
            rmtree(self.path)

    def test_new_init_args(self, **buffer_args):
        """Проверяет переинициализацию с новыми аргументами"""
        self.buffer = ReplayBuffer(**buffer_args)

    def test_add_data(self):
        """Проверяет возможность добавить в буфер данные
        """
        pass

    def test_samples(self):
        """Сэмплирует батчи из буфера и проверяет размерности,
        количество аргументов
        """
        pass

    def test_save(self):
        """Выполняет сохранение буфера и проверяет появился ли файл буфера
        """
        self.buffer.save(self.path)
        print("Буфер сохранен")
        files = os.listdir(self.path)
        file_names = [f.split('.')[0] for f in files if os.path.isfile(
            os.path.join(self.path, f))]
        print("Проверка сохранения буфера")
        print(f"Найдено {len(file_names)} файлов: ", *file_names)
        assert self.buffer.name in file_names, """Файл не найден,
                                                проверка не пройдена"""
        print('Успешно тест сохранения данных')

    def test_load(self):
        """Выполняет test_save, потом загружает и
        проверяет соответствуют ли загруженные файлы сохраненным
        """
        self.buffer.save(self.path)
        copy_buffer = deepcopy(self.buffer)

        self.buffer.load(self.path)
        assert self.check_load_data(
            copy_buffer.buffer.__dict__, self.buffer.buffer.__dict__), "Файлы загрузки не соответствуют настоящим файлам"
        print("Успешный тест зарузки данных")

    def check_load_data(self, real_data: dict, loaded_data: dict) -> bool:
        for key, value in real_data.items():
            if key == 'tree':
                continue
            if key == 'trace_window':
                if not self.check_load_data(value.__dict__,
                                            loaded_data[key].__dict__):
                    return False
                continue
            if key == 'data':
                if loaded_data[key].all() != value.all():
                    return False
                continue
            if loaded_data[key] != value:
                return False
        return True

    def test_all_buffers(self, buffers: list):
        for buffer_type in buffers:
            self.test_new_init_args(buffer_type)
            self.test_add_data()
            self.test_samples()
            self.test_save()
            self.test_load()
