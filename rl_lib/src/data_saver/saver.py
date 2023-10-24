import os
from shutil import copy, make_archive


class Saver:
    """Хранит в себе пути сохранения этапа обучения
      и путь резервного копирования.
    При инициализации создает папки для сохранения и резервного копирования.
    Args:
      name: str. Необязательно, название алгоритма
      path: str. Путь сохранения
      copy_path: str. Путь резервного копирования
    """

    def __init__(self, algo_name="None", copy_path="", name="", path="",
                 **kwargs):
        self.algo_name = algo_name
        self.copy_path = copy_path
        self.name = name
        self.original_path = os.getcwd()
        self.path = path

        self.validate_path()

        self.init_copy_dir()
        self.init_save_dir()

    @property
    def get_save_path(self):
        return self.path

    @property
    def get_copy_path(self):
        if self.copy_path != "":
            return self.copy_path
        else:
            return "Path is not defined"

    def init_copy_dir(self):
        if self.copy_path != "":
            self.copy_path = self.copy_path + self.algo_name + "/"
            if not os.path.isdir(self.copy_path):
                os.makedirs(self.copy_path)

    def init_save_dir(self):
        """Создает путь сохранения и директорию сохранения"""
        if self.path == "":
            self.path = self.original_path + "/models/" + \
                self.algo_name + "/" + self.name + "/"
        else:
            self.path = self.path + self.name + "/"
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def make_copy(self):
        """Резервное копирование архива директории"""
        copy(self.path + '/' + self.name+'.zip', self.copy_path)

    def make_archive(self):
        """Создает архив директории"""
        make_archive(base_name=self.name, format='zip', root_dir=self.path)

    def validate_path(self):
        assert isinstance(
            self.algo_name, str), "Неверный тип аргумента, должно быть str"
        assert isinstance(
            self.copy_path, str), "Неверный тип аргумента, должно быть str"
        assert isinstance(
            self.name, str), "Неверный тип аргумента, должно быть str"
        assert isinstance(
            self.path, str), "Неверный тип аргумента, должно быть str"
        if len(self.path) > 0:
            assert self.path[-1] == "/", "В конце пути должен быть /"
        if len(self.copy_path) > 0:
            assert self.copy_path[-1] == "/", "В конце пути должен быть /"
