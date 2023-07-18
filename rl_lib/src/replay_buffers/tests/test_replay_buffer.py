from ..replay_buffer import Replay_Buffer
import os
from shutil import rmtree

class Test_Replay_Buffer:
  def __init__(self, buffer_args):
    self.buffer = Replay_Buffer(**buffer_args)
    self.path = os.getcwd() + '/test_replay_buffer/'
    if not os.path.isdir(self.path):
      os.mkdir(self.path)

  def __exit__(self):
    """Удаляет созданную папку с файлами, если есть"""
    if os.path.isdir(self.path):
      rmtree(directory)
    
  def test_new_init_args(self, buffer_args):
     """Проверяет переинициализацию с новыми аргументами"""
    self.buffer = Replay_Buffer(**buffer_args)
    pass

  def test_add_data(self):
    """Проверяет возможность добавить в буфер данные"""
    pass

  def test_samples(self):
    """Сэмплирует батчи из буфера и проверяет размерности, количество аргументов"""
    pass

  def test_save(self):
    """Выполняет сохранение буфера и проверяет появился ли файл буфера"""
    self.buffer.save(self.path)
    print("Буфер сохранен")
    files = os.listdir(self.path)
    file_names = [f.split('.')[0] for f in files if os.path.isfile(os.path.join(self.path, f))]
    print("Проверка сохранения буфера")
    print(f"Найдено {len(file_names)} файлов: ", *file_names)
    assert self.buffer.name in file_names, "Файл не найден, проверка не пройдена"
    print('Успешно')
    
  def test_load(self):
    """Выполняет test_save, потом загружает и проверяет соответствуют ли загруженные файлы сохраненным"""
    pass

  def test_all_buffers(self, buffers: list):
    for buffer_type in buffers:
      self.test_new_init_args(buffer_type)
      self.test_add_data()
      self.test_samples()
      self.test_save()
      self.test_load()
