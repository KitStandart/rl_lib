import os

class Saver:
  """Хранит в себе пути сохранения этапа обучения и путь резервного копирования.
  При инициализации создает папки для сохранения и резервного копирования.
  Args:
    name: str. Необязательно, название алгоритма
    path: str. Путь сохранения
    copy_path: str. Путь резервного копирования
  """
  def __init__(self, name="unkown", path="", copy_path="", **kwargs):
    super().__init__(**kwargs)
    self.original_path = os.getcwd()
    self.name = name
    self.copy_path = copy_path
    self.path = path

    self.validate_path()
    
    self.init_save_dir()
    self.init_copy_dir()

  def init_save_dir(self):
    if not os.path.isdir(self.original_path+"/models/"):
        os.mkdir(self.original_path+"/models/")
    self.path = self.original_path + "/models/" + self.name + '/'
    # if not os.path.isdir(self.path):
    #     os.mkdir(self.path)  
    self.save_path = self.path + self.name

  def init_copy_dir(self):
    if self.copy_path != "" and not os.path.isdir(self.copy_path+"/models/"):
        os.mkdir(self.copy_path+"/models/")
        
    self.copy_path = self.copy_path +"/models/" + self.name +'/'
    if self.copy_path != "/models/" + self.name +'/' and not os.path.isdir(self.copy_path):
        os.mkdir(self.copy_path)

  def validate_path(self):
    assert type(self.path) == str, "Неверный тип аргумента"
    assert type(self.copy_path) == str, "Неверный тип аргумента"
    assert type(self.name) == str, "Неверный тип аргумента"

  @property
  def get_save_path(self):
    return self.path
  
  @property
  def get_copy_path(self):
    if self.copy_path != "/models/" + self.name +'/': return self.copy_path 
    else: return "Path is not defined"
