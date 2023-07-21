from ..simple_q import SimpleQ
from rl_lib.rl_lib.src.models.model import Model

import os

class Simple_Model(Model):
  def __init__(self, config = {},**kwargs):
    super().__init__(model_config = config.get('model_config', {}), config = config,  default_config_path=__file__, **kwargs)
  
  def _prediction_processing(self, input_data):
    pass

  def _update_next_state(self, state, action):
    pass

  def initial_state(self):
    pass
    
  @staticmethod
  def create_model(input_shape: tuple, action_space: int):
    pass

  @staticmethod
  def create_model_with_conv(input_shape: tuple, action_space: int):
    pass

class Test_Simple_Q:
  def __init__(self, config):
    action_model = Simple_Model(config, algo_name = "SimpleQ",) #name = "Simple_action_Model_" + config['model_config'].get("name", ""))
    target_model = Simple_Model(config, algo_name = "SimpleQ",) #name = "Simple_target_Model_" + config['model_config'].get("name", ""))
    config.update(action_model.config)
    self.simple_q = SimpleQ(action_model, target_model, config, algo_name = "SimpleQ")

  def test_save(self):
    self.simple_q.save()
    real_structure = get_directory_structure(self.simple_q.path)
    assert self.simple_q.path != self.simple_q.config['data_saver']['path'], "Пути не совпадают"
    correct_structure = {self.simple_q.name: 
                         {
                           self.simple_q.exploration.name + ".data": None, 
                           self.simple_q.buffer.name + ".data": None,
                           self.simple_q.action_model.name + ".h5": None,
                           self.simple_q.target_model.name + ".h5": None,
                         }
                        }
    assert compare_directory_structures(real_structure, correct_structure), "Каталоги разные"

def compare_directory_structures(dir_structure1: dict, dir_structure2: dict) -> bool:
  """Проверяет одинаковые ли структуры каталогов"""
  if dir_structure1.keys() != dir_structure2.keys():
      return False

  for key in dir_structure1.keys():
      if isinstance(dir_structure1[key], dict) and isinstance(dir_structure2[key], dict):
          if not compare_directory_structures(dir_structure1[key], dir_structure2[key]):
              return False
      elif dir_structure1[key] != dir_structure2[key]:
          return False

  return True
  
def get_directory_structure(path: str) -> dict:
    """Получает всю структуру переданного каталога"""
    structure = {}
    for dirpath, dirnames, filenames in os.walk(path):
        current_level = structure
        for dirname in dirpath.split(os.sep):
            current_level = current_level.setdefault(dirname, {})
        for filename in filenames:
            current_level[filename] = None
    return structure
