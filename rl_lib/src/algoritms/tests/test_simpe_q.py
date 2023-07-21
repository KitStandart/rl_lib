from ..algoritms.simple_q import SimpleQ
import os

class Test_SimpleQ:
  def __init__(self, config):
    self.simple_q = SimpleQ(**config)

  def test_save:
    self.simple_q.save()
    real_structure = get_directory_strucrure(self.simple_q.path)
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

def compare_directory_structures(dir_structure1, dir_structure2):
    if dir_structure1.keys() != dir_structure2.keys():
        return False

    for key in dir_structure1.keys():
        if isinstance(dir_structure1[key], dict) and isinstance(dir_structure2[key], dict):
            if not compare_directory_structures(dir_structure1[key], dir_structure2[key]):
                return False
        elif dir_structure1[key] != dir_structure2[key]:
            return False

    return True
  
def get_directory_structure(directory):
    structure = {}
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            structure[item] = get_directory_structure(item_path)
        else:
            structure[item] = None
    return structure
