from ..saver import Saver
from rl_lib.rl_lib.src.algoritms.tests.test_simpe_q import get_directory_structure, compare_directory_structures

class Test_Saver:
  def __init__(self, **kwargs):
    self.saver = Saver(**kwargs)

  def test_init(self, path, copy_path):
    self.check_structure(self.saver.path, path)
    self.check_structure(self.saver.copy_path, copy_path)
    print("Тест пройден успешно")


  def check_structure(self, real_path, corrrect_path):
    real_structure = get_directory_structure(real_path)
    assert real_path == corrrect_path, "Пути не совпадают"
    if corrrect_path != "":
      correct_structure = {corrrect_path.replace("/", ""): {"":{}, self.saver.name: {}}}
      assert compare_directory_structures(real_structure, correct_structure), "Каталоги разные"
