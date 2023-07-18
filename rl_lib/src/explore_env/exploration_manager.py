from .epsilon_greedy import Epsilon_Greedy
from .base_algo import Base_Algo

class ExplorationManager(Base_Explore):
  """Выбирает стратегию исследования и выполняет все ее функции
  Kwargs:
    strategy_name: str, Название стратегии 
    strategy_config: dict, Параметры стратегии
  """
  def __init__(self, strategy_name="epsilon_greedy", strategy_config = {}):
    self.strategy_name = strategy_name
    if strategy_name.lower() == "epsilon_greedy":
      self.strategy = Epsilon_Greedy(strategy_config)
    if strategy_name.lower() == "soft_q":
      pass
      
  def reset(self, ):
    self.strategy.reset()
  
  def save(self, path):
    self.strategy.save(path+self.strategy_name)
  
  def load(self, path):
    self.strategy.load(path+self.strategy_name)
  
  def __call__(self, action):
      return self.strategy(action)
  
  def test(self, action):
      return self.strategy.test(action)
