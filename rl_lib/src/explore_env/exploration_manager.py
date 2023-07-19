from .epsilon_greedy import Epsilon_Greedy
from .soft_q import Soft_Q
from .base_explore import Base_Explore

class ExplorationManager(Base_Explore):
  """Выбирает стратегию исследования и выполняет все ее функции
  Kwargs:
    strategy_name: str, Название стратегии 
    strategy_config: dict, Параметры стратегии
  """
  def __init__(self, strategy_name="epsilon_greedy", strategy_config = {}):
    self.config = {strategy_name="epsilon_greedy", strategy_config = {}}
    if strategy_name.lower() == "epsilon_greedy":
      self.strategy = Epsilon_Greedy(strategy_config)
    if strategy_name.lower() == "soft_q":
      self.strategy = Soft_Q(strategy_config)
    self.strategy_name = self.strategy.name
      
  def reset(self, ):
    self.strategy.reset()
  
  def save(self, path):
    self.strategy.save(path)
  
  def load(self, path):
    self.strategy.load(path)
  
  def __call__(self, Q):
      return self.strategy(Q)
  
  def test(self, Q):
      return self.strategy.test(Q)
