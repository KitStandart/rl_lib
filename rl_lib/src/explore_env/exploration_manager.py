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
    self._config = {"strategy_name": strategy_name, "strategy_config": strategy_config}
    if strategy_name.lower() == "epsilon_greedy":
      self.strategy = Epsilon_Greedy(**strategy_config)
    if strategy_name.lower() == "soft_q":
      self.strategy = Soft_Q(**strategy_config)
    self.strategy_name = self.strategy.name
    
  def __call__(self, Q):
      return self.strategy(Q)
    
  @property
  def config(self):
    return self.config
    
  @property
  def name(self):
    return self.strategy.name  
    
  def load(self, path):
    self.strategy.load(path)    
    
  def reset(self, ):
    self.strategy.reset()
  
  def save(self, path):
    self.strategy.save(path)
  
  def test(self, Q):
      return self.strategy.test(Q)
