import numpy as np 
from saving_data.utils import save_data, load_data

class Epsilon_Greedy:
    '''Epsilon greedy explore strategy'''
    def __init__(self, **kwargs):
        self.eps_desay_steps = kwargs.get("eps_decay_steps", 1e6)
        self.eps_min = kwargs.get("eps_min", 1e-1)
        self.eps_max = kwargs.get("eps_max", 1.)
        self.eps_test = kwargs.get("eps_test", 1e-3)
        self.action_space = kwargs.get("action_space", 2)
        self.reset()

    def reset(self, ):
        self.count = 0
        self.eps = self.eps_max
    
    def save(self, path):
        save_data(path ,{
                'count': self.count,
                    })

    def load(self, path):
        data = load_data(path)
        self.count = data['count']

    def __call__(self, action):
        self.eps = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * self.count/self.eps_desay_steps)
        self.count += 1
        return self.get_action(self.eps, action)

    def test(self, action):
        return self.get_action(self.eps_test, action)

    def get_action(self, eps, action):
        if np.random.random() < eps: return  np.random.randint(self.action_space)
        else: return action