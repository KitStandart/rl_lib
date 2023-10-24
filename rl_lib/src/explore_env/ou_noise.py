import numpy as np
from tensorflow import clip_by_value

from ..data_saver.utils import load_data, save_data
from .base_explore import Base_Explore


class OU_Noise_generator:
    def __init__(self, mean, sigma, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """Formula taken from
        https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        """
        dx = (self.theta * (self.mean - self.x_prev) * self.dt + self.sigma *
              np.sqrt(self.dt) * np.random.normal(size=self.mean.shape,
                                                  scale=self.sigma))
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev += dx
        return self.x_prev

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class OU_Noise(Base_Explore):
    """Шум Орнштейна — Уленбека стратегия исследования,
    применяется к предсказанным непрерывным действиям.

    Kwargs:
        action_spase: int, Размер пространтства действий
        alpha: int, Количество внутренних шагов исследований
                    до установки минимального эпсилон
        axis: int, ось вычислений
        sigma: float, Максимальный эпсилон
    """

    def __init__(self, action_space=None,
                 axis=-1, alpha=0.9, dt=0.01,
                 lower_bound=-1.0, mean: np.ndarray = None,
                 sigma=1.0, theta=0.15,
                 upper_bound=1.0,
                 **kwargs):
        self.action_space = action_space
        self.alpha = alpha
        self.axis = axis
        self.ou_gen = OU_Noise_generator(np.zeros(
            action_space) if mean == "None" else mean, sigma, theta=theta,
                                                    dt=dt, x_initial=None)
        self.eps = self.ou_gen()
        self.lower_bound = lower_bound
        self.sigma = sigma
        self._name = "ou_noise"
        self.upper_bound = upper_bound

    def __call__(self, action):
        action += self.eps
        self.eps = self.alpha*self.eps + self.ou_gen()
        return clip_by_value(action,
                             clip_value_min=self.lower_bound,
                             clip_value_max=self.upper_bound)

    def load(self, path):
        self.__dict__ = load_data(path+self.name)

    @property
    def name(self):
        return self._name

    def reset(self, ):
        self.eps = self.ou_gen()

    def save(self, path):
        save_data(path+self.name, self.__dict__)

    def test(self, action):
        return clip_by_value(action,
                             clip_value_min=self.lower_bound,
                             clip_value_max=self.upper_bound)
