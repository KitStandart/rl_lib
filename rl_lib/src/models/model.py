import abc

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import clone_model

from ..optimizers.optimizer import get_optimizer
from .base_models import BaseModel, ModelIO, ModelNN


class Model(ModelNN, ModelIO, BaseModel, abc.ABC):
    """Абстрактный класс модели,
    который соединяет все методы классов ModelNN, ModelIO, BaseModel
    """

    def __init__(self, **config: dict):
        super().__init__(**config)
        self.initial_model()

    def _initial_model(self):
        input_shape = self._config['model_config']["input_shape"]
        action_space = self._config['model_config']["action_space"]
        if len(input_shape) == 1:
            return self.create_model(input_shape, action_space)
        else:
            return self.create_model_with_conv(input_shape, action_space)

    def check_input_shape(self, inputs, key=None):
        if not isinstance(inputs, (tf.Tensor, np.ndarray)):
            print(inputs)
            for key, inpt in inputs.items() if isinstance(inputs, dict) else enumerate(inputs):
                inputs[key] = self.check_input_shape(inpt, key=key)
            return inputs
        while len(inputs.shape) < len(self.input_spec(key=key)):
            inputs = tf.expand_dims(inputs, 0)
        if len(inputs.shape) > len(self.input_spec(key=key)):
            assert 0  # inputs.shape не может быть больше входа модели
        return inputs

    def initial_model(self):
        """Инициализирует модель в соответствии с типом алгоритма"""
        if str(self.config['model_config']['model']) == 'None':
            model = self._initial_model()
        else:
            model = clone_model(self.config['model_config']['model'])
        optimizer = self.config.get("optimizer_config")
        optimizer = get_optimizer(**optimizer)
        self.set_new_model(model, optimizer)

    def input_spec(self, key=None):
        if key is not None:
            return self.model.input[key].shape
        elif isinstance(self.model.input, list):
            if self.lstm_size:
                return self.model.input[0].shape
        return self.model.input.shape

    def load(self, path):
        self.model = tf.keras.models.load_model(path+self.name+'.keras')

    def output_spec(self):
        """Возвращает кортеж размера выходных данных Модели"""
        return self.model.layers[-1].output_shape

    def save(self, path):
        self.model.save(path+self.name+'.keras')

    @staticmethod
    @abc.abstractclassmethod
    def create_model(input_shape: tuple,
                     action_space: int) -> tf.keras.Model:
        """Создает модель по умолчанию и возвращает tf.keras.Model,
        архитектура в соответствии с алгоритмом, начальные слои - полносвязные
        """

    @staticmethod
    @abc.abstractclassmethod
    def create_model_with_conv(input_shape: tuple,
                               action_space: int) -> tf.keras.Model:
        """Создает модель по умолчанию  и возвращает tf.keras.Model,
        архитектура в соответствии с алгоритмом, начальные слои - сверточные
        """
