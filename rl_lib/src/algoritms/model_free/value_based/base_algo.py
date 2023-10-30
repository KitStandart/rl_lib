import abc
from copy import copy
from typing import Union

import tensorflow as tf

from rl_lib.src.data_saver.saver import Saver

from rl_lib.src.data_saver.utils import load_default_config
from .utils import update_config


class Base_Algo(Saver, abc.ABC):
    """Базовый абстрактный класс алгоритма.
    Хранит все методы, необходимые для вычислений в каком либо алгоритме.
    """

    def __init__(self, action_model: object,
                 target_model: object,
                 config: dict,
                 default_config_path: str,
                 *args, **kwargs):
        self._config = load_default_config(default_config_path)
        update_config(self._config, config)

        self.action_model = action_model(
            config=copy(self._config),
            algo_name=kwargs.get("algo_name", "unkown"),
            name=(kwargs.get("name", "unkown_name") +
                  "_action_" +
                  config.get("model_config", {}).get("name", ""))
            )
        self.target_model = target_model(
            config=copy(self._config),
            algo_name=kwargs.get("algo_name", "unkown"),
            name=(kwargs.get("name", "unkown_name") +
                  "_target_" +
                  config.get("model_config", {}).get("name", ""))
            )
        super().__init__(**self.config.get('data_saver', {}), **kwargs)
        self.target_model.set_weights(self.action_model.get_weights())

    @property
    def config(self):
        return self._config

    @abc.abstractclassmethod
    def calculate_new_best_action(self) -> tf.Tensor:
        """Вычислеят новое лучшее действие для получения таргета"""

    @abc.abstractclassmethod
    def calculate_target(self) -> dict:
        """Вычисляет таргет для обучения"""

    @abc.abstractclassmethod
    def get_action(self, observation) -> float:
        """Возвращает действие на основе наблюдения с учетом исследования"""

    @abc.abstractclassmethod
    def get_test_action(self, observation) -> float:
        """Возвращает действие на основе наблюдения без исследования"""

    @abc.abstractclassmethod
    def get_gradients(self) -> tf.Tensor:
        """Вычисляет градиенты и возвращает их"""

    @abc.abstractclassmethod
    def load(self, path) -> None:
        """Загружает алгоритм"""

    @abc.abstractclassmethod
    def reset(self) -> None:
        """Сбрасывает внутренние данные модели"""

    @abc.abstractclassmethod
    def _train_step(self) -> dict:
        """Вспомогательная train_step"""

    @abc.abstractclassmethod
    def train_step(self) -> dict:
        """Вычисляет полный обучающий шаг"""

    @abc.abstractclassmethod
    def save(self, path) -> None:
        """Сохраняет алгоритм"""

    @abc.abstractclassmethod
    def summary(self) -> None:
        """Выводит архитектуру модели"""

    @tf.function(reduce_retracing=None,
                 jit_compile=None,
                 experimental_autograph_options=None)
    def _copy_weights(self, action_model_weights: list,
                      target_model_weights: list,
                      tau: float) -> tf.constant:
        """Копирует веса из модели действия в целевую модель"""
        for a_w, t_w in zip(action_model_weights, target_model_weights):
            new_weights = tf.add(tf.multiply(tau, a_w),
                                 tf.multiply((1-tau), t_w))
            t_w.assign(tf.identity(new_weights))
        return tf.constant(1)

    def copy_weights(self) -> tf.constant:
        """Копирует веса из модели действия в целевую модель"""
        res = self._copy_weights(
            self.action_model.weights, self.target_model.weights, self.tau)
        return res

    def _expand_dims_like(self, tensor: tf.Tensor,
                          tensor_like: tf.Tensor) -> tf.Tensor:
        len_tensor_like_shape = len(tensor_like.shape)
        
        while len(tensor.shape) < len_tensor_like_shape:
            tensor = tf.expand_dims(tensor, axis=-1)
        return tensor
    
    @tf.function(reduce_retracing=True,
                 jit_compile=True,
                 experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def sample_action(self,
                      state: Union[tf.Tensor, tuple]
                      ) -> Union[tf.Tensor, list]:
        """Возвращает предсказания модели на основе текущих наблюдений"""
        predict = self.action_model(state)
        if isinstance(predict, list):
            return self.squeeze_predict(predict[0]), predict[1], predict[2]
        return self.squeeze_predict(predict)

    @tf.function(reduce_retracing=None,
                 jit_compile=None,
                 experimental_autograph_options=None)
    def set_weights(self, target_weights: list) -> tf.constant:
        """Устанавливает переданные как аргумент веса в основную сеть"""
        for a_w, t_w in zip(self.action_model.weights, target_weights):
            a_w.assign(tf.identity(t_w))
        return tf.constant(1)

    @staticmethod
    def squeeze_predict(predict) -> tf.Tensor:
        """Удаляет единичные измерения из предсказаний"""
        while len(predict.shape) >= 1 and predict.shape[0] == 1:
            predict = tf.squeeze(predict, axis=0)
        return predict
