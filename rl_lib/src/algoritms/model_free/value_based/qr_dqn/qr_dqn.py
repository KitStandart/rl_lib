import tensorflow as tf
from tensorflow.keras import layers

from rl_lib.src.models.model import Model
from ..simple_q import SimpleQ

class QR_DQN_Model(Model):
    def __init__(self, config={}, **kwargs):
        super().__init__(
            model_config=config.get('model_config', {}),
            config=config,
            **kwargs)
        self.num_atoms = config['model_config'].get("num_atoms", 200)
        self.implicit_tau = tf.reshape([i/self.num_atoms for i in range(1, self.num_atoms+1)], (1, 1, self.num_atoms))
        
        self.hubber_k = config['model_config'].get("hubber_k", 1.0)
    
    def _prediction_processing(self, inputs: tf.Tensor, **kwargs):
        mask = self.make_mask(tf.cast(kwargs['action'], dtype=tf.int32))
        if len(inputs.shape) != len(mask.shape):
            mask = tf.expand_dims(mask, -1)
        return tf.reduce_sum(
            tf.multiply(inputs, mask),
            axis=kwargs['batch_dims']
            )

    def loss(self, target: tf.Tensor, predict: tf.Tensor) -> tf.Tensor:
        """Вычисляет и возвращает потери в соответствии с функцией потерь"""
        error = target - predict
        huber_loss = self.huber_loss_func(error, k=self.hubber_k)
        quantill_loss = tf.abs(self.implicit_tau - tf.cast(error < 0, dtype = tf.float32)) * (huber_loss) #/k IQN loss при k=0 бесконечность...
        quantill_loss = tf.reduce_mean(quantill_loss, axis = -1)  #N' sum and 1/N'
        quantill_loss = tf.reduce_sum(quantill_loss, -1) #N sum
        return quantill_loss

    def huber_loss_func(self, error, k=1.0):
        return tf.where(tf.abs(error) <= k, 0.5 * tf.square(error), k * (tf.abs(error) - 0.5 * k))
    
    def make_mask(self, action) -> tf.Tensor:
        """Создает маску по действиям """
        return tf.one_hot(action, self.output_spec()[-2])

    def _update_next_state(self, state, action):
        pass

    def initial_state(self):
        pass

    @staticmethod
    def create_model(input_shape: tuple, action_space: int,
                     quantile_dim: int = 200) -> tf.keras.Model:
        """Создает модель tf.keras.Model, архитектура DQN"""
        input_layer = layers.Input(shape=input_shape, )
        dence_layer1 = layers.Dense(256, activation='relu')(input_layer)
        dence_layer2 = layers.Dense(256, activation='relu')(dence_layer1)
        dence_out = layers.Dense(action_space * quantile_dim,
                                 activation=None)(dence_layer2)

        out = layers.Reshape((action_space, quantile_dim))(dence_out)
        return tf.keras.Model(inputs=input_layer, outputs=out)

    @staticmethod
    def create_model_with_conv(input_shape: tuple,
                               action_space: int,
                               quantile_dim: int = 200) -> tf.keras.Model:
        """Создает модель tf.keras.Model, архитектура DQN,
        начальные слои - сверточные"""
        input_layer = layers.Input(shape=input_shape, )
        cov_layer1 = layers.Conv2D(32, 7, activation='relu')(input_layer)
        cov_layer2 = layers.Conv2D(64, 5, activation='relu')(cov_layer1)
        cov_layer3 = layers.Conv2D(64, 3, activation='relu')(cov_layer2)
        conv_out = layers.Flatten()(cov_layer3)

        dence_layer1 = layers.Dense(256, activation='relu')(conv_out)
        dence_layer2 = layers.Dense(256, activation='relu')(dence_layer1)
        dence_out = layers.Dense(action_space * quantile_dim,
                                 activation=None)(dence_layer2)

        out = layers.Reshape((action_space, quantile_dim))(dence_out)
        return tf.keras.Model(inputs=input_layer, outputs=out)


class QR_DQN(SimpleQ):
    def __init__(self, config):
        super().__init__(QR_DQN_Model, QR_DQN_Model, config,
                         default_config_path=__file__,
                         algo_name="QR_DQN",
                         name=("QR_DQN_Model_" +
                               config.get('model_config', '').get('name', ''))
                         )
        self.BATCH_DIMS = 1
        self.IND_AXIS = 1
        self.MEAN_AXIS = 2

    def _prediction_processing(self, input_data):
        pass

    def _update_next_state(self, state, action):
        pass

    def initial_state(self):
        pass

    def get_best_action(self, Z_action: tf.Tensor, Z_target: tf.Tensor):
        q = tf.reduce_mean(Z_action, axis=self.MEAN_AXIS)
        ind = tf.expand_dims(tf.argmax(q, axis=self.IND_AXIS),-1)
        Z_target = tf.gather(Z_target, ind, batch_dims=self.BATCH_DIMS)
        return Z_target

    def _train_step(self, **batch) -> dict:
        result = super()._train_step(**batch)
        result['td_error'] = tf.reduce_mean(
            tf.reduce_mean(
                result['td_error'], -1
            ), -1
        )
        return result

    def _get_action(self, observation: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(
            super()._get_action(observation),
            axis=-1
            ) 