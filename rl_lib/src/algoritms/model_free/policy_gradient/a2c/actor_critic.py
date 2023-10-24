import tensorflow as tf
from tensorflow.keras import layers

from rl_lib.src.algoritms.model_free.value_based import DQN_Model


class Actor_Model(DQN_Model):
    def __init__(self, config={}, **kwargs):
        config['model_config'] = config['actor_model_config']['model_config']
        config['optimizer_config'] = config['actor_optimizer_config']['optimizer_config']
        super().__init__(config=config, **kwargs)
        self.name = kwargs.get('name', 'error_name') + '_actor_'

    def _prediction_processing(self, inputs: tf.Tensor, **kwargs):
        return kwargs['critic_model']([kwargs['state'], inputs])

    def loss(self, target: tf.Tensor, predict: tf.Tensor) -> tf.Tensor:
        """Вычисляет и возвращает потери в соответствии с функцией потерь"""
        return tf.reduce_mean(predict, axis=0) * (-1)


class Critic_Model(DQN_Model):
    def __init__(self, config={}, **kwargs):
        config['model_config'] = config['critic_model_config']['model_config']
        config['optimizer_config'] = config['critic_optimizer_config']['optimizer_config']
        super().__init__(config=config, **kwargs)
        self.name = kwargs.get('name', 'error_name') + '_critic_'

    def _prediction_processing(self, inputs: tf.Tensor, **kwargs):
        return inputs

    @tf.function(reduce_retracing=True,
                 jit_compile=False,
                 experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def calculate_gradients(self, **kwargs) -> dict:
        """
        Вычисляет градиенты, лосс, td-ошибку

        Kwargs:
            dict содержащий батч, таргет, маску, опционально приоритетные веса

        Returns:
            dict содержащий лоссы и td-ошибку
        """
        with tf.GradientTape(persistent=False) as tape:
            Q = self.model([kwargs['state'], kwargs['action']], training=True)
            Q = self.prediction_processing(Q, **kwargs)
            if len(Q.shape) != len(kwargs['Qtarget'].shape):
                Q = tf.expand_dims(Q, -1)

            td_error = kwargs['Qtarget'] - Q
            loss = self.loss(kwargs['Qtarget'], Q)*kwargs.get('weights', 1.0)
            E_loss = tf.reduce_mean(loss, axis=0)
        gradients = tape.gradient(E_loss, self.model.trainable_variables)
        loss = tf.reduce_mean(loss, axis=-1)
        return {'gradients': gradients, 'loss': loss, 'td_error': td_error}

    @staticmethod
    def create_model(input_shape: tuple, action_space: int) -> tf.keras.Model:
        """Создает модель tf.keras.Model, архитектура DQN"""
        input_layer = layers.Input(shape=input_shape, )
        action_layer = layers.Input(shape=action_space, )
        concat = layers.Concatenate()((input_layer, action_layer))
        flatten = layers.Flatten()(concat)
        dence_layer1 = layers.Dense(256, activation='relu')(flatten)
        dence_layer2 = layers.Dense(256, activation='relu')(dence_layer1)
        dence_out = layers.Dense(action_space, activation=None)(dence_layer2)

        return tf.keras.Model(
            inputs=[input_layer, action_layer],
            outputs=dence_out
            )

    @staticmethod
    def create_model_with_conv(input_shape: tuple,
                               action_space: int) -> tf.keras.Model:
        """Создает модель tf.keras.Model, архитектура DQN,
        начальные слои - сверточные"""
        input_layer = layers.Input(shape=input_shape, )
        action_layer = layers.Input(shape=action_space, )
        cov_layer1 = layers.Conv2D(32, 7, activation='relu')(input_layer)
        cov_layer2 = layers.Conv2D(64, 5, activation='relu')(cov_layer1)
        cov_layer3 = layers.Conv2D(64, 3, activation='relu')(cov_layer2)
        conv_out = layers.Flatten()(cov_layer3)

        concat = layers.Concatenate()((conv_out, action_layer))
        flatten = layers.Flatten()(concat)
        dence_layer1 = layers.Dense(256, activation='relu')(flatten)
        dence_layer2 = layers.Dense(256, activation='relu')(dence_layer1)
        dence_out = layers.Dense(action_space, activation=None)(dence_layer2)

        return tf.keras.Model(
            inputs=[input_layer, action_layer],
            outputs=dence_out
            )


class Actor_Critic_Model(DQN_Model):
    def __init__(self, config={}, **kwargs):
        config['actor_model_config']['model_config']['name'] = config['model_config']['name']
        config['actor_model_config']['model_config']['input_shape'] = config['model_config']['input_shape']
        config['actor_model_config']['model_config']['action_space'] = config['model_config']['action_space']

        config['critic_model_config']['model_config']['name'] = config['model_config']['name']
        config['critic_model_config']['model_config']['input_shape'] = config['model_config']['input_shape']
        config['critic_model_config']['model_config']['action_space'] = config['model_config']['action_space']
        self.actor_model = Actor_Model(config=config, **kwargs)
        self.critic_model = Critic_Model(config=config, **kwargs)

    def __call__(self, input: tf.Tensor) -> tf.Tensor:
        return self.critic_model([input, self.actor_model(input)])

    def update_weights(self, **kwargs):
        _ = self.update_weights_actor(**kwargs)
        return self.update_weights_critic(**kwargs)

    def update_weights_actor(self, **kwargs):
        kwargs['critic_model'] = self.critic_model.model
        loss = self.actor_model.update_weights(**kwargs)
        return {'loss': loss['loss'], 'td_error': loss['td_error']}

    def update_weights_critic(self, **kwargs) -> dict:
        loss = self.critic_model.update_weights(**kwargs)
        return {'loss': loss['loss'], 'td_error': loss['td_error']}

    def calculate_gradients(self, **kwargs) -> dict:
        kwargs['action'] = self.actor_model(kwargs['next_state'])
        gradients = self.critic_model.calculate_gradients(**kwargs)
        return gradients

    def get_weights(self, ) -> dict:
        return {
            'actor': self.actor_model.get_weights(),
            'critic': self.critic_model.get_weights()
        }

    def input_spec(self, key=None):
        return self.actor_model.input_spec(key=key)

    def load(self, path):
        self.actor_model.load(path)
        self.critic_model.load(path)

    def save(self, path):
        self.actor_model.save(path)
        self.critic_model.save(path)

    def set_weights(self, weights: dict) -> None:
        self.actor_model.set_weights(weights=weights['actor'])
        self.critic_model.set_weights(weights=weights['critic'])

    @property
    def summary(self):
        self.actor_model.summary
        self.critic_model.summary
