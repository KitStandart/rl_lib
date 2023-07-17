import tensorflow as tf

class Base_Algo(abc.ABC):
  """
  """
  def __init__(self, action_model = object, target_model = object, **kwargs):
      super().__init__(**kwargs)
      self.action_model = action_model
      self.target_model = target_model
  
  @abc.abstractclassmethod
  def initial_model(self) -> None:
    """Инициализирует модель в соответствии с типом алгоритма"""

  @tf.function(reduce_retracing=True,
                jit_compile=True,
                experimental_autograph_options = tf.autograph.experimental.Feature.ALL)
  def sample_action(self, state):
      predict = self.action_model(state)
      while len(predict.shape) > 1:
          predict = tf.squeeze(predict)
      return tf.argmax(predict)

  @tf.function(reduce_retracing=None, jit_compile=None, experimental_autograph_options=None)
  def copy_weights(self,):
      for a_w, t_w in zip(self.action_model.weights, self.target_model.weights):
          new_weights = tf.add(tf.multiply(self.tau, a_w), tf.multiply((1-self.tau), t_w))
          t_w.assign(tf.identity(new_weights))
      return tf.constant(1)

  @tf.function(reduce_retracing=None, jit_compile=None, experimental_autograph_options=None)
  def set_weights(self, target_weights):
      for a_w, t_w in zip(self.action_model.weights, target_weights):
        a_w.assign(tf.identity(t_w))
      return tf.constant(1)
