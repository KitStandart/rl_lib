from .replay_buffers.replay_buffer import Replay_Buffer

class Test_Replay_Buffer:
  def __init__(self):
    self.buffer = Replay_Buffer

  def test_add_args():
    pass

  def test_add_data():
    pass

  def test_samples():
    pass

  def test_save():
    pass

  def test_load():
    pass

  def test_all_buffers(buffers):
    for buffer_type in buffers:
      self.test_add_args(buffer_type)
      self.test_add_data()
      self.test_samples()
      self.test_save()
      self.test_load()
