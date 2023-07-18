from .replay_buffers.replay_buffer import Replay_Buffer
import os

class Test_Replay_Buffer:
  def __init__(self, buffer_args):
    self.buffer = Replay_Buffer(**buffer_args)
    self.path = os.getcwd() + '/test_replay_buffer/'
    if not os.path.isdir(self.path):
      os.mkdir(self.path)
    
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
