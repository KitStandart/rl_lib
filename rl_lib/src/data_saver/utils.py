from pickle import dump, load
from yaml import safe_load
import os.path as os_path

def save_data(path, data):
  with open(path+'.data', 'wb') as f:
    dump(data, f)

def load_data(path):
  with open(path+'.data', 'rb') as f:
    loaded_data = load(f)
  return loaded_data

def load_default_config(path):
  return safe_load(open(os_path.join(os_path.dirname(path),"./config.yaml"), "rb"))
