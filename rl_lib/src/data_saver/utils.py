from pickle import dump, load

def save_data(path, data):
  with open(path+'.data', 'wb') as f:
    dump(data, f)

def load_data(path):
  with open(path+'.data', 'rb') as f:
    loaded_data = load(f)
  return loaded_data
