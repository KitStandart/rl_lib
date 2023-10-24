import os.path as os_path
from pickle import dump, load

from yaml import safe_load


def save_data(path, data):
    with open(path+'.data', 'wb') as f:
        dump(data, f)


def load_data(path):
    with open(path+'.data', 'rb') as f:
        loaded_data = load(f)
    return loaded_data


def load_default_config(path):
    file_name = "./config.yaml"
    if path.split('/')[-1].split('.')[-1] in 'yaml':
        file_name = path.split('/')[-1]
    return safe_load(
        open(
            os_path.join(
                os_path.dirname(path),
                file_name
                ), "rb"
            )
        )