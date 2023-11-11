from setuptools import setup, find_packages

import os

def find_yaml_files(root):
    yaml_files = []
    for foldername, subfolders, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.yaml'):
                yaml_files.append(os.path.relpath(os.path.join(foldername, filename), root))
    return yaml_files

if __name__ == '__main__':
    setup(
        name='rl_lib',
        version=os.getenv('PACKAGE_VERSION', '0.2.dev0'),
        # package_dir={'rl_lib': ''},
        packages=find_packages(),
        description='A dev version of the reinforcement learning library.',
        package_data={
        '': find_yaml_files('rl_lib'),
    },
    )