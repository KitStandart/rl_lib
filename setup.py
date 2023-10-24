from setuptools import setup, find_packages

import os

if __name__ == '__main__':
    setup(
        name='rl_lib',
        version=os.getenv('PACKAGE_VERSION', '0.1.dev0'),
        # package_dir={'rl_lib': ''},
        packages=find_packages(),
        description='A dev version of the reinforcement learning library.',
    )