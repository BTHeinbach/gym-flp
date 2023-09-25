from setuptools import find_packages, setup

setup(name='gym_flp',
      version='0.2.1',
      description='Implementation of facility layout problems in OpenAI Gym',
      url='https://github.com/BTHeinbach/gym-flp',
      author='Benjamin Heinbach',
      author_email='benjamin.heinbach@uni-siegen.de',
      license='MIT',
      install_requires=['gym<0.26', 'numpy', 'anytree', 'pygame', 'Pillow'],
      packages=find_packages(),
      package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.dat', '*.sln', '*.prn', '*.pkl', '*.json'],})
