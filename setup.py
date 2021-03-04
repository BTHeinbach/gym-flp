from setuptools import find_packages, setup

setup(name='gym_flp',
      version='0.1.0',
      description='Implementation of facility layout problems in OpenAI Gym',
      url='https://github.com/BTHeinbach/gym-flp',
      author='Benjamin Heinbach',
      author_email='benjamin.heinbach@uni-siegen.de',
      license='MIT',
      install_requires=['gym', 'numpy'],
      packages=find_packages(),
      include_package_data=True)