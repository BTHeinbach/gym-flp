from setuptools import find_packages, setup

setup(name='gym_flp',
      version='0.0.3',
      install_requires=['gym', 'numpy'],
      packages=find_packages(),
      include_package_data=True)
