from setuptools import setup, find_packages

setup(name='trainer', 
      version='1.0', 
      packages=find_packages(),
      package_data={'trainer': ['trainer/.config.yaml']}
      )