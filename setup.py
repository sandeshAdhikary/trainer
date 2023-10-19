from setuptools import setup, find_packages

setup(name='trainer', 
      version='1.0', 
      packages=find_packages(),
      package_dir={"trainer": ""},
      include_package_data=True,
      package_data={
          "": ["*.yaml"],
      }
      )