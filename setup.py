from setuptools import setup, find_packages

setup(name='trainer', 
      version='1.0', 
      packages=find_packages(),
      package_dir={"trainer": "trainer"},
      package_data={
          "trainer": ["*.yaml"],
      },
      scripts=['bin/app'],
      include_package_data=True,
      )