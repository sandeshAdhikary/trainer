from setuptools import setup, find_packages

setup(name='trainer', 
      version='2.0', 
      packages=find_packages(),
      package_dir={"trainer": "trainer"},
      package_data={
          "trainer": ["*.yaml"],
      },
      scripts=['bin/trainer-app', 'bin/trainer-run'],
      include_package_data=True,
      )