from setuptools import setup
from setuptools import find_packages


setup(name='tester-from-hell',
      version='0.1.0',
      description='Framework for testing self-adaptive software systems',
      author='André Reichstaller',
      author_email='reichstaller@isse.de',
      install_requires=[
	  'keras>=1.0.7',
	  'keras_rl==0.3.1',
          'h5py',
          'action-provider-rl',
	  ],
      dependency_links=[
        "git+https://github.com/dremonaut/action-provider-rl.git#egg=action-provider-rl-0.2.0"
      ]
      )