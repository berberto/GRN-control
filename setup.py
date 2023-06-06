from setuptools import setup

setup(name='grn_control',
      version='0.0.1',
      install_requires=['gym<=0.22.0', 'torch', 'tqdm'],
      py_modules=['grn_control','algorithms']
)