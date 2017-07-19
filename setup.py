from setuptools import setup
setup(
  name='optimus_ml',
  packages=['optimus_ml', 'optimus_ml.extra', 'optimus_ml.optimizer', 'optimus_ml.vault', 'optimus_ml.transcoder'],
  version='0.2',
  description='Automated machine learning tool',
  author='Jeroen van Hoof',
  author_email='jeroen@jeroenvanhoof.nl',
  url='https://github.com/Yatoom/Optimus',
  download_url='https://github.com/Yatoom/Optimus/archive/0.1.tar.gz',
  keywords=['machine learning', 'automated machine learning'],
  classifiers=[],
  install_requires=[
    "scipy", "numpy", "pandas", "matplotlib", "tqdm", "sklearn", "pynisher", "pymongo"
  ]
)