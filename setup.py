from setuptools import setup
setup(
  name='optimus_ml',
  packages=['optimus_ml'],
  version='0.1',
  description='Automated machine learning tool',
  author='Jeroen van Hoof',
  author_email='jeroen@jeroenvanhoof.nl',
  url='https://github.com/Yatoom/Optimus',
  download_url='',
  keywords=['machine learning', 'automated machine learning'],
  classifiers=[],
  install_requires=[
    "scipy", "numpy", "pandas", "matplotlib", "tqdm", "sklearn", "pynisher"
  ]
)