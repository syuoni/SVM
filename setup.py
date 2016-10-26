from setuptools import setup

setup(name             = 'SVM',
      version          = '0.3',
      description      = 'A implementation of Support Vector Machine (SVM) with Python',
      long_description = open('README.rst').read(),
      url              = 'https://github.com/syuoni/SVM',
      author           = 'syuoni',
      author_email     = 'spiritas@163.com',
      license          = 'MIT',
      packages         = ['SVM'],
      install_requires = ['numpy',
                          'pandas',
                          'scipy',
                          'matplotlib',
                          'seaborn'],
      zip_safe         = False)