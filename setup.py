from setuptools import setup
setup(name='TT-IGA',
version='1.0',
description='Tensor-Train decomposition in pytorch',
url='https://github.com/ion-g-ion/code-paper-tt-iga',
author='Ion Gabriel Ion',
author_email='ion.ion.gabriel@gmail.com',
license='MIT',
packages=['tt_iga'],
install_requires=['numpy>=1.18','torch>=1.7','opt_einsum','torchtt'],
test_suite='tests',
zip_safe=False) 