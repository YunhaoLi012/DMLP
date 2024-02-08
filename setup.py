from setuptools import find_packages, setup

setup(
    name='DMLP',
    packages=find_packages(include=['DMLP']),
    version='0.1.0',
    description='Diffusion model learning platform',
    author='Yunhao012',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)