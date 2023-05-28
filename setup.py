from setuptools import setup, find_packages

setup(
    name='sleep',
    version='0.1',
    packages=find_packages(include=['sleep', 'sleep.*']),
    url='https://github.com/GergelyTuri/sleep',
    license='MIT',
    author='Gergely Turi',
    author_email='gt2253@cumc.columbia.edu',
    description='some basic analysis code'
)
