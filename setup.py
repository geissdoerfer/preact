from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='enmanage',
    version='0.1.0',
    description='Sample package for Python-Guide.org',
    long_description=readme,
    author='Kai Geissdoerfer',
    author_email='kai.geissdoerfer@tu-dresden.de',
    license=license,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
       'numpy',
       'scipy'
    ],
    packages=find_packages(exclude=('tests', 'docs', 'scripts', 'data'))
)
