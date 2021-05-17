from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="enmanage",
    version="0.1.0",
    description="Energy Management on Solar Harvesting Sensor nodes",
    long_description=readme,
    author="Kai Geissdoerfer",
    author_email="kai.geissdoerfer@tu-dresden.de",
    license=license,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,
    install_requires=["numpy", "scipy", "pyyaml", "matplotlib"],
    packages=find_packages(exclude=("tests", "docs", "scripts", "data")),
)
