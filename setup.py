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
    include_package_data=True,
    install_requires=["numpy", "scipy", "pyyaml", "matplotlib", "pandas", "parse"],
    packages=["enmanage"],
    package_dir={"enmanage": "enmanage"},  # the one line where all the magic happens
    package_data={
        "enmanage": ["data/*.txt", "config/default.yml"],
    },
)
