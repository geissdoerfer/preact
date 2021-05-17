# PREACT simulator

This repo contains the code accompanying our [IPSN'19 paper](https://wwwpub.zih.tu-dresden.de/~mzimmerl/pubs/geissdoerfer19preact.pdf) titled *Getting more out of energy-harvesting systems: energy management under time-varying utility with PreAct*.
It provides a Python implementation of PreAct as well as the comparison algorithms LT-ENO, ENO-MAX and others.
There are three key components: The energy prediction algorithms in `enmanage/prediction.py` are used to predict harvested energy in the future.
The energy management algorithms in `enmanage/managers.py` use these predictions and the current state of the system to calculate a varying energy budget according to their individual objective.
The simulator in `enmanage/__init__.py` can be used to conveniently evaluate the performance of the various prediction methods and energy management algorithms for given energy traces, capacity sizes, etc..


## Installation

Install with

```
python setup.py install
```


## Examples

There are two examples provided in the `examples` directory: `examples/capacity.py` runs PREACT on an example trace for two different battery capacities. `examples/utility.py` runs PREACT on an example trace for two different, time-varying utility functions.
