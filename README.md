# Supporting python modules for X

Python package for the replication package of the paper.
It includes a module that implements the analytical model described in the paper.
It also includes a module for creating plots of this model, as well as plots from finite element calculations output from a related package ([name]()) and plots from Monte Carlo simulations from another related package ([name]()).

Some example scripts for creating plots are provided in the [`scripts`](scripts/) directory.
More examples can be found in the data [component of the replication package]().

## Installation

This package was developed and used on Linux.
[It is available on the PyPi respository]().
It can be installed by running
```
pip install X
```
If you do not have root access and it does not automatically default to installing locally, the `--user` flag may be used.
To install directly from this repository, run
```
python -m build
pip install dist/actinrings-0.0.0-py3-none-any.whl
```
To run the above, it may be necessary to update a few packages:
```
python3 -m pip install --upgrade pip setuptools wheel
```

For more information on building and installing python packages, see the documentation from the [Python Packaging Authority](https://packaging.python.org/en/latest/).
