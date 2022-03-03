# Supporting python modules for "Constriction of actin rings by passive crosslinkers"

Python package that implements the analytical model from [Ref. 1](#references).

It also includes a module for creating plots with this model, as well as plots from finite element calculations output from a related package ([elastic-rings](https://github.com/cumberworth/elastic-rings)) and plots from Monte Carlo simulations from another related package ([ActinRingsMC.jl](https://github.com/cumberworth/ActinRingsMC.jl)).

Some example scripts for creating plots are provided in the [`scripts`](scripts/) directory.

## Installation

This package was developed and used on Linux.
[It is available on the PyPI respository](https://pypi.org/project/actinrings/).
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

## References

[1] A. Cumberworth and P. R. t. Wolde, Constriction of actin rings by passive crosslinkers.

## Links

[Python Packaging Authority](https://packaging.python.org/en/latest/)

[elastic-rings](https://github.com/cumberworth/elastic-rings)

[ActinRingsMC.jl](https://github.com/cumberworth/ActinRingsMC.jl)
