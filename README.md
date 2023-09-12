# jax_russell


[![pypi](https://img.shields.io/pypi/v/jax_russell.svg)](https://pypi.org/project/jax_russell/)
[![python](https://img.shields.io/pypi/pyversions/jax_russell.svg)](https://pypi.org/project/jax_russell/)
[![Build Status](https://github.com/SeanEaster/jax_russell/actions/workflows/dev.yml/badge.svg)](https://github.com/SeanEaster/jax_russell/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/SeanEaster/jax_russell/branch/main/graphs/badge.svg)](https://codecov.io/github/SeanEaster/jax_russell)



`jax-rusell` is a package that implements financial option formulas, and leverages Jax's autodifferentiation to support calculating "the greeks." 

Formulas are generally taken from Espen Haug's _The Complete Guide to Option Pricing Formulas_.


* Documentation: <https://SeanEaster.github.io/jax_russell>
* GitHub: <https://github.com/SeanEaster/jax_russell>
* PyPI: <https://pypi.org/project/jax_russell/>
* Free software: GPL-3.0-only


## Features

* Classes implementing standard tree methods, like Cox-Ross-Rubinstein and Rendleman Bartter

### Planned

- Black Scholes and variations
- More comprehensive testing


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
