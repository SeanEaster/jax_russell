# jax_russell


[![pypi](https://img.shields.io/pypi/v/jax_russell.svg)](https://pypi.org/project/jax_russell/)
[![python](https://img.shields.io/pypi/pyversions/jax_russell.svg)](https://pypi.org/project/jax_russell/)
[![Build Status](https://github.com/SeanEaster/jax_russell/actions/workflows/dev.yml/badge.svg)](https://github.com/SeanEaster/jax_russell/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/SeanEaster/jax_russell/branch/main/graphs/badge.svg)](https://codecov.io/github/SeanEaster/jax_russell)



`jax-rusell` is a package that implements financial option formulas, and leverages Jax's autodifferentiation to support calculating "the greeks." 

Formulas are taken from Espen Haug's _The Complete Guide to Option Pricing Formulas_, unless otherwise noted.

**Pre-alpha, API unstable.**


* Documentation: <https://SeanEaster.github.io/jax_russell>
* GitHub: <https://github.com/SeanEaster/jax_russell>
* PyPI: <https://pypi.org/project/jax_russell/>
* Free software: GPL-3.0-only


## Features

* Standard tree methods, like Cox-Ross-Rubinstein and Rendleman Bartter, for American and European options
* Generalized Black-Scholes-Merton 
* First- and second-order risk measures ("the greeks") via auto-differentiation
* Support for options on stocks, stocks with continuous dividend, futures and margined futures (see the section "Usage")


### Planned

- More comprehensive testing (greeks against first- and second-differences)
- Skewed and leptokurtic methods


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
