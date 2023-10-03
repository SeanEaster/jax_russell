# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2023-09-29

* Black-Scholes-Merton for all supported asset types
* Tested support for solving volatility, time to expiration, interest rate, cost of carry and strike
* JIT optimizations, dependency fixes

## [0.1.1] - 2023-09-12

* First release on PyPI.

### Added

 - Cox-Ross-Rubinstein binomial trees (supports American and European)
 - Rendleman Bartter trees (supports American and European)
 - Mixin classes to support specific assets