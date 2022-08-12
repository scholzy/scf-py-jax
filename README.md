# `scf.py`: a toy Hartree-Fock code in Python and JAX

This repository contains the source code (`scf/scf.py`) for a very simple Hartree-Fock code written in Python using the JAX linear algebra library.

The code adheres closely to the sample program from "Modern Quantum Chemistry" by Szabo and Ostlund but is implemented in modern vectorised Python.

## Requirements

The only requirements for this code are Python 3.7+ and JAX. It was written using Python 3.10.5 and JAX 0.3.15, both installed from conda-forge on MacOS 12.5.

## Installation

Just clone the repository, make sure you have appropriately recent versions of Python and JAX installed, and run `python3 scf/scf.py'. You should get an energy of -2.860662 Hartree, which agrees exactly with the published value of -2.8606619915 Hartree albeit to reduced precision, thanks to JAX using 32-bit floats by default.

## License

This code is distributed under the MIT license.