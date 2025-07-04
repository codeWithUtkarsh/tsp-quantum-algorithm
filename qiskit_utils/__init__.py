# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
=========================================================
Qiskit optimization module (:mod:`qiskit_utils`)
=========================================================

.. currentmodule:: qiskit_utils

Qiskit optimization module covers the whole range from high-level modeling of optimization
problems, with automatic conversion of problems to different required representations,
to a suite of easy-to-use quantum optimization algorithms that are ready to run on
classical simulators, as well as on real quantum devices via Qiskit.

This module enables easy, efficient modeling of optimization problems using `docplex
<https://ibmdecisionoptimization.github.io/docplex-doc/>`_.
A uniform interface as well as automatic conversion between different problem representations
allows users to solve problems using a large set of algorithms, from variational quantum algorithms,
such as the Quantum Approximate Optimization Algorithm
(:class:`~qiskit_algorithms.QAOA`), to
`Grover Adaptive Search <https://arxiv.org/abs/quant-ph/9607014>`_
(:class:`~algorithms.GroverOptimizer`), leveraging
fundamental `minimum eigensolvers
<https://qiskit-community.github.io/qiskit-algorithms/apidocs/qiskit_algorithms.html#minimum-eigensolvers>`_
provided by
`Qiskit Algorithms <https://qiskit-community.github.io/qiskit-algorithms/>`_.
Furthermore, the modular design
of the optimization module allows it to be easily extended and facilitates rapid development and
testing of new algorithms. Compatible classical optimizers are also provided for testing,
validation, and benchmarking.

Qiskit optimization module supports Quadratically Constrained Quadratic Programs – for simplicity
we refer to them just as Quadratic Programs – with binary, integer, and continuous variables, as
well as equality and inequality constraints. This class of optimization problems has a vast amount
of relevant applications, while still being efficiently representable by matrices and vectors.
This class covers some very interesting sub-classes, from Convex Continuous Quadratic Programs,
which can be solved efficiently by classical optimization algorithms, to Quadratic Unconstrained
Binary Optimization (QUBO) problems, which cover many NP-complete, i.e., classically intractable,
problems.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    QuadraticProgram

Representation of a Quadratically Constrained Quadratic Program supporting inequality and
equality constraints as well as continuous, binary, and integer variables.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    QiskitOptimizationError

In addition to standard Python errors the optimization module will raise this error if circumstances
are that it cannot proceed to completion.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    ~infinity.INFINITY

Submodules
==========

.. autosummary::
   :toctree:

   algorithms
   applications
   converters
   problems
   translators

"""

from .exceptions import QiskitOptimizationError, AlgorithmError
from .infinity import INFINITY  # must be at the top of the file
from .problems.quadratic_program import QuadraticProgram
from .version import __version__

__all__ = [
    "__version__",
    "QuadraticProgram",
    "QiskitOptimizationError",
    "AlgorithmError",
    "INFINITY",
]
