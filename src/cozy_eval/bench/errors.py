"""The one error hierarchy. Every exception this library raises is a
:class:`CozyBenchError`.

The distinction the classes draw is the one a caller can act on:

  ConfigError    you passed something wrong — an unknown set name, a bad
                 argument, mismatched input lengths. Fix the call.
  DataError      a checklist / prompt set / spec on disk is malformed or
                 internally inconsistent. Fix the data.
  RegistryError  a metric registration or lookup is invalid. Fix the metric.
  BackendError   a scoring backend could not run (missing package, missing
                 weights, a model that answered nonsense). Fix the environment.

Every class also subclasses a builtin, so ``except ValueError`` written against
an earlier version keeps working.
"""

from __future__ import annotations


class CozyBenchError(Exception):
    """Base class for everything this library raises."""


class ConfigError(CozyBenchError, ValueError):
    """A caller passed an invalid argument or named something unknown."""


class DataError(CozyBenchError, ValueError):
    """A checklist, prompt set, or spec is malformed or inconsistent."""


class RegistryError(CozyBenchError, ValueError):
    """A metric registration or lookup is invalid."""


class BackendError(CozyBenchError, RuntimeError):
    """A scoring backend was unavailable or returned something unusable."""


__all__ = [
    "BackendError",
    "ConfigError",
    "CozyBenchError",
    "DataError",
    "RegistryError",
]
