"""The one error hierarchy. Every exception this library raises is a
:class:`CozyEvalError`.

The distinction the classes draw is the one a caller can act on:

  ConfigError    you passed something wrong — an unknown set name, a bad
                 argument, mismatched input lengths. Fix the call.
  DataError      a checklist / prompt set / spec on disk is malformed or
                 internally inconsistent. Fix the data.
  RegistryError  a metric registration or lookup is invalid. Fix the metric.
  BackendError   a scoring backend could not run (missing package, missing
                 weights, a model that answered nonsense). Fix the environment.
  DecodeError    frames could not be read out of a source. Fix the input.
  ProtocolError  the run's declared protocol is incomplete or inconsistent.
                 Fix the protocol.
  SampleSizeError
                 a population statistic was asked for below the sample size it
                 needs. Render more prompts.
  TrajectoryPerturbingError
                 reference metrics were asked for on a change that moves the
                 sampling trajectory. A REFUSAL, not a failure: the fix is a
                 different question, not a different argument.

Every class also subclasses the builtin a caller would reach for first, so
``except ValueError`` keeps working.
"""

from __future__ import annotations


class CozyEvalError(Exception):
    """Base class for everything this library raises."""


class ConfigError(CozyEvalError, ValueError):
    """A caller passed an invalid argument or named something unknown."""


class DataError(CozyEvalError, ValueError):
    """A checklist, prompt set, or spec is malformed or inconsistent."""


class RegistryError(CozyEvalError, ValueError):
    """A metric registration or lookup is invalid."""


class BackendError(CozyEvalError, RuntimeError):
    """A scoring backend was unavailable or returned something unusable."""


class DecodeError(CozyEvalError, RuntimeError):
    """A frame source could not be decoded."""


class ProtocolError(CozyEvalError, ValueError):
    """The protocol stamp is incomplete or internally inconsistent."""


class SampleSizeError(CozyEvalError, RuntimeError):
    """A population statistic was asked for below the sample size it needs."""


class TrajectoryPerturbingError(CozyEvalError, RuntimeError):
    """Reference metrics were requested for a trajectory-perturbing change."""


__all__ = [
    "BackendError",
    "ConfigError",
    "CozyEvalError",
    "DataError",
    "DecodeError",
    "ProtocolError",
    "RegistryError",
    "SampleSizeError",
    "TrajectoryPerturbingError",
]
