"""Exception families shared with the asynchronous recognition library."""


class TTSError(Exception):
    """Base exception for speech synthesis."""


class ConfigurationError(TTSError, ValueError):
    """A configuration or input value is invalid."""


class EngineStateError(TTSError, RuntimeError):
    """The requested operation is incompatible with the engine state."""


class EngineClosedError(EngineStateError):
    """The engine has closed or closing it cancelled an operation."""


class InferenceError(TTSError, RuntimeError):
    """A model operation or resource release failed."""


class EmptyOutputError(InferenceError):
    """Synthesis of nonempty text completed without audio samples."""


class EnvironmentUnavailableError(TTSError, RuntimeError):
    """Preparation cannot proceed in the available native environment."""


class ModuleNotInstalledError(TTSError, ImportError):
    """A registered engine module is not installed."""
