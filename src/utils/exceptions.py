class DataLoadingError(Exception):
    """Raised when the dataset cannot be loaded."""


class DataValidationError(Exception):
    """Raised when the dataset does not satisfy validation rules."""


class ModelConfigurationError(Exception):
    """Raised when the selected model or hyperparameters are invalid."""


class TrainingExecutionError(Exception):
    """Raised when model training fails."""


class MLflowLoggingError(Exception):
    """Raised when MLflow tracking or logging fails."""