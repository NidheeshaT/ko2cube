class KWOKError(Exception):
    """Base exception for all KWOK related errors."""
    pass

class NodeValidationError(KWOKError):
    """Raised when a Node resource fails validation."""
    pass

class PodValidationError(KWOKError):
    """Raised when a Pod resource fails validation (e.g., assigned to non-existent node)."""
    pass

class InstanceTypeError(KWOKError):
    """Raised when an invalid instance type is encountered."""
    pass
