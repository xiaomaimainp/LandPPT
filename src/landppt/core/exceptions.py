"""
LandPPT Exception Classes
"""

class LandPPTException(Exception):
    """Base exception for all LandPPT exceptions"""
    pass

class ConfigurationError(LandPPTException):
    """Raised when there is a configuration error"""
    pass

class AIProviderError(LandPPTException):
    """Raised when there is an error with the AI provider"""
    pass

class FileProcessingError(LandPPTException):
    """Raised when there is an error processing a file"""
    pass

class DatabaseError(LandPPTException):
    """Raised when there is a database error"""
    pass

class AuthenticationError(LandPPTException):
    """Raised when there is an authentication error"""
    pass

class ResourceNotFoundError(LandPPTException):
    """Raised when a requested resource is not found"""
    pass

class ValidationError(LandPPTException):
    """Raised when validation fails"""
    pass

class ServiceUnavailableError(LandPPTException):
    """Raised when a service is unavailable"""
    pass

class ImageProcessingError(LandPPTException):
    """Raised when there is an error processing an image"""
    pass

class ResearchError(LandPPTException):
    """Raised when there is an error with research functionality"""
    pass

class PPTGenerationError(LandPPTException):
    """Raised when there is an error generating a PowerPoint presentation"""
    pass