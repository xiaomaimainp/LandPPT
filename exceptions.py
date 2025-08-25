"""
LandPPT Exceptions Module

This module defines exceptions used throughout the LandPPT application.
"""

# 标准警告类，从内置的 warnings 模块中导入
# 这些是 Python 内置的标准警告类
DeprecationWarning = DeprecationWarning
PendingDeprecationWarning = PendingDeprecationWarning
RuntimeWarning = RuntimeWarning
SyntaxWarning = SyntaxWarning
UserWarning = UserWarning
FutureWarning = FutureWarning
ImportWarning = ImportWarning
UnicodeWarning = UnicodeWarning
BytesWarning = BytesWarning
ResourceWarning = ResourceWarning

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