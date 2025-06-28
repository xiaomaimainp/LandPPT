"""
Shared service instances to ensure data consistency across modules
"""

from .enhanced_ppt_service import EnhancedPPTService
from .db_project_manager import DatabaseProjectManager

# Global service instances (lazy initialization)
_ppt_service = None
_project_manager = None

def get_ppt_service() -> EnhancedPPTService:
    """Get PPT service instance (lazy initialization)"""
    global _ppt_service
    if _ppt_service is None:
        _ppt_service = EnhancedPPTService()
    return _ppt_service

def get_project_manager() -> DatabaseProjectManager:
    """Get project manager instance (lazy initialization)"""
    global _project_manager
    if _project_manager is None:
        _project_manager = DatabaseProjectManager()
    return _project_manager

def reload_services():
    """Reload all service instances to pick up new configuration"""
    global _ppt_service, _project_manager
    _ppt_service = None
    _project_manager = None

# Backward compatibility - create module-level variables that get updated
def _update_module_vars():
    """Update module-level variables for backward compatibility"""
    import sys
    current_module = sys.modules[__name__]
    current_module.ppt_service = get_ppt_service()
    current_module.project_manager = get_project_manager()

# Initialize module variables
_update_module_vars()

# Override reload_services to also update module variables
_original_reload_services = reload_services

def reload_services():
    """Reload all service instances and update module variables"""
    _original_reload_services()
    _update_module_vars()

# Export for easy import
__all__ = ['get_ppt_service', 'get_project_manager', 'reload_services', 'ppt_service', 'project_manager']
