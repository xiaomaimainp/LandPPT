"""
Configuration management service for LandPPT
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv, set_key, unset_key

logger = logging.getLogger(__name__)


class ConfigService:
    """Configuration management service"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self.env_path = Path(env_file)
        
        # Ensure .env file exists
        if not self.env_path.exists():
            self.env_path.touch()
        
        # Load environment variables
        load_dotenv(self.env_file)
        
        # Configuration schema
        self.config_schema = {
            # AI Provider Configuration
            "openai_api_key": {"type": "password", "category": "ai_providers"},
            "openai_base_url": {"type": "url", "category": "ai_providers", "default": "https://api.openai.com/v1"},
            "openai_model": {"type": "select", "category": "ai_providers", "default": "gpt-3.5-turbo"},
            
            "anthropic_api_key": {"type": "password", "category": "ai_providers"},
            "anthropic_model": {"type": "select", "category": "ai_providers", "default": "claude-3-haiku-20240307"},

            "google_api_key": {"type": "password", "category": "ai_providers"},
            "google_model": {"type": "text", "category": "ai_providers", "default": "gemini-1.5-flash"},
            
            "azure_openai_api_key": {"type": "password", "category": "ai_providers"},
            "azure_openai_endpoint": {"type": "url", "category": "ai_providers"},
            "azure_openai_deployment_name": {"type": "text", "category": "ai_providers"},
            "azure_openai_api_version": {"type": "text", "category": "ai_providers", "default": "2024-02-15-preview"},
            
            "ollama_base_url": {"type": "url", "category": "ai_providers", "default": "http://localhost:11434"},
            "ollama_model": {"type": "text", "category": "ai_providers", "default": "llama2"},
            
            "default_ai_provider": {"type": "select", "category": "ai_providers", "default": "openai"},
            
            # Generation Parameters
            "max_tokens": {"type": "number", "category": "generation_params", "default": "2000"},
            "temperature": {"type": "number", "category": "generation_params", "default": "0.7"},
            "top_p": {"type": "number", "category": "generation_params", "default": "1.0"},
            
            "tavily_api_key": {"type": "password", "category": "generation_params"},
            "tavily_max_results": {"type": "number", "category": "generation_params", "default": "10"},
            "tavily_search_depth": {"type": "select", "category": "generation_params", "default": "advanced"},

            "apryse_license_key": {"type": "password", "category": "generation_params"},
            
            # Feature Flags
            "enable_network_mode": {"type": "boolean", "category": "feature_flags", "default": "true"},
            "enable_local_models": {"type": "boolean", "category": "feature_flags", "default": "false"},
            "enable_streaming": {"type": "boolean", "category": "feature_flags", "default": "true"},
            "log_level": {"type": "select", "category": "feature_flags", "default": "INFO"},
            "log_ai_requests": {"type": "boolean", "category": "feature_flags", "default": "false"},
            "debug": {"type": "boolean", "category": "feature_flags", "default": "true"},
            
            # App Configuration
            "host": {"type": "text", "category": "app_config", "default": "0.0.0.0"},
            "port": {"type": "number", "category": "app_config", "default": "8000"},
            "base_url": {"type": "url", "category": "app_config", "default": "http://localhost:8000"},
            "reload": {"type": "boolean", "category": "app_config", "default": "true"},
            "secret_key": {"type": "password", "category": "app_config", "default": "your-very-secure-secret-key"},
            "access_token_expire_minutes": {"type": "number", "category": "app_config", "default": "30"},
            "max_file_size": {"type": "number", "category": "app_config", "default": "10485760"},
            "upload_dir": {"type": "text", "category": "app_config", "default": "uploads"},
            "cache_ttl": {"type": "number", "category": "app_config", "default": "3600"},
            "database_url": {"type": "text", "category": "app_config", "default": "sqlite:///./landppt.db"},

            # Image Service Configuration
            "enable_image_service": {"type": "boolean", "category": "image_service", "default": "false"},

            # Multi-source Image Configuration
            "enable_local_images": {"type": "boolean", "category": "image_service", "default": "true"},
            "enable_network_search": {"type": "boolean", "category": "image_service", "default": "false"},
            "enable_ai_generation": {"type": "boolean", "category": "image_service", "default": "false"},

            # Local Images Configuration
            "local_images_smart_selection": {"type": "boolean", "category": "image_service", "default": "true"},
            "max_local_images_per_slide": {"type": "number", "category": "image_service", "default": "2"},

            # Network Search Configuration
            "max_network_images_per_slide": {"type": "number", "category": "image_service", "default": "2"},

            # AI Generation Configuration
            "default_ai_image_provider": {"type": "select", "category": "image_service", "default": "dalle"},
            "max_ai_images_per_slide": {"type": "number", "category": "image_service", "default": "1"},
            "ai_image_quality": {"type": "select", "category": "image_service", "default": "standard"},

            # Global Image Configuration
            "max_total_images_per_slide": {"type": "number", "category": "image_service", "default": "3"},
            "enable_smart_image_selection": {"type": "boolean", "category": "image_service", "default": "true"},

            # Image Generation Providers
            "openai_api_key_image": {"type": "password", "category": "image_service"},
            "stability_api_key": {"type": "password", "category": "image_service"},
            "siliconflow_api_key": {"type": "password", "category": "image_service"},
            "default_image_provider": {"type": "select", "category": "image_service", "default": "dalle"},

            # Image Search Providers
            "unsplash_access_key": {"type": "password", "category": "image_service"},
            "pixabay_api_key": {"type": "password", "category": "image_service"},
            "dalle_image_size": {"type": "select", "category": "image_service", "default": "1792x1024"},
            "dalle_image_quality": {"type": "select", "category": "image_service", "default": "standard"},
            "dalle_image_style": {"type": "select", "category": "image_service", "default": "natural"},
            "siliconflow_image_size": {"type": "select", "category": "image_service", "default": "1024x1024"},
            "siliconflow_steps": {"type": "number", "category": "image_service", "default": 20},
            "siliconflow_guidance_scale": {"type": "number", "category": "image_service", "default": 7.5},
        }
        

    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values"""
        config = {}
        
        for key, schema in self.config_schema.items():
            env_key = key.upper()
            value = os.getenv(env_key)
            
            if value is None:
                value = schema.get("default", "")
            
            # Convert boolean strings
            if schema["type"] == "boolean":
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
            
            config[key] = value
        
        return config
    
    def get_config_by_category(self, category: str) -> Dict[str, Any]:
        """Get configuration values by category"""
        config = {}
        
        for key, schema in self.config_schema.items():
            if schema["category"] == category:
                env_key = key.upper()
                value = os.getenv(env_key)
                
                if value is None:
                    value = schema.get("default", "")
                
                # Convert boolean strings
                if schema["type"] == "boolean":
                    if isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes", "on")
                
                config[key] = value
        
        return config
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update configuration values"""
        try:
            for key, value in config.items():
                if key in self.config_schema:
                    env_key = key.upper()

                    # Convert boolean values to strings
                    if isinstance(value, bool):
                        value = "true" if value else "false"
                    else:
                        value = str(value)

                    # Update .env file (without quotes)
                    set_key(self.env_file, env_key, value, quote_mode="never")

                    # Update current environment
                    os.environ[env_key] = value

            # Reload environment variables
            load_dotenv(self.env_file, override=True)

            # Reload AI configuration if any AI-related config was updated
            ai_related_keys = [k for k in config.keys() if k in self.config_schema and
                             self.config_schema[k]["category"] in ["ai_providers", "generation_params"]]
            if ai_related_keys:
                self._reload_ai_config()

            # Reload app configuration if any app-related config was updated
            app_related_keys = [k for k in config.keys() if k in self.config_schema and
                              self.config_schema[k]["category"] == "app_config"]
            if app_related_keys:
                self._reload_app_config()

            # Reload image service configuration if any image-related config was updated
            image_related_keys = [k for k in config.keys() if k in self.config_schema and
                                self.config_schema[k]["category"] == "image_service"]
            if image_related_keys:
                self._reload_image_config()

            logger.info(f"Updated {len(config)} configuration values")
            return True

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def _reload_ai_config(self):
        """Reload AI configuration"""
        try:
            from ..core.config import reload_ai_config, ai_config
            from ..ai.providers import reload_ai_providers
            from .service_instances import reload_services

            logger.info("Starting AI configuration reload process...")

            # Reload AI configuration
            reload_ai_config()
            logger.info(f"AI config reloaded. Tavily API key: {'***' + ai_config.tavily_api_key[-4:] if ai_config.tavily_api_key and len(ai_config.tavily_api_key) > 4 else 'None'}")

            # Clear AI provider cache to force reload with new config
            reload_ai_providers()
            logger.info("AI providers reloaded")

            # Reload service instances to pick up new configuration
            reload_services()
            logger.info("Service instances reloaded")

            logger.info("AI configuration, providers, and services reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload AI configuration: {e}")
            import traceback
            logger.error(f"Reload traceback: {traceback.format_exc()}")

    def _reload_app_config(self):
        """Reload application configuration"""
        try:
            from ..core.config import app_config

            # Force reload of app configuration
            app_config.__init__()

            logger.info("Application configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload application configuration: {e}")

    def _reload_image_config(self):
        """Reload image service configuration"""
        try:
            from ..services.image.config.image_config import image_config

            # 重新加载环境变量配置
            image_config._load_env_config()

            logger.info("Image service configuration reloaded")
        except Exception as e:
            logger.error(f"Failed to reload image service configuration: {e}")

    def update_config_by_category(self, category: str, config: Dict[str, Any]) -> bool:
        """Update configuration values for a specific category"""
        # Filter config to only include keys from the specified category
        filtered_config = {}
        
        for key, value in config.items():
            if key in self.config_schema and self.config_schema[key]["category"] == category:
                filtered_config[key] = value
        
        return self.update_config(filtered_config)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema"""
        return self.config_schema
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate configuration values"""
        errors = {}
        
        for key, value in config.items():
            if key not in self.config_schema:
                if "unknown" not in errors:
                    errors["unknown"] = []
                errors["unknown"].append(f"Unknown configuration key: {key}")
                continue
            
            schema = self.config_schema[key]
            field_errors = []
            
            # Type validation
            if schema["type"] == "number":
                try:
                    num_value = float(value)
                    # Special validation for access_token_expire_minutes - allow 0 for never expire
                    if key == "access_token_expire_minutes" and num_value < 0:
                        field_errors.append(f"{key} must be 0 (never expire) or a positive number")
                except (ValueError, TypeError):
                    field_errors.append(f"{key} must be a number")
            
            elif schema["type"] == "boolean":
                if isinstance(value, str):
                    if value.lower() not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
                        field_errors.append(f"{key} must be a boolean value")
            
            elif schema["type"] == "url":
                if value and not (value.startswith("http://") or value.startswith("https://")):
                    field_errors.append(f"{key} must be a valid URL")
            
            if field_errors:
                errors[key] = field_errors
        
        return errors
    
    def reset_to_defaults(self, category: Optional[str] = None) -> bool:
        """Reset configuration to default values"""
        try:
            config_to_reset = {}
            
            for key, schema in self.config_schema.items():
                if category is None or schema["category"] == category:
                    default_value = schema.get("default", "")
                    config_to_reset[key] = default_value
            
            return self.update_config(config_to_reset)
            
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def backup_config(self, backup_file: str) -> bool:
        """Backup current configuration"""
        try:
            config = self.get_all_config()
            
            with open(backup_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration backed up to {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
            return False
    
    def restore_config(self, backup_file: str) -> bool:
        """Restore configuration from backup"""
        try:
            with open(backup_file, 'r') as f:
                config = json.load(f)
            
            return self.update_config(config)
            
        except Exception as e:
            logger.error(f"Failed to restore configuration: {e}")
            return False


# Global config service instance
config_service = ConfigService()


def get_config_service() -> ConfigService:
    """Get config service instance"""
    return config_service
