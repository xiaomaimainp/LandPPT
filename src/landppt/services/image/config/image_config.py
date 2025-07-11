"""
图片服务配置管理
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ImageServiceConfig:
    """图片服务配置管理器"""
    
    def __init__(self):
        self._config = self._load_default_config()
        self._load_env_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            # DALL-E配置
            'dalle': {
                'api_key': '',
                'api_base': 'https://api.openai.com/v1',
                'model': 'dall-e-3',
                'default_size': '1792x1024',  # 16:9比例，适合PPT
                'default_quality': 'standard',  # standard, hd
                'default_style': 'natural',  # vivid, natural
                'rate_limit_requests': 50,  # 每分钟请求数
                'rate_limit_window': 60,  # 时间窗口（秒）
                'timeout': 180  # 请求超时（秒）
            },
            
            # Stable Diffusion配置
            'stable_diffusion': {
                'api_key': '',
                'api_base': 'https://api.stability.ai/v1',
                'engine_id': 'stable-diffusion-xl-1024-v1-0',
                'default_width': 1024,
                'default_height': 576,  # 16:9比例
                'default_steps': 30,
                'default_cfg_scale': 7.0,
                'default_sampler': 'K_DPM_2_ANCESTRAL',
                'rate_limit_requests': 150,  # 每分钟请求数
                'rate_limit_window': 60,  # 时间窗口（秒）
                'timeout': 180  # 请求超时（秒）
            },

            # SiliconFlow配置
            'siliconflow': {
                'api_key': '',
                'api_base': 'https://api.siliconflow.cn/v1',
                'model': 'Kwai-Kolors/Kolors',
                'default_size': '1024x1024',
                'default_batch_size': 1,
                'default_steps': 20,
                'default_guidance_scale': 7.5,
                'rate_limit_requests': 60,  # 每分钟请求数
                'rate_limit_window': 60,  # 时间窗口（秒）
                'timeout': 120  # 请求超时（秒）
            },
            
            # Unsplash配置（网络搜索）
            'unsplash': {
                'api_key': '',
                'api_base': 'https://api.unsplash.com',
                'per_page': 20,
                'rate_limit_requests': 50,
                'rate_limit_window': 3600  # 1小时
            },
            
            # Pixabay配置（网络搜索）
            'pixabay': {
                'api_key': '',
                'api_base': 'https://pixabay.com/api',
                'per_page': 20,
                'rate_limit_requests': 100,
                'rate_limit_window': 3600  # 1小时
            },
            
            # 缓存配置 - 简化配置，图片永久有效
            'cache': {
                'base_dir': 'temp/images_cache',
                'max_size_gb': 5.0,  # 默认最大缓存大小5GB
                'cleanup_interval_hours': 24  # 默认24小时清理一次（虽然图片永久有效，但保留配置项）
            },
            
            # 图片处理配置
            'processing': {
                'max_file_size_mb': 50,
                'supported_formats': ['jpg', 'jpeg', 'png', 'webp', 'gif'],
                'default_quality': 85,
                'thumbnail_size': (300, 200),
                'watermark_enabled': False,
                'watermark_text': 'LandPPT',
                'watermark_opacity': 0.3
            },
            
            # 智能匹配配置
            'matching': {
                'similarity_threshold': 0.3,
                'max_suggestions': 10,
                'keyword_weight': 0.4,
                'tag_weight': 0.3,
                'usage_weight': 0.2,
                'freshness_weight': 0.1
            },
            
            # PPT适配器配置
            'ppt_adapter': {
                'default_size': '1792x1024',
                'quality_mode': 'standard',
                'style_preference': 'natural',
                'enable_negative_prompts': True,
                'prompt_enhancement': True
            }
        }
    
    def _load_env_config(self):
        """从环境变量加载配置"""
        # DALL-E配置
        if os.getenv('OPENAI_API_KEY'):
            self._config['dalle']['api_key'] = os.getenv('OPENAI_API_KEY')
        
        if os.getenv('DALLE_API_BASE'):
            self._config['dalle']['api_base'] = os.getenv('DALLE_API_BASE')
        
        # Stable Diffusion配置
        if os.getenv('STABILITY_API_KEY'):
            self._config['stable_diffusion']['api_key'] = os.getenv('STABILITY_API_KEY')

        if os.getenv('STABILITY_API_BASE'):
            self._config['stable_diffusion']['api_base'] = os.getenv('STABILITY_API_BASE')

        # SiliconFlow配置
        if os.getenv('SILICONFLOW_API_KEY'):
            self._config['siliconflow']['api_key'] = os.getenv('SILICONFLOW_API_KEY')

        if os.getenv('SILICONFLOW_API_BASE'):
            self._config['siliconflow']['api_base'] = os.getenv('SILICONFLOW_API_BASE')

        # 从配置服务加载SiliconFlow配置
        try:
            from ...config_service import get_config_service
            config_service = get_config_service()
            all_config = config_service.get_all_config()

            # 加载SiliconFlow API密钥
            siliconflow_api_key = all_config.get('siliconflow_api_key')
            if siliconflow_api_key:
                self._config['siliconflow']['api_key'] = siliconflow_api_key

            # 加载SiliconFlow生成参数
            siliconflow_size = all_config.get('siliconflow_image_size')
            if siliconflow_size:
                self._config['siliconflow']['default_size'] = siliconflow_size

            siliconflow_steps = all_config.get('siliconflow_steps')
            if siliconflow_steps:
                self._config['siliconflow']['default_steps'] = int(siliconflow_steps)

            siliconflow_guidance = all_config.get('siliconflow_guidance_scale')
            if siliconflow_guidance:
                self._config['siliconflow']['default_guidance_scale'] = float(siliconflow_guidance)

        except Exception as e:
            logger.warning(f"Failed to load SiliconFlow config from config service: {e}")
        
        # Unsplash配置
        if os.getenv('UNSPLASH_ACCESS_KEY'):
            self._config['unsplash']['api_key'] = os.getenv('UNSPLASH_ACCESS_KEY')
        
        # Pixabay配置
        if os.getenv('PIXABAY_API_KEY'):
            self._config['pixabay']['api_key'] = os.getenv('PIXABAY_API_KEY')
        
        # 缓存目录配置
        if os.getenv('IMAGE_CACHE_DIR'):
            self._config['cache']['base_dir'] = os.getenv('IMAGE_CACHE_DIR')
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config.copy()
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """获取特定提供者的配置"""
        return self._config.get(provider, {}).copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self._config, updates)
    
    def is_provider_configured(self, provider: str) -> bool:
        """检查提供者是否已配置"""
        provider_config = self._config.get(provider, {})
        
        # 检查API密钥是否存在
        api_key = provider_config.get('api_key', '')
        return bool(api_key and api_key.strip())
    
    def get_configured_providers(self) -> List[str]:
        """获取已配置的提供者列表"""
        providers = []

        for provider in ['dalle', 'stable_diffusion', 'siliconflow', 'unsplash', 'pixabay']:
            if self.is_provider_configured(provider):
                providers.append(provider)

        return providers

    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config
    
    def validate_config(self) -> Dict[str, List[str]]:
        """验证配置并返回错误信息"""
        errors = {}

        # 验证缓存配置
        cache_config = self._config.get('cache', {})
        cache_errors = []

        # 检查max_size_gb（如果存在的话）
        max_size_gb = cache_config.get('max_size_gb')
        if max_size_gb is not None and max_size_gb <= 0:
            cache_errors.append("max_size_gb must be greater than 0")

        # 检查cleanup_interval_hours（如果存在的话）
        cleanup_interval = cache_config.get('cleanup_interval_hours')
        if cleanup_interval is not None and cleanup_interval <= 0:
            cache_errors.append("cleanup_interval_hours must be greater than 0")

        # 检查base_dir是否存在
        if not cache_config.get('base_dir'):
            cache_errors.append("base_dir is required")

        if cache_errors:
            errors['cache'] = cache_errors
        
        # 验证处理配置
        processing_config = self._config.get('processing', {})
        processing_errors = []
        
        if processing_config.get('max_file_size_mb', 0) <= 0:
            processing_errors.append("max_file_size_mb must be greater than 0")
        
        if not processing_config.get('supported_formats'):
            processing_errors.append("supported_formats cannot be empty")
        
        if processing_errors:
            errors['processing'] = processing_errors
        
        return errors
    
    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        import json
        
        config_path = Path(file_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, file_path: str):
        """从文件加载配置"""
        import json
        
        config_path = Path(file_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self.update_config(file_config)


# 创建全局配置实例
image_config = ImageServiceConfig()


def get_image_config() -> ImageServiceConfig:
    """获取图片服务配置实例"""
    return image_config


def update_image_config(updates: Dict[str, Any]):
    """更新图片服务配置"""
    image_config.update_config(updates)


def get_provider_config(provider: str) -> Dict[str, Any]:
    """获取特定提供者的配置"""
    return image_config.get_provider_config(provider)


def is_provider_configured(provider: str) -> bool:
    """检查提供者是否已配置"""
    return image_config.is_provider_configured(provider)
