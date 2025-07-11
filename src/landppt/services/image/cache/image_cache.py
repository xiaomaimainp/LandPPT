"""
图片缓存管理器
"""

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
from datetime import datetime, timedelta

from ..models import (
    ImageInfo, ImageCacheInfo, ImageSourceType, ImageProvider
)

logger = logging.getLogger(__name__)


class ImageCacheManager:
    """图片缓存管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 缓存配置 - 移除过期和清理设置，图片永久有效
        self.cache_root = Path(config.get('base_dir', config.get('cache_root', 'temp/images_cache')))

        # 缓存大小和清理配置（虽然图片永久有效，但保留配置项）
        self.max_size_gb = config.get('max_size_gb', 100.0)
        self.cleanup_interval_hours = config.get('cleanup_interval_hours', 240000)

        # 缓存目录结构
        self.ai_generated_dir = self.cache_root / 'ai_generated'
        self.web_search_dir = self.cache_root / 'web_search'
        self.local_storage_dir = self.cache_root / 'local_storage'
        self.metadata_dir = self.cache_root / 'metadata'
        self.thumbnails_dir = self.cache_root / 'thumbnails'

        # 创建目录结构
        self._create_cache_directories()

        # 缓存索引
        self._cache_index: Dict[str, ImageCacheInfo] = {}
        self._load_cache_index()
    
    def _create_cache_directories(self):
        """创建缓存目录结构"""
        directories = [
            self.cache_root,
            self.ai_generated_dir,
            self.ai_generated_dir / 'dalle',
            self.ai_generated_dir / 'stable_diffusion',
            self.web_search_dir,
            self.web_search_dir / 'unsplash',
            self.web_search_dir / 'pixabay',
            self.local_storage_dir,
            self.local_storage_dir / 'user_uploads',
            self.local_storage_dir / 'processed',
            self.metadata_dir,
            self.thumbnails_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, source_type: ImageSourceType, provider: ImageProvider) -> Path:
        """获取缓存路径"""
        if source_type == ImageSourceType.AI_GENERATED:
            if provider == ImageProvider.DALLE:
                return self.ai_generated_dir / 'dalle'
            elif provider == ImageProvider.STABLE_DIFFUSION:
                return self.ai_generated_dir / 'stable_diffusion'
            else:
                return self.ai_generated_dir
        
        elif source_type == ImageSourceType.WEB_SEARCH:
            if provider == ImageProvider.UNSPLASH:
                return self.web_search_dir / 'unsplash'
            elif provider == ImageProvider.PIXABAY:
                return self.web_search_dir / 'pixabay'
            else:
                return self.web_search_dir
        
        elif source_type == ImageSourceType.LOCAL_STORAGE:
            if provider == ImageProvider.USER_UPLOAD:
                return self.local_storage_dir / 'user_uploads'
            else:
                return self.local_storage_dir / 'processed'
        
        return self.cache_root
    
    def _generate_cache_key(self, image_info: ImageInfo) -> str:
        """生成缓存键"""
        # 使用图片ID、来源类型和提供者生成唯一键
        content = f"{image_info.image_id}_{image_info.source_type}_{image_info.provider}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def cache_image(self, image_info: ImageInfo, image_data: bytes) -> str:
        """缓存图片"""
        try:
            cache_key = self._generate_cache_key(image_info)
            cache_path = self._get_cache_path(image_info.source_type, image_info.provider)
            
            # 确定文件扩展名
            file_extension = Path(image_info.filename).suffix
            if not file_extension:
                file_extension = f".{image_info.metadata.format.value}"
            
            file_path = cache_path / f"{cache_key}{file_extension}"
            
            # 保存图片文件
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_image_file, file_path, image_data
            )
            
            # 更新图片信息中的本地路径
            image_info.local_path = str(file_path)
            
            # 创建缓存信息 - 移除过期时间，图片永久有效
            cache_info = ImageCacheInfo(
                cache_key=cache_key,
                file_path=str(file_path),
                file_size=len(image_data),
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                expires_at=None  # 永不过期
            )
            
            # 更新缓存索引
            self._cache_index[cache_key] = cache_info
            
            # 保存图片元数据
            await self._save_image_metadata(cache_key, image_info)
            
            # 异步保存缓存索引
            asyncio.create_task(self._save_cache_index())
            
            logger.info(f"Image cached successfully: {cache_key}")
            return cache_key
            
        except Exception as e:
            logger.error(f"Failed to cache image {image_info.image_id}: {e}")
            raise
    
    def _save_image_file(self, file_path: Path, image_data: bytes):
        """保存图片文件"""
        with open(file_path, 'wb') as f:
            f.write(image_data)
    
    async def get_cached_image(self, cache_key: str) -> Optional[Tuple[ImageInfo, Path]]:
        """获取缓存的图片"""
        try:
            cache_info = self._cache_index.get(cache_key)
            if not cache_info:
                return None

            # 检查文件是否存在
            file_path = Path(cache_info.file_path)
            if not file_path.exists():
                await self.remove_from_cache(cache_key)
                return None

            # 加载图片元数据
            image_info = await self._load_image_metadata(cache_key)
            if not image_info:
                return None

            # 更新访问信息
            cache_info.update_access()
            asyncio.create_task(self._save_cache_index())

            logger.debug(f"Cache hit: {cache_key}")
            return image_info, file_path

        except Exception as e:
            logger.error(f"Failed to get cached image {cache_key}: {e}")
            return None
    
    async def is_cached(self, image_info: ImageInfo) -> Optional[str]:
        """检查图片是否已缓存"""
        cache_key = self._generate_cache_key(image_info)
        cache_info = self._cache_index.get(cache_key)

        if cache_info:
            file_path = Path(cache_info.file_path)
            if file_path.exists():
                return cache_key

        return None
    
    async def remove_from_cache(self, cache_key: str) -> bool:
        """从缓存中移除图片"""
        try:
            cache_info = self._cache_index.get(cache_key)
            if not cache_info:
                return False
            
            # 删除图片文件
            file_path = Path(cache_info.file_path)
            if file_path.exists():
                await asyncio.get_event_loop().run_in_executor(None, file_path.unlink)
            
            # 删除缩略图
            thumbnail_path = self.thumbnails_dir / f"{cache_key}.jpg"
            if thumbnail_path.exists():
                await asyncio.get_event_loop().run_in_executor(None, thumbnail_path.unlink)
            
            # 删除元数据
            metadata_path = self.metadata_dir / f"{cache_key}.json"
            if metadata_path.exists():
                await asyncio.get_event_loop().run_in_executor(None, metadata_path.unlink)
            
            # 从索引中移除
            del self._cache_index[cache_key]
            
            # 保存索引
            await self._save_cache_index()
            
            logger.info(f"Removed from cache: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove from cache {cache_key}: {e}")
            return False
    
    async def _save_image_metadata(self, cache_key: str, image_info: ImageInfo):
        """保存图片元数据"""
        metadata_path = self.metadata_dir / f"{cache_key}.json"
        metadata = image_info.model_dump()
        
        def _save():
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def _load_image_metadata(self, cache_key: str) -> Optional[ImageInfo]:
        """加载图片元数据"""
        metadata_path = self.metadata_dir / f"{cache_key}.json"
        if not metadata_path.exists():
            return None
        
        def _load():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        try:
            metadata = await asyncio.get_event_loop().run_in_executor(None, _load)
            return ImageInfo(**metadata)
        except Exception as e:
            logger.error(f"Failed to load image metadata {cache_key}: {e}")
            return None
    
    def _load_cache_index(self):
        """加载缓存索引"""
        index_path = self.cache_root / 'cache_index.json'
        if not index_path.exists():
            return
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for cache_key, cache_data in data.items():
                self._cache_index[cache_key] = ImageCacheInfo(**cache_data)
            
            logger.info(f"Loaded cache index with {len(self._cache_index)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")
    
    async def _save_cache_index(self):
        """保存缓存索引"""
        index_path = self.cache_root / 'cache_index.json'
        
        def _save():
            data = {
                cache_key: cache_info.model_dump()
                for cache_key, cache_info in self._cache_index.items()
            }
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        try:
            await asyncio.get_event_loop().run_in_executor(None, _save)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    # 移除清理任务相关方法，因为图片永久有效，无需清理
    
    async def get_cache_size(self) -> int:
        """获取缓存总大小"""
        total_size = 0
        for cache_info in self._cache_index.values():
            total_size += cache_info.file_size
        return total_size
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_entries = len(self._cache_index)
        total_size = await self.get_cache_size()
        
        # 按来源类型统计
        source_stats = {}
        for cache_info in self._cache_index.values():
            file_path = Path(cache_info.file_path)
            source_type = file_path.parent.parent.name if file_path.parent.parent.name in ['ai_generated', 'web_search', 'local_storage'] else 'unknown'
            
            if source_type not in source_stats:
                source_stats[source_type] = {'count': 0, 'size': 0}
            
            source_stats[source_type]['count'] += 1
            source_stats[source_type]['size'] += cache_info.file_size
        
        # 转换为API需要的格式
        categories = {}
        for source_type, stats in source_stats.items():
            categories[source_type] = stats['count']

        return {
            'total_entries': total_entries,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'categories': categories,
            'source_stats': source_stats
        }
    
    async def clear_cache(self, source_type: Optional[ImageSourceType] = None) -> int:
        """清空缓存"""
        if source_type:
            # 清空特定来源的缓存
            keys_to_remove = []
            for cache_key, cache_info in self._cache_index.items():
                file_path = Path(cache_info.file_path)
                if source_type.value in str(file_path):
                    keys_to_remove.append(cache_key)
            
            for cache_key in keys_to_remove:
                await self.remove_from_cache(cache_key)
            
            return len(keys_to_remove)
        else:
            # 清空所有缓存
            keys_to_remove = list(self._cache_index.keys())
            for cache_key in keys_to_remove:
                await self.remove_from_cache(cache_key)
            
            return len(keys_to_remove)
