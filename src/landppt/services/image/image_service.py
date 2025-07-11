"""
图片服务主入口
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import time

from .models import (
    ImageInfo, ImageSearchRequest, ImageGenerationRequest, ImageUploadRequest,
    ImageSearchResult, ImageOperationResult, ImageProcessingOptions,
    ImageSourceType, ImageProvider
)
from .providers.base import provider_registry, ImageSearchProvider, ImageGenerationProvider, LocalStorageProvider
from .processors.image_processor import ImageProcessor
from .cache.image_cache import ImageCacheManager
from .matching.image_matcher import ImageMatcher
from .adapters.ppt_prompt_adapter import PPTPromptAdapter, PPTSlideContext

logger = logging.getLogger(__name__)


class ImageService:
    """图片服务主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化组件
        self.processor = ImageProcessor(config.get('processing', {}))
        self.cache_manager = ImageCacheManager(config.get('cache', {}))
        self.matcher = ImageMatcher(config.get('matching', {}))
        self.ppt_adapter = PPTPromptAdapter(config.get('ppt_adapter', {}))

        # 服务状态
        self.initialized = False
        
    async def initialize(self):
        """初始化服务"""
        if self.initialized:
            return
        
        try:
            # 初始化提供者（这里需要在具体实现中注册）
            await self._initialize_providers()
            
            logger.info("Image service initialized successfully")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize image service: {e}")
            raise
    
    async def _initialize_providers(self):
        """初始化图片提供者"""
        try:
            # 初始化本地存储提供者（总是注册）
            from .providers.local_storage_provider import FileSystemStorageProvider

            storage_config = self.config.get('storage', {})
            storage_provider = FileSystemStorageProvider(storage_config)
            provider_registry.register(storage_provider)
            logger.info("Local storage provider registered")

            # 初始化AI图片生成提供者
            from .providers.dalle_provider import DalleProvider
            from .providers.stable_diffusion_provider import StableDiffusionProvider
            from .providers.silicon_flow_provider import SiliconFlowProvider
            from .config.image_config import is_provider_configured

            # 注册DALL-E提供者
            if is_provider_configured('dalle'):
                dalle_config = self.config.get('dalle', {})
                dalle_provider = DalleProvider(dalle_config)
                provider_registry.register(dalle_provider)
                logger.info("DALL-E provider registered")
            else:
                logger.warning("DALL-E API key not configured, skipping provider registration")

            # 注册Stable Diffusion提供者
            if is_provider_configured('stable_diffusion'):
                sd_config = self.config.get('stable_diffusion', {})
                sd_provider = StableDiffusionProvider(sd_config)
                provider_registry.register(sd_provider)
                logger.info("Stable Diffusion provider registered")
            else:
                logger.warning("Stable Diffusion API key not configured, skipping provider registration")

            # 注册SiliconFlow提供者
            if is_provider_configured('siliconflow'):
                sf_config = self.config.get('siliconflow', {})
                sf_provider = SiliconFlowProvider(sf_config)
                provider_registry.register(sf_provider)
                logger.info("SiliconFlow provider registered")
            else:
                logger.warning("SiliconFlow API key not configured, skipping provider registration")

            # 统计已注册的提供者数量
            total_providers = (len(provider_registry.get_generation_providers()) +
                             len(provider_registry.get_search_providers()) +
                             len(provider_registry.get_storage_providers()))
            logger.info(f"Initialized {total_providers} image providers")

        except Exception as e:
            logger.error(f"Failed to initialize image providers: {e}")
    
    async def search_images(self, request: ImageSearchRequest) -> ImageSearchResult:
        """搜索图片"""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        all_images = []
        provider_results = {}
        
        try:
            # 获取启用的搜索提供者
            search_providers = provider_registry.get_search_providers()
            
            if not search_providers:
                return ImageSearchResult(
                    images=[], total_count=0, page=request.page,
                    per_page=request.per_page, has_next=False, has_prev=False,
                    search_time=time.time() - start_time
                )
            
            # 过滤提供者
            if request.preferred_providers:
                search_providers = [
                    p for p in search_providers 
                    if p.provider in request.preferred_providers
                ]
            
            if request.excluded_providers:
                search_providers = [
                    p for p in search_providers 
                    if p.provider not in request.excluded_providers
                ]
            
            # 并行搜索
            search_tasks = []
            for provider in search_providers:
                task = asyncio.create_task(self._search_with_provider(provider, request))
                search_tasks.append(task)
            
            # 等待所有搜索完成
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Search failed for provider {search_providers[i].provider}: {result}")
                    continue
                
                if isinstance(result, ImageSearchResult):
                    all_images.extend(result.images)
                    provider_results[search_providers[i].provider.value] = len(result.images)
            
            # 使用智能匹配器排序和过滤结果
            if all_images:
                all_images = await self.matcher.rank_images(request.query, all_images)
            
            # 分页处理
            total_count = len(all_images)
            start_idx = (request.page - 1) * request.per_page
            end_idx = start_idx + request.per_page
            page_images = all_images[start_idx:end_idx]
            
            return ImageSearchResult(
                images=page_images,
                total_count=total_count,
                page=request.page,
                per_page=request.per_page,
                has_next=end_idx < total_count,
                has_prev=request.page > 1,
                search_time=time.time() - start_time,
                provider_results=provider_results
            )
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return ImageSearchResult(
                images=[], total_count=0, page=request.page,
                per_page=request.per_page, has_next=False, has_prev=False,
                search_time=time.time() - start_time
            )
    
    async def _search_with_provider(self, provider: ImageSearchProvider, request: ImageSearchRequest) -> ImageSearchResult:
        """使用特定提供者搜索"""
        try:
            result = await provider.search(request)
            
            # 缓存搜索到的图片
            for image_info in result.images:
                cache_key = await self.cache_manager.is_cached(image_info)
                if not cache_key:
                    # 异步下载和缓存图片
                    asyncio.create_task(self._cache_image_from_provider(provider, image_info))
            
            return result
            
        except Exception as e:
            logger.error(f"Search failed for provider {provider.provider}: {e}")
            return ImageSearchResult(
                images=[], total_count=0, page=request.page,
                per_page=request.per_page, has_next=False, has_prev=False,
                search_time=0.0
            )
    
    async def _cache_image_from_provider(self, provider: ImageSearchProvider, image_info: ImageInfo):
        """从提供者缓存图片"""
        try:
            # 创建临时文件路径
            temp_path = Path(f"temp/{image_info.image_id}")
            
            # 下载图片
            download_result = await provider.download_image(image_info, temp_path)
            if not download_result.success:
                return
            
            # 读取图片数据
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            # 缓存图片
            await self.cache_manager.cache_image(image_info, image_data)
            
            # 清理临时文件
            temp_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to cache image {image_info.image_id}: {e}")
    
    async def generate_image(self, request: ImageGenerationRequest) -> ImageOperationResult:
        """生成图片"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # 获取生成提供者
            generation_providers = provider_registry.get_generation_providers()
            
            # 选择提供者
            provider = None
            for p in generation_providers:
                if p.provider == request.provider:
                    provider = p
                    break
            
            if not provider:
                return ImageOperationResult(
                    success=False,
                    message=f"Generation provider {request.provider} not available",
                    error_code="provider_not_found"
                )
            
            # 生成图片
            result = await provider.generate(request)
            
            # 如果生成成功，缓存图片
            if result.success and result.image_info:
                # 读取生成的图片
                with open(result.image_info.local_path, 'rb') as f:
                    image_data = f.read()
                
                # 缓存图片
                cache_key = await self.cache_manager.cache_image(result.image_info, image_data)
                logger.info(f"Generated image cached: {cache_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return ImageOperationResult(
                success=False,
                message=f"Image generation failed: {str(e)}",
                error_code="generation_error"
            )
    
    async def upload_image(self, request: ImageUploadRequest, file_data: bytes) -> ImageOperationResult:
        """上传图片"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # 获取本地存储提供者
            storage_providers = provider_registry.get_storage_providers()
            
            if not storage_providers:
                return ImageOperationResult(
                    success=False,
                    message="No storage provider available",
                    error_code="no_storage_provider"
                )
            
            # 使用第一个可用的存储提供者
            provider = storage_providers[0]
            
            # 上传图片
            result = await provider.upload(request, file_data)
            
            # 如果上传成功，缓存图片
            if result.success and result.image_info:
                cache_key = await self.cache_manager.cache_image(result.image_info, file_data)
                logger.info(f"Uploaded image cached: {cache_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            return ImageOperationResult(
                success=False,
                message=f"Image upload failed: {str(e)}",
                error_code="upload_error"
            )
    
    async def get_image(self, image_id: str) -> Optional[ImageInfo]:
        """获取图片信息"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # 首先尝试从缓存获取
            for cache_key, cache_info in self.cache_manager._cache_index.items():
                cached_result = await self.cache_manager.get_cached_image(cache_key)
                if cached_result:
                    image_info, _ = cached_result
                    if image_info.image_id == image_id:
                        return image_info
            
            # 如果缓存中没有，尝试从存储提供者获取
            storage_providers = provider_registry.get_storage_providers()
            for provider in storage_providers:
                image_info = await provider.get_image(image_id)
                if image_info:
                    return image_info
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get image {image_id}: {e}")
            return None
    
    async def process_image(self, image_id: str, options: ImageProcessingOptions) -> ImageOperationResult:
        """处理图片"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # 获取原始图片
            image_info = await self.get_image(image_id)
            if not image_info:
                return ImageOperationResult(
                    success=False,
                    message=f"Image {image_id} not found",
                    error_code="image_not_found"
                )
            
            # 获取缓存的图片文件
            cache_key = await self.cache_manager.is_cached(image_info)
            if not cache_key:
                return ImageOperationResult(
                    success=False,
                    message=f"Image {image_id} not in cache",
                    error_code="image_not_cached"
                )
            
            cached_result = await self.cache_manager.get_cached_image(cache_key)
            if not cached_result:
                return ImageOperationResult(
                    success=False,
                    message=f"Failed to load cached image {image_id}",
                    error_code="cache_load_error"
                )
            
            _, input_path = cached_result
            
            # 生成输出路径
            output_path = input_path.parent / f"processed_{input_path.name}"
            
            # 处理图片
            result = await self.processor.process_image(input_path, output_path, options)
            
            # 如果处理成功，缓存处理后的图片
            if result.success and result.image_info:
                with open(output_path, 'rb') as f:
                    processed_data = f.read()
                
                await self.cache_manager.cache_image(result.image_info, processed_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing failed for {image_id}: {e}")
            return ImageOperationResult(
                success=False,
                message=f"Image processing failed: {str(e)}",
                error_code="processing_error"
            )
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return await self.cache_manager.get_cache_stats()

    async def list_cached_images(self,
                                page: int = 1,
                                per_page: int = 20,
                                category: Optional[str] = None,
                                search: Optional[str] = None,
                                sort: str = "created_desc") -> Dict[str, Any]:
        """列出缓存的图片"""
        if not self.initialized:
            await self.initialize()

        try:
            # 获取所有缓存的图片
            all_images = []

            for cache_key, cache_info in self.cache_manager._cache_index.items():
                try:
                    # 检查是否过期
                    if cache_info.is_expired():
                        continue

                    # 加载图片元数据
                    image_info = await self.cache_manager._load_image_metadata(cache_key)
                    if not image_info:
                        continue

                    # 分类筛选
                    if category and image_info.source_type.value != category:
                        continue

                    # 搜索筛选
                    if search:
                        search_lower = search.lower()
                        if (search_lower not in (image_info.title or "").lower() and
                            search_lower not in (image_info.filename or "").lower() and
                            search_lower not in " ".join(image_info.tags or []).lower()):
                            continue

                    # 构建图片信息
                    image_data = {
                        "image_id": image_info.image_id,
                        "title": image_info.title,
                        "filename": image_info.filename,
                        "file_size": cache_info.file_size,
                        "source_type": image_info.source_type.value,
                        "provider": image_info.provider.value,
                        "created_at": cache_info.created_at,
                        "last_accessed": cache_info.last_accessed,
                        "access_count": cache_info.access_count,
                        "tags": image_info.tags or []
                    }

                    all_images.append(image_data)

                except Exception as e:
                    logger.warning(f"Failed to process cached image {cache_key}: {e}")
                    continue

            # 排序
            if sort == "created_desc":
                all_images.sort(key=lambda x: x["created_at"], reverse=True)
            elif sort == "created_asc":
                all_images.sort(key=lambda x: x["created_at"])
            elif sort == "accessed_desc":
                all_images.sort(key=lambda x: x["last_accessed"], reverse=True)
            elif sort == "size_desc":
                all_images.sort(key=lambda x: x["file_size"], reverse=True)
            elif sort == "size_asc":
                all_images.sort(key=lambda x: x["file_size"])

            # 分页
            total_count = len(all_images)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            page_images = all_images[start_idx:end_idx]

            return {
                "images": page_images,
                "total_count": total_count
            }

        except Exception as e:
            logger.error(f"Failed to list cached images: {e}")
            return {
                "images": [],
                "total_count": 0
            }

    async def delete_image(self, image_id: str) -> bool:
        """删除图片"""
        if not self.initialized:
            await self.initialize()

        try:
            # 查找对应的缓存键
            cache_key = None
            for key, cache_info in self.cache_manager._cache_index.items():
                try:
                    image_info = await self.cache_manager._load_image_metadata(key)
                    if image_info and image_info.image_id == image_id:
                        cache_key = key
                        break
                except Exception:
                    continue

            if not cache_key:
                return False

            # 从缓存中删除
            await self.cache_manager.remove_from_cache(cache_key)
            return True

        except Exception as e:
            logger.error(f"Failed to delete image {image_id}: {e}")
            return False

    async def get_thumbnail(self, image_id: str) -> Optional[str]:
        """获取图片缩略图路径"""
        if not self.initialized:
            await self.initialize()

        try:
            # 查找图片信息
            image_info = await self.get_image(image_id)
            if not image_info:
                return None

            # 生成缩略图路径
            thumbnail_dir = self.cache_manager.thumbnails_dir
            thumbnail_path = thumbnail_dir / f"{image_id}_thumb.jpg"

            # 如果缩略图已存在，直接返回
            if thumbnail_path.exists():
                return str(thumbnail_path)

            # 如果原图存在，生成缩略图
            if image_info.local_path and Path(image_info.local_path).exists():
                try:
                    from PIL import Image

                    # 创建缩略图目录
                    thumbnail_dir.mkdir(parents=True, exist_ok=True)

                    # 生成缩略图
                    with Image.open(image_info.local_path) as img:
                        # 如果是RGBA模式，转换为RGB以支持JPEG保存
                        if img.mode in ('RGBA', 'LA', 'P'):
                            # 创建白色背景
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            # 将原图粘贴到白色背景上
                            if img.mode == 'RGBA':
                                background.paste(img, mask=img.split()[-1])
                            else:
                                background.paste(img)
                            img = background

                        img.thumbnail((300, 200), Image.Resampling.LANCZOS)
                        img.save(str(thumbnail_path), "JPEG", quality=85)

                    return str(thumbnail_path)

                except ImportError:
                    logger.warning("PIL not available, cannot generate thumbnail")
                    return image_info.local_path
                except Exception as e:
                    logger.warning(f"Failed to generate thumbnail for {image_id}: {e}")
                    return image_info.local_path

            return None

        except Exception as e:
            logger.error(f"Failed to get thumbnail for {image_id}: {e}")
            return None
    
    async def cleanup_cache(self) -> Dict[str, int]:
        """清理缓存 - 由于图片永久有效，此方法仅返回统计信息"""
        return {
            'expired_removed': 0,
            'oversized_removed': 0,
            'total_removed': 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        provider_health = await provider_registry.health_check_all()
        cache_stats = await self.get_cache_stats()
        
        return {
            'service_initialized': self.initialized,
            'providers': provider_health,
            'cache': cache_stats,
            'status': 'healthy' if self.initialized else 'not_initialized'
        }

    # PPT集成相关方法
    async def generate_ppt_slide_image(self,
                                     slide_context: PPTSlideContext,
                                     provider: ImageProvider = ImageProvider.DALLE) -> ImageOperationResult:
        """为PPT幻灯片生成图片"""
        if not self.initialized:
            await self.initialize()

        try:
            # 使用PPT适配器创建生成请求
            generation_request = await self.ppt_adapter.create_generation_request(
                slide_context, provider
            )

            # 生成图片
            result = await self.generate_image(generation_request)

            if result.success:
                logger.info(f"Generated PPT slide image for slide {slide_context.page_number}")

            return result

        except Exception as e:
            logger.error(f"Failed to generate PPT slide image: {e}")
            return ImageOperationResult(
                success=False,
                message=f"Failed to generate slide image: {str(e)}",
                error_code="ppt_generation_error"
            )

    async def suggest_images_for_ppt_slide(self,
                                         slide_context: PPTSlideContext,
                                         max_suggestions: int = 5) -> List[ImageInfo]:
        """为PPT幻灯片推荐图片"""
        if not self.initialized:
            await self.initialize()

        try:
            # 构建搜索查询
            search_query = f"{slide_context.title} {slide_context.topic} {slide_context.scenario}"

            # 搜索相关图片
            search_request = ImageSearchRequest(
                query=search_query,
                per_page=max_suggestions * 2,  # 搜索更多以便筛选
                filters={
                    'scenario': slide_context.scenario,
                    'language': slide_context.language
                }
            )

            search_result = await self.search_images(search_request)

            # 使用智能匹配器进一步筛选和排序
            if search_result.images:
                content_text = f"{slide_context.title}\n{slide_context.content}"
                suggested_images = await self.matcher.suggest_images_for_content(
                    content_text, search_result.images, max_suggestions
                )
                return suggested_images

            return []

        except Exception as e:
            logger.error(f"Failed to suggest images for PPT slide: {e}")
            return []

    async def create_ppt_image_prompt(self, slide_context: PPTSlideContext) -> str:
        """为PPT幻灯片创建图片生成提示词"""
        return await self.ppt_adapter.generate_image_prompt(slide_context)
