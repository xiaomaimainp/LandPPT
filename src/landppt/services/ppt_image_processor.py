"""
PPT图片处理器
负责在PPT生成过程中处理图片相关逻辑，包括本地图片选择、网络图片搜索、AI图片生成
支持多图片处理，由AI决定每种来源的图片数量
"""

import logging
from typing import Dict, Any, Optional, List
import aiohttp
import json
import asyncio
from pathlib import Path

from .models.slide_image_info import (
    SlideImageInfo, SlideImagesCollection, SlideImageRequirements,
    ImageRequirement, ImageSource, ImagePurpose
)

logger = logging.getLogger(__name__)


class PPTImageProcessor:
    """PPT图片处理器"""
    
    def __init__(self, image_service=None, ai_provider=None):
        self.image_service = image_service
        self.ai_provider = ai_provider
        self._base_url = None
        # 搜索缓存，避免重复搜索
        self._search_cache = {}
        self._search_lock = asyncio.Lock()

    def _get_base_url(self) -> str:
        """获取基础URL，用于构建绝对图片链接"""
        if self._base_url is None:
            try:
                from .config_service import config_service
                app_config = config_service.get_config_by_category('app_config')
                self._base_url = app_config.get('base_url', 'http://localhost:8000')

                # 确保URL不以斜杠结尾
                if self._base_url.endswith('/'):
                    self._base_url = self._base_url[:-1]

            except Exception as e:
                logger.warning(f"无法获取基础URL配置，使用默认值: {e}")
                self._base_url = 'http://localhost:8000'

        return self._base_url

    def _build_absolute_image_url(self, relative_path: str) -> str:
        """构建绝对图片URL"""
        base_url = self._get_base_url()
        # 确保相对路径以斜杠开头
        if not relative_path.startswith('/'):
            relative_path = '/' + relative_path
        return f"{base_url}{relative_path}"

    async def process_slide_image(self, slide_data: Dict[str, Any], confirmed_requirements: Dict[str, Any],
                                 page_number: int, total_pages: int, template_html: str = "") -> Optional[SlideImagesCollection]:
        """处理幻灯片多图片生成/搜索/选择逻辑"""
        try:
            # 检查是否启用图片生成服务
            from .config_service import config_service
            image_config = config_service.get_config_by_category('image_service')

            enable_image_service = image_config.get('enable_image_service', False)
            if not enable_image_service:
                logger.debug("图片生成服务未启用")
                return None

            # 获取项目信息
            project_topic = confirmed_requirements.get('project_topic', '')
            project_scenario = confirmed_requirements.get('project_scenario', 'general')
            slide_title = slide_data.get('title', f'第{page_number}页')
            slide_content = slide_data.get('content_points', [])
            slide_content_text = '\n'.join(slide_content) if isinstance(slide_content, list) else str(slide_content)

            # 检查启用的图片来源
            enabled_sources = []
            if image_config.get('enable_local_images', True):
                enabled_sources.append(ImageSource.LOCAL)
            if image_config.get('enable_network_search', False):
                enabled_sources.append(ImageSource.NETWORK)
            if image_config.get('enable_ai_generation', False):
                enabled_sources.append(ImageSource.AI_GENERATED)

            if not enabled_sources:
                logger.info(f"第{page_number}页没有启用任何图片来源，跳过图片处理")
                return None

            # 让AI分析并决定图片需求（只考虑启用的来源）
            image_requirements = await self._ai_analyze_image_requirements(
                slide_data, project_topic, project_scenario, page_number, total_pages, template_html, enabled_sources, image_config
            )

            if not image_requirements or not image_requirements.requirements:
                logger.info(f"AI判断第{page_number}页不需要添加图片，跳过图片处理")
                return None

            logger.info(f"第{page_number}页图片需求: 总计{image_requirements.total_images_needed}张图片")

            # 创建图片集合
            images_collection = SlideImagesCollection(page_number=page_number, images=[])

            # 根据需求处理各种来源的图片
            for requirement in image_requirements.requirements:
                if requirement.source == ImageSource.LOCAL and ImageSource.LOCAL in enabled_sources:
                    local_images = await self._process_local_images(
                        requirement, project_topic, project_scenario, slide_title, slide_content_text
                    )
                    images_collection.images.extend(local_images)

                elif requirement.source == ImageSource.NETWORK and ImageSource.NETWORK in enabled_sources:
                    network_images = await self._process_network_images(
                        requirement, project_topic, project_scenario, slide_title, slide_content_text, image_config
                    )
                    images_collection.images.extend(network_images)

                elif requirement.source == ImageSource.AI_GENERATED and ImageSource.AI_GENERATED in enabled_sources:
                    ai_images = await self._process_ai_generated_images(
                        requirement, project_topic, project_scenario, slide_title, slide_content_text,
                        image_config, page_number, total_pages, template_html
                    )
                    images_collection.images.extend(ai_images)

            # 重新计算统计信息
            images_collection.__post_init__()

            if images_collection.total_count > 0:
                logger.info(f"第{page_number}页成功处理{images_collection.total_count}张图片: "
                          f"本地{images_collection.local_count}张, "
                          f"网络{images_collection.network_count}张, "
                          f"AI生成{images_collection.ai_generated_count}张")
                return images_collection
            else:
                logger.info(f"第{page_number}页未能获取到任何图片")
                return None

        except Exception as e:
            logger.error(f"处理幻灯片图片失败: {e}")
            return None

    async def _ai_analyze_image_requirements(self, slide_data: Dict[str, Any], project_topic: str,
                                           project_scenario: str, page_number: int, total_pages: int,
                                           template_html: str = "", enabled_sources: List[ImageSource] = None,
                                           image_config: Dict[str, Any] = None) -> Optional[SlideImageRequirements]:
        """使用AI分析幻灯片的图片需求"""
        if not self.ai_provider:
            logger.warning("AI提供者未初始化")
            return None

        # 提取幻灯片内容信息
        slide_title = slide_data.get('title', '')
        slide_content = slide_data.get('content_points', [])
        slide_content_text = '\n'.join(slide_content) if isinstance(slide_content, list) else str(slide_content)
        content_length = len(slide_content_text.strip())
        content_points_count = len(slide_content) if isinstance(slide_content, list) else 0

        # 处理启用的来源和配置限制
        if not enabled_sources:
            enabled_sources = [ImageSource.LOCAL, ImageSource.NETWORK, ImageSource.AI_GENERATED]

        if not image_config:
            image_config = {}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 获取各来源的最大数量限制
                max_local = image_config.get('max_local_images_per_slide', 2)
                max_network = image_config.get('max_network_images_per_slide', 2)
                max_ai = image_config.get('max_ai_images_per_slide', 1)
                max_total = image_config.get('max_total_images_per_slide', 3)

                # 构建启用来源的说明
                enabled_sources_desc = []
                if ImageSource.LOCAL in enabled_sources:
                    enabled_sources_desc.append(f"local: 本地图床中的图片，适合通用性图片 (最多{max_local}张)")
                if ImageSource.NETWORK in enabled_sources:
                    enabled_sources_desc.append(f"network: 网络搜索图片，适合特定主题的高质量图片 (最多{max_network}张)")
                if ImageSource.AI_GENERATED in enabled_sources:
                    enabled_sources_desc.append(f"ai_generated: AI生成图片，适合定制化、创意性图片 (最多{max_ai}张)")

                # 构建包含模板HTML的提示词
                template_context = ""
                if template_html.strip():
                    template_context = f"""
当前PPT模板HTML参考：
{template_html[:500]}...
"""

                prompt = f"""作为专业的PPT设计师，请分析以下幻灯片的图片需求，并决定需要多少张图片以及每种来源的分配。

【项目信息】
- 主题：{project_topic}
- 场景：{project_scenario}
- 当前页：{page_number}/{total_pages}

【幻灯片内容】
- 标题：{slide_title}
- 内容要点数量：{content_points_count}个
- 内容字数：{content_length}字
- 具体内容：
{slide_content_text}

{template_context}

【可用图片来源及限制】
{chr(10).join(enabled_sources_desc)}

【图片用途说明】
1. decoration: 装饰性图片，美化页面
2. illustration: 说明性图片，辅助理解内容
3. background: 背景图片，营造氛围
4. icon: 图标，简化表达
5. chart_support: 图表辅助，支持数据展示
6. content_visual: 内容可视化，直观展示概念

【分析要求】
请综合考虑以下因素来决定图片需求：
1. 内容复杂度：复杂内容需要更多说明性图片
2. 页面类型：封面页、章节页通常需要装饰性图片
3. 视觉平衡：文字密集的页面需要图片调节
4. 主题匹配：根据主题选择合适的图片来源
5. 设计风格：根据模板风格决定图片类型

【重要限制】
- 总图片数量不能超过{max_total}张
- 只能使用已启用的图片来源
- 每种来源都有数量限制，请严格遵守

请以JSON格式返回分析结果，格式如下：
{{
    "needs_images": true/false,
    "total_images": 数字,
    "requirements": [
        {{
            "source": "仅限已启用的来源",
            "count": 数字,
            "purpose": "decoration/illustration/background/icon/chart_support/content_visual",
            "description": "具体需求描述",
            "priority": 1-5
        }}
    ],
    "reasoning": "分析理由"
}}

【重要要求】：
- 如果不需要图片，设置needs_images为false，requirements为空数组
- 每种来源可以有多个需求项，支持不同用途
- 优先级1-5，5为最高优先级
- 严格遵守数量限制，避免页面过于拥挤
- 必须返回有效的JSON格式，不要添加任何解释文字
- 不要使用markdown代码块包装
- 确保所有字符串值都用双引号包围
- 确保布尔值使用true/false（小写）

请直接返回纯JSON格式的结果："""

                response = await self.ai_provider.text_completion(
                    prompt=prompt,
                    temperature=0.7
                )

                # 解析AI响应
                # 清理AI响应内容
                raw_content = response.content.strip()
                logger.debug(f"AI原始响应内容: {raw_content}")

                # 尝试提取JSON部分
                json_content = self._extract_json_from_response(raw_content)
                if not json_content:
                    logger.error(f"无法从AI响应中提取有效JSON: {raw_content}")
                    raise json.JSONDecodeError("无法提取有效JSON", raw_content, 0)

                result = json.loads(json_content)

                if not result.get('needs_images', False):
                    logger.info(f"AI判断第{page_number}页不需要图片")
                    return None

                # 创建需求对象
                requirements = SlideImageRequirements(page_number=page_number, requirements=[])

                for req_data in result.get('requirements', []):
                    requirement = ImageRequirement(
                        source=ImageSource(req_data['source']),
                        count=req_data['count'],
                        purpose=ImagePurpose(req_data['purpose']),
                        description=req_data['description'],
                        priority=req_data.get('priority', 1)
                    )
                    requirements.add_requirement(requirement)

                logger.info(f"AI分析第{page_number}页图片需求: {result.get('reasoning', '')}")
                return requirements

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"第{attempt + 1}次尝试解析AI图片需求分析结果失败: {e}")
                logger.debug(f"AI响应内容: {response.content}")
                if attempt < max_retries - 1:
                    logger.info(f"等待1秒后进行第{attempt + 2}次重试...")
                    import asyncio
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"所有{max_retries}次尝试都失败，放弃图片需求分析")
                    return None

            except Exception as e:
                logger.warning(f"第{attempt + 1}次尝试AI分析图片需求失败: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"等待1秒后进行第{attempt + 2}次重试...")
                    import asyncio
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"所有{max_retries}次尝试都失败，放弃图片需求分析")
                    return None

        # 如果所有重试都失败了
        logger.error("AI分析图片需求失败，已达到最大重试次数")
        return None

    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """从AI响应中提取JSON内容"""
        try:
            # 移除可能的markdown代码块标记
            content = content.strip()

            # 如果内容被```json包围，提取其中的内容
            if content.startswith('```json') and content.endswith('```'):
                content = content[7:-3].strip()
            elif content.startswith('```') and content.endswith('```'):
                content = content[3:-3].strip()

            # 查找第一个{和最后一个}
            start_idx = content.find('{')
            end_idx = content.rfind('}')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_content = content[start_idx:end_idx + 1]
                # 验证是否为有效JSON
                json.loads(json_content)
                return json_content

            # 如果直接是JSON格式
            json.loads(content)
            return content

        except (json.JSONDecodeError, ValueError):
            pass

        return None

    async def _process_local_images(self, requirement: ImageRequirement, project_topic: str,
                                  project_scenario: str, slide_title: str, slide_content: str) -> List[SlideImageInfo]:
        """处理本地图片需求"""
        images = []
        try:
            if not self.image_service:
                logger.warning("图片服务未初始化")
                return images

            # 获取本地图片库信息
            cache_stats = await self.image_service.get_cache_stats()
            total_images = 0
            if 'categories' in cache_stats:
                for _, count in cache_stats['categories'].items():
                    total_images += count

            if total_images == 0:
                logger.info("本地图片库为空，跳过本地图片选择")
                return images

            # 让AI生成搜索关键词
            search_keywords = await self._ai_generate_local_search_keywords(
                slide_title, slide_content, project_topic, project_scenario, requirement
            )

            if not search_keywords:
                logger.warning("无法生成本地搜索关键词")
                return images

            # 搜索并选择多张图片
            selected_images = await self._search_multiple_local_images(search_keywords, requirement.count)

            for image_id in selected_images:
                relative_url = f"/api/image/view/{image_id}"
                absolute_url = self._build_absolute_image_url(relative_url)

                # 获取图片详细信息
                image_info = await self._get_local_image_details(image_id)

                slide_image = SlideImageInfo(
                    image_id=image_id,
                    absolute_url=absolute_url,
                    source=ImageSource.LOCAL,
                    purpose=requirement.purpose,
                    content_description=requirement.description,
                    search_keywords=search_keywords,
                    alt_text=image_info.get('title', ''),
                    title=image_info.get('title', ''),
                    width=image_info.get('width'),
                    height=image_info.get('height'),
                    file_size=image_info.get('file_size'),
                    format=image_info.get('format')
                )
                images.append(slide_image)

            logger.info(f"成功选择{len(images)}张本地图片")
            return images

        except Exception as e:
            logger.error(f"处理本地图片失败: {e}")
            return images

    async def _process_network_images(self, requirement: ImageRequirement, project_topic: str,
                                    project_scenario: str, slide_title: str, slide_content: str,
                                    image_config: Dict[str, Any]) -> List[SlideImageInfo]:
        """处理网络图片需求"""
        images = []
        try:
            # 检查是否有可用的网络搜索提供商
            if not self._has_network_search_providers(image_config):
                logger.warning("没有配置可用的网络搜索提供商")
                return images

            # 让AI生成搜索关键词
            search_query = await self._ai_generate_search_query(
                slide_title, slide_content, project_topic, project_scenario, requirement
            )

            if not search_query:
                logger.warning("无法生成搜索关键词")
                return images

            network_images = await self._search_images_directly(search_query, requirement.count)

            # 下载网络图片到本地缓存文件夹
            for i, image_data in enumerate(network_images):
                try:
                    # 下载图片到本地缓存
                    cached_image_info = await self._download_network_image_to_cache(image_data, f"网络图片_{i+1}")

                    if cached_image_info:
                        slide_image = SlideImageInfo(
                            image_id=cached_image_info['image_id'],
                            absolute_url=cached_image_info['absolute_url'],
                            source=ImageSource.NETWORK,
                            purpose=requirement.purpose,
                            content_description=requirement.description,
                            search_keywords=search_query,
                            alt_text=image_data.get('tags', ''),
                            title=f"网络图片 {i+1}",
                            width=image_data.get('imageWidth'),
                            height=image_data.get('imageHeight'),
                            format=cached_image_info.get('format', 'jpg')
                        )
                        images.append(slide_image)
                        logger.info(f"网络图片缓存成功: {cached_image_info['absolute_url']}")
                    else:
                        logger.warning(f"网络图片缓存失败，跳过第{i+1}张图片")

                except Exception as e:
                    logger.error(f"处理第{i+1}张网络图片失败: {e}")
                    continue

            logger.info(f"成功获取{len(images)}张网络图片")
            return images

        except Exception as e:
            logger.error(f"处理网络图片失败: {e}")
            return images

    def _has_network_search_providers(self, image_config: Dict[str, Any]) -> bool:
        """检查是否有可用的网络搜索提供商"""
        try:
            # 获取默认网络搜索提供商配置
            from .config_service import get_config_service
            config_service = get_config_service()
            all_config = config_service.get_all_config()
            default_provider = all_config.get('default_network_search_provider', 'unsplash')

            # 检查默认提供商的API密钥是否配置
            if default_provider == 'unsplash':
                unsplash_key = image_config.get('unsplash_access_key')
                return bool(unsplash_key and unsplash_key.strip())
            elif default_provider == 'pixabay':
                pixabay_key = image_config.get('pixabay_api_key')
                return bool(pixabay_key and pixabay_key.strip())

            return False

        except Exception as e:
            logger.warning(f"Failed to check network search providers: {e}")
            # 降级：检查是否有任何配置的API密钥
            unsplash_key = image_config.get('unsplash_access_key')
            pixabay_key = image_config.get('pixabay_api_key')
            return bool((unsplash_key and unsplash_key.strip()) or (pixabay_key and pixabay_key.strip()))

    async def _search_images_with_service(self, query: str, count: int) -> List[Dict[str, Any]]:
        """使用图片服务搜索图片"""
        # 创建搜索缓存键
        search_key = f"{query}_{count}"

        # 检查缓存
        async with self._search_lock:
            if search_key in self._search_cache:
                logger.debug(f"使用缓存的搜索结果: {query}")
                return self._search_cache[search_key]

        try:
            # 检查图片服务是否可用
            if not self.image_service:
                logger.error("图片服务未初始化，无法使用图片服务搜索")
                return []

            image_service = self.image_service

            from .image.models import ImageSearchRequest

            # 创建搜索请求
            search_request = ImageSearchRequest(
                query=query,
                per_page=max(3, min(count * 2, 20)),  # 搜索更多以便筛选，确保>=3
                page=1
            )

            # 执行搜索
            search_result = await image_service.search_images(search_request)

            # 转换为旧格式以兼容现有代码
            images = []
            for image_info in search_result.images[:count]:
                image_data = {
                    'id': image_info.image_id,
                    'webformatURL': image_info.original_url,
                    'largeImageURL': image_info.original_url,
                    'tags': ', '.join([tag.name for tag in (image_info.tags or [])]),
                    'user': image_info.author or 'Unknown',
                    'pageURL': image_info.source_url or '',
                    'imageWidth': image_info.metadata.width if image_info.metadata else 0,
                    'imageHeight': image_info.metadata.height if image_info.metadata else 0
                }
                images.append(image_data)

            # 缓存结果
            async with self._search_lock:
                self._search_cache[search_key] = images
                # 限制缓存大小，避免内存泄漏
                if len(self._search_cache) > 50:
                    # 删除最旧的缓存项
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]

            return images

        except Exception as e:
            logger.error(f"使用图片服务搜索失败: {e}")
            return []

    async def _process_ai_generated_images(self, requirement: ImageRequirement, project_topic: str,
                                         project_scenario: str, slide_title: str, slide_content: str,
                                         image_config: Dict[str, Any], page_number: int, total_pages: int,
                                         template_html: str = "") -> List[SlideImageInfo]:
        """处理AI生成图片需求"""
        images = []
        try:
            if not self.image_service:
                logger.warning("图片服务未初始化")
                return images

            # 获取默认AI图片提供商
            default_provider = image_config.get('default_ai_image_provider', 'dalle')

            # 让AI决定图片尺寸（对于多张图片，使用相同尺寸保持一致性）
            width, height = await self._ai_decide_image_dimensions(
                slide_title, slide_content, project_topic, project_scenario, requirement
            )

            # 为每张图片生成不同的提示词
            for i in range(requirement.count):
                # 让AI生成图片提示词
                image_prompt = await self._ai_generate_image_prompt(
                    slide_title, slide_content, project_topic, project_scenario,
                    page_number, total_pages, template_html, requirement, i + 1
                )

                if not image_prompt:
                    logger.warning(f"无法生成第{i+1}张图片的提示词")
                    continue

                # 创建图片生成请求
                from .image.models import ImageGenerationRequest, ImageProvider

                # 解析提供商
                provider = ImageProvider.DALLE
                if default_provider == 'siliconflow':
                    provider = ImageProvider.SILICONFLOW
                elif default_provider == 'stable_diffusion':
                    provider = ImageProvider.STABLE_DIFFUSION

                generation_request = ImageGenerationRequest(
                    prompt=image_prompt,
                    provider=provider,
                    width=width,
                    height=height,
                    quality="standard"
                )

                # 生成图片
                result = await self.image_service.generate_image(generation_request)

                if result.success and result.image_info:
                    relative_url = f"/api/image/view/{result.image_info.image_id}"
                    absolute_url = self._build_absolute_image_url(relative_url)

                    slide_image = SlideImageInfo(
                        image_id=result.image_info.image_id,
                        absolute_url=absolute_url,
                        source=ImageSource.AI_GENERATED,
                        purpose=requirement.purpose,
                        content_description=requirement.description,
                        generation_prompt=image_prompt,
                        alt_text=f"AI生成图片 {i+1}",
                        title=f"AI生成图片 {i+1}",
                        width=width,
                        height=height,
                        format=getattr(result.image_info, 'format', 'png')
                    )
                    images.append(slide_image)
                    logger.info(f"AI生成第{i+1}张图片成功: {absolute_url}")
                else:
                    logger.error(f"AI生成第{i+1}张图片失败: {result.message}")

            logger.info(f"成功生成{len(images)}张AI图片")
            return images

        except Exception as e:
            logger.error(f"处理AI生成图片失败: {e}")
            return images







    async def _search_multiple_local_images(self, keywords: str, count: int) -> List[str]:
        """搜索多张本地图片"""
        try:
            if not self.image_service:
                return []

            # 获取所有本地图片
            gallery_result = await self.image_service.list_cached_images(page=1, per_page=100)
            if not gallery_result.get('images'):
                return []

            # 将关键词分割成列表
            keyword_list = keywords.lower().split()

            # 计算所有图片的匹配分数
            scored_images = []
            for img in gallery_result['images']:
                score = self._calculate_image_match_score(img, keyword_list)
                if score > 0:
                    scored_images.append((img.get('image_id'), score))

            # 按分数排序并选择前N张
            scored_images.sort(key=lambda x: x[1], reverse=True)
            selected_images = [img_id for img_id, _ in scored_images[:count]]

            logger.info(f"从{len(gallery_result['images'])}张本地图片中选择了{len(selected_images)}张")
            return selected_images

        except Exception as e:
            logger.error(f"搜索多张本地图片失败: {e}")
            return []



    async def _search_images_directly(self, query: str, count: int) -> List[Dict[str, Any]]:
        """使用配置的默认网络搜索提供商搜索图片"""
        # 创建搜索缓存键
        search_key = f"direct_{query}_{count}"

        # 检查缓存
        async with self._search_lock:
            if search_key in self._search_cache:
                logger.debug(f"使用缓存的直接搜索结果: {query}")
                return self._search_cache[search_key]

        try:
            from .image.models import ImageSearchRequest
            from .image.config.image_config import ImageServiceConfig

            # 获取配置
            config_manager = ImageServiceConfig()
            config = config_manager.get_config()

            # 获取默认网络搜索提供商配置
            from .config_service import get_config_service
            config_service = get_config_service()
            all_config = config_service.get_all_config()
            default_provider = all_config.get('default_network_search_provider', 'unsplash')

            logger.debug(f"使用默认网络搜索提供商: {default_provider}")

            # 根据配置的默认提供商创建相应的提供者
            provider = None
            if default_provider == 'pixabay':
                pixabay_config = config.get('pixabay', {})
                if not pixabay_config.get('api_key'):
                    logger.warning("Pixabay API key not configured")
                    return []
                from .image.providers.pixabay_provider import PixabaySearchProvider
                provider = PixabaySearchProvider(pixabay_config)
            else:  # 默认使用unsplash
                unsplash_config = config.get('unsplash', {})
                if not unsplash_config.get('api_key'):
                    logger.warning("Unsplash API key not configured")
                    return []
                from .image.providers.unsplash_provider import UnsplashSearchProvider
                provider = UnsplashSearchProvider(unsplash_config)

            if not provider:
                logger.error("无法创建网络搜索提供商")
                return []

            # 创建搜索请求
            # 注意：Pixabay API 要求 per_page 范围为 3-200
            search_request = ImageSearchRequest(
                query=query,
                per_page=max(3, min(count, 200)),  # 确保在有效范围内
                page=1
            )

            # 执行搜索
            search_result = await provider.search(search_request)

            # 转换为旧格式以兼容现有代码
            images = []
            for image_info in search_result.images[:count]:
                image_data = {
                    'id': image_info.image_id,
                    'webformatURL': image_info.original_url,
                    'largeImageURL': image_info.original_url,
                    'tags': ', '.join([tag.name for tag in (image_info.tags or [])]),
                    'user': image_info.author or 'Unknown',
                    'pageURL': image_info.source_url or '',
                    'imageWidth': image_info.metadata.width if image_info.metadata else 0,
                    'imageHeight': image_info.metadata.height if image_info.metadata else 0
                }
                images.append(image_data)

            # 缓存结果
            async with self._search_lock:
                self._search_cache[search_key] = images
                # 限制缓存大小
                if len(self._search_cache) > 50:
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]

            logger.debug(f"直接搜索获得{len(images)}张图片: {query}")
            return images

        except Exception as e:
            logger.error(f"直接搜索失败: {e}")
            return []

    async def _download_network_image_to_cache(self, image_data: Dict[str, Any], title: str) -> Optional[Dict[str, Any]]:
        """下载网络图片并上传到图床系统"""
        try:
            # 检查图片服务是否可用
            if not self.image_service:
                logger.error("图片服务未初始化，无法下载网络图片到缓存")
                return None

            # 获取图片URL
            image_url = (image_data.get('webformatURL') or 
                        image_data.get('url') or
                        image_data.get('largeImageURL') or
                        image_data.get('original_url'))

            if not image_url:
                logger.warning(f"网络图片URL为空，图片数据: {image_data}")
                return None

            # 下载图片数据
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data_bytes = await response.read()

                        # 获取文件扩展名
                        content_type = response.headers.get('content-type', 'image/jpeg')
                        if 'jpeg' in content_type or 'jpg' in content_type:
                            file_extension = 'jpg'
                        elif 'png' in content_type:
                            file_extension = 'png'
                        elif 'webp' in content_type:
                            file_extension = 'webp'
                        else:
                            file_extension = 'jpg'  # 默认

                        # 创建上传请求
                        from .image.models import ImageUploadRequest

                        upload_request = ImageUploadRequest(
                            filename=f"{title}.{file_extension}",
                            content_type=content_type,
                            file_size=len(image_data_bytes),
                            title=title,
                            description=f"从网络下载的图片: {image_data.get('tags', '')}",
                            tags=image_data.get('tags', '').split(', ') if image_data.get('tags') else [],
                            category="network_search"
                        )

                        # 上传到图床系统
                        result = await self.image_service.upload_image(upload_request, image_data_bytes)

                        if result.success and result.image_info:
                            # 构建图床API的绝对URL
                            relative_url = f"/api/image/view/{result.image_info.image_id}"
                            absolute_url = self._build_absolute_image_url(relative_url)

                            return {
                                'image_id': result.image_info.image_id,
                                'absolute_url': absolute_url,
                                'format': file_extension,
                                'width': image_data.get('imageWidth'),
                                'height': image_data.get('imageHeight')
                            }
                        else:
                            logger.error(f"上传网络图片到图床失败: {result.message}")
                            return None
                    else:
                        logger.error(f"下载网络图片失败，状态码: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"下载网络图片到图床失败: {e}")
            return None

    async def _get_local_image_details(self, image_id: str) -> Dict[str, Any]:
        """获取本地图片详细信息"""
        try:
            if not self.image_service:
                return {}

            # 这里可以调用图片服务的方法获取详细信息
            # 暂时返回基本信息
            return {
                'title': f'本地图片 {image_id}',
                'width': None,
                'height': None,
                'file_size': None,
                'format': None
            }
        except Exception as e:
            logger.error(f"获取本地图片详细信息失败: {e}")
            return {}

    def _calculate_image_match_score(self, img: Dict[str, Any], keyword_list: List[str]) -> int:
        """计算图片匹配分数"""
        score = 0

        # 处理标题、文件名、标签
        title = (img.get('title') or '').lower()
        filename = (img.get('filename') or '').lower()

        tags = img.get('tags', [])
        if tags and len(tags) > 0 and hasattr(tags[0], 'name'):
            tag_names = [tag.name.lower() for tag in tags]
        else:
            tag_names = [str(tag).lower() for tag in tags if tag]

        # 标题匹配（权重最高）
        for keyword in keyword_list:
            if keyword in title:
                score += 3

        # 标签匹配（权重中等）
        for keyword in keyword_list:
            for tag in tag_names:
                if keyword in tag or tag in keyword:
                    score += 2
                    break  # 每个关键词只匹配一次

        # 文件名匹配（权重较低）
        for keyword in keyword_list:
            if keyword in filename:
                score += 1

        return score

    async def _ai_generate_local_search_keywords(self, slide_title: str, slide_content: str,
                                               project_topic: str, project_scenario: str,
                                               requirement: ImageRequirement = None) -> Optional[str]:
        """使用AI生成本地图片搜索关键词"""
        try:
            if not self.ai_provider:
                logger.warning("AI提供者未初始化")
                return None

            # 构建需求信息
            requirement_info = ""
            if requirement:
                requirement_info = f"""
图片需求信息：
- 用途：{requirement.purpose.value}
- 描述：{requirement.description}
- 优先级：{requirement.priority}
"""

            prompt = f"""作为专业的图片搜索专家，请为以下PPT幻灯片生成本地图片搜索关键词。

项目主题：{project_topic}
项目场景：{project_scenario}
幻灯片标题：{slide_title}
幻灯片内容：{slide_content}
{requirement_info}

要求：
1. 生成3-5个中英文关键词，用空格分隔
2. 关键词要准确描述所需图片的内容和主题
3. 考虑项目场景和图片用途，选择合适的图片风格
4. 优先选择具体的视觉元素和概念
5. 适合在本地图片库中进行标题、描述、标签匹配

示例格式：商务 会议 图表 business chart
请只回复关键词，不要其他内容："""

            response = await self.ai_provider.text_completion(
                prompt=prompt,
                temperature=0.5
            )

            search_keywords = response.content.strip()
            logger.info(f"AI生成本地搜索关键词: {search_keywords}")
            return search_keywords

        except Exception as e:
            logger.error(f"AI生成本地搜索关键词失败: {e}")
            return None

    async def _search_local_images_by_keywords(self, keywords: str) -> Optional[str]:
        """使用关键词搜索本地图片，返回相关度最高的图片ID"""
        try:
            if not self.image_service:
                logger.warning("图片服务未初始化")
                return None

            # 将关键词分割成列表
            keyword_list = keywords.lower().split()

            # 获取所有本地图片
            gallery_result = await self.image_service.list_cached_images(page=1, per_page=100)
            if not gallery_result.get('images'):
                return None

            best_match = None
            best_score = 0

            for img in gallery_result['images']:
                score = 0
                image_id = img.get('image_id')

                # 计算匹配分数
                title = (img.get('title') or '').lower()
                filename = (img.get('filename') or '').lower()

                # 处理标签
                tags = img.get('tags', [])
                if tags and len(tags) > 0 and hasattr(tags[0], 'name'):
                    tag_names = [tag.name.lower() for tag in tags]
                else:
                    tag_names = [str(tag).lower() for tag in tags if tag]

                # 标题匹配（权重最高）
                title_matches = 0
                for keyword in keyword_list:
                    if keyword in title:
                        score += 3
                        title_matches += 1

                # 标签匹配（权重中等）
                tag_matches = 0
                for keyword in keyword_list:
                    for tag in tag_names:
                        if keyword in tag or tag in keyword:
                            score += 2
                            tag_matches += 1
                            break  # 每个关键词只匹配一次

                # 文件名匹配（权重较低）
                filename_matches = 0
                for keyword in keyword_list:
                    if keyword in filename:
                        score += 1
                        filename_matches += 1

                # 记录详细匹配信息
                if score > 0:
                    logger.debug(f"图片 {image_id} 匹配分数: {score} (标题:{title_matches}, 标签:{tag_matches}, 文件名:{filename_matches})")

                # 更新最佳匹配
                if score > best_score:
                    best_score = score
                    best_match = image_id
                    logger.debug(f"更新最佳匹配: {best_match}, 新分数: {best_score}")

            if best_match and best_score > 0:
                logger.info(f"找到最佳匹配图片: {best_match}, 分数: {best_score}")
                return best_match
            else:
                logger.info("未找到匹配的本地图片")
                return None

        except Exception as e:
            logger.error(f"本地图片搜索失败: {e}")
            return None



    def _truncate_search_query(self, query: str, max_length: int = 100) -> str:
        """截断搜索查询以符合API限制，保持单词完整性"""
        if not query or len(query) <= max_length:
            return query

        # 在最大长度内找到最后一个空格
        truncated = query[:max_length]
        last_space = truncated.rfind(' ')

        if last_space > 0:
            # 在最后一个空格处截断，保持单词完整
            return truncated[:last_space]
        else:
            # 如果没有空格，直接截断
            return truncated

    async def _ai_generate_search_query(self, slide_title: str, slide_content: str,
                                      project_topic: str, project_scenario: str,
                                      requirement: ImageRequirement = None) -> Optional[str]:
        """使用AI生成网络搜索关键词"""
        try:
            if not self.ai_provider:
                logger.warning("AI提供者未初始化")
                return None

            # 构建需求信息
            requirement_info = ""
            if requirement:
                requirement_info = f"""
图片需求信息：
- 用途：{requirement.purpose.value}
- 描述：{requirement.description}
- 优先级：{requirement.priority}
"""

            prompt = f"""作为专业的图片搜索专家，请为以下PPT幻灯片生成最佳的英文搜索关键词。

项目主题：{project_topic}
项目场景：{project_scenario}
幻灯片标题：{slide_title}
幻灯片内容：{slide_content}
{requirement_info}

要求：
1. 生成3-5个英文关键词，用空格分隔，总长度不超过80个字符
2. 关键词要准确描述所需图片的内容和用途
3. 考虑项目场景和图片用途，选择合适的图片风格
4. 避免过于抽象的词汇，优先选择具体的视觉元素
5. 确保关键词适合在Pixabay等图片库中搜索

示例格式：business meeting presentation chart
请只回复关键词，不要其他内容："""

            response = await self.ai_provider.text_completion(
                prompt=prompt,
                temperature=0.5
            )

            search_query = response.content.strip()

            # 截断查询以符合Pixabay API的100字符限制
            truncated_query = self._truncate_search_query(search_query, 100)

            if len(search_query) > 100:
                logger.warning(f"搜索关键词过长，已截断: '{search_query}' -> '{truncated_query}'")

            logger.info(f"AI生成搜索关键词: {truncated_query}")
            return truncated_query

        except Exception as e:
            logger.error(f"AI生成搜索关键词失败: {e}")
            return None

    async def _ai_decide_image_dimensions(self, slide_title: str, slide_content: str,
                                        project_topic: str, project_scenario: str,
                                        requirement: ImageRequirement = None) -> tuple:
        """使用AI决定图片的最佳尺寸"""
        try:
            if not self.ai_provider:
                logger.warning("AI提供者未初始化，使用默认尺寸")
                return (2048, 1152)  # 默认16:9横向

            # 构建需求信息
            requirement_info = ""
            if requirement:
                requirement_info = f"""
图片需求信息：
- 用途：{requirement.purpose.value}
- 描述：{requirement.description}
- 优先级：{requirement.priority}
"""

            prompt = f"""作为专业的PPT设计师，请根据以下信息为图片选择最佳的尺寸规格。

项目信息：
- 主题：{project_topic}
- 场景：{project_scenario}

幻灯片信息：
- 标题：{slide_title}
- 内容：{slide_content}

{requirement_info}

可选尺寸规格：
1. 2048x1152 (16:9横向) - 适合：横向展示、风景、全屏背景、宽屏演示
2. 1152x2048 (9:16竖向) - 适合：人物肖像、竖向图表、移动端展示
3. 2048x2048 (1:1正方形) - 适合：产品展示、图标、对称构图、社交媒体
4. 1920x1080 (16:9标准) - 适合：标准演示、视频截图、常规横向内容
5. 1080x1920 (9:16标准) - 适合：手机屏幕、竖向海报、故事模式

请根据内容特点、用途和展示效果选择最合适的尺寸。

要求：
1. 考虑内容的视觉特点（横向/竖向/方形更适合）
2. 考虑图片用途（背景/装饰/说明/图标等）
3. 考虑PPT演示的整体效果
4. 只回复对应的数字编号（1-5），不要其他内容"""

            response = await self.ai_provider.text_completion(
                prompt=prompt,
                temperature=0.3
            )

            choice = response.content.strip()

            # 解析AI的选择
            dimensions_map = {
                "1": (2048, 1152),  # 16:9横向
                "2": (1152, 2048),  # 9:16竖向
                "3": (1024, 1024),  # 1:1正方形
                "4": (1920, 1080),  # 16:9标准
                "5": (1080, 1920),  # 9:16标准
            }

            selected_dimensions = dimensions_map.get(choice, (2048, 1152))
            logger.info(f"AI选择图片尺寸: {selected_dimensions[0]}x{selected_dimensions[1]} (选项{choice})")

            return selected_dimensions

        except Exception as e:
            logger.error(f"AI决定图片尺寸失败: {e}")
            return (2048, 1152)  # 默认尺寸

    async def _ai_generate_image_prompt(self, slide_title: str, slide_content: str, project_topic: str,
                                      project_scenario: str, page_number: int, total_pages: int,
                                      template_html: str = "", requirement: ImageRequirement = None,
                                      image_index: int = 1) -> Optional[str]:
        """使用AI生成图片生成提示词"""
        try:
            if not self.ai_provider:
                logger.warning("AI提供者未初始化")
                return None

            # 构建包含模板HTML的提示词
            template_context = ""
            if template_html.strip():
                template_context = f"""
当前PPT模板HTML参考：
{template_html[:500]}...
"""

            # 构建需求信息
            requirement_info = ""
            if requirement:
                requirement_info = f"""
图片需求信息：
- 用途：{requirement.purpose.value}
- 描述：{requirement.description}
- 优先级：{requirement.priority}
- 当前是第{image_index}张图片
"""

            prompt = f"""作为专业的AI图片生成提示词专家，请为以下PPT幻灯片生成高质量的英文图片生成提示词。

项目信息：
- 主题：{project_topic}
- 场景：{project_scenario}
- 当前页：{page_number}/{total_pages}

幻灯片信息：
- 标题：{slide_title}
- 内容：{slide_content}

{requirement_info}
{template_context}

要求：
1. 生成详细的英文提示词，描述所需图片的视觉内容
2. 根据项目场景、图片用途和模板风格选择合适的风格
3. 包含具体的视觉元素描述，确保与模板风格协调
4. 确保图片适合PPT演示使用，符合指定用途
5. 考虑16:9或4:3的横向构图
6. 避免包含文字内容
7. 如果是多张图片中的一张，确保风格一致但内容有所区别

风格指导：
- business: professional, clean, modern office, corporate style
- technology: futuristic, digital, high-tech, innovation
- education: clear, informative, academic, learning environment
- general: clean, modern, professional presentation style

用途指导：
- decoration: 装饰性，美观、和谐、不抢夺主要内容焦点
- illustration: 说明性，直观、清晰、辅助理解内容
- background: 背景性，淡雅、不干扰前景内容
- icon: 图标性，简洁、符号化、易识别
- chart_support: 图表辅助，数据可视化、专业、清晰
- content_visual: 内容可视化，概念具象化、生动、准确

请生成一个完整的英文提示词（不超过120词），直接输出提示词，不要添加任何其他内容"""

            response = await self.ai_provider.text_completion(
                prompt=prompt,
                temperature=0.7
            )

            image_prompt = response.content.strip()
            logger.info(f"AI生成第{image_index}张图片提示词: {image_prompt}")
            return image_prompt

        except Exception as e:
            logger.error(f"AI生成图片提示词失败: {e}")
            return None

    async def _ai_should_add_image(self, slide_data: Dict[str, Any], project_topic: str,
                                 project_scenario: str, page_number: int, total_pages: int) -> bool:
        """使用AI判断该页是否需要或适合插入图片"""
        try:
            if not self.ai_provider:
                logger.warning("AI提供者未初始化，默认不添加图片")
                return False

            # 提取幻灯片内容信息
            slide_title = slide_data.get('title', '')
            slide_content = slide_data.get('content_points', [])
            slide_content_text = '\n'.join(slide_content) if isinstance(slide_content, list) else str(slide_content)
            content_length = len(slide_content_text.strip())
            content_points_count = len(slide_content) if isinstance(slide_content, list) else 0

            prompt = f"""作为专业的PPT设计师，请根据以下标准判断该幻灯片是否需要插入配图：

【项目信息】
- 主题：{project_topic}
- 场景：{project_scenario}
- 当前页：{page_number}/{total_pages}

【幻灯片内容】
- 标题：{slide_title}
- 内容要点数量：{content_points_count}个
- 内容字数：{content_length}字
- 具体内容：
{slide_content_text}

【判断标准】
请综合考虑以下因素：

1. 内容丰富程度：
   - 内容过少（<50字或<3个要点）：建议添加图片增强视觉效果
   - 内容适中（50-200字，3-6个要点）：根据内容性质判断
   - 内容丰富（>200字或>6个要点）：通常不需要额外图片

2. 理解难度：
   - 抽象概念、复杂流程、技术原理：需要图片辅助理解
   - 数据统计、对比分析：适合图表或图示
   - 简单陈述、常识内容：通常不需要图片

3. 内容类型：
   - 封面页、章节页：通常需要装饰性图片
   - 总结页、结论页：根据内容量判断
   - 纯文字列表：可能需要图片平衡版面
   - 已有充实内容的页面：通常不需要额外图片

4. 视觉平衡：
   - 页面显得空旷：需要图片填充
   - 文字密集：不建议添加图片
   - 版面协调：根据整体设计需要

请基于以上标准进行专业判断，只回复"是"或"否"："""

            response = await self.ai_provider.text_completion(
                prompt=prompt,
                temperature=0.7
            )
            # logger.info(f"AI判断是否需要图片的回复: {response.content}")
            decision = response.content.strip().lower()
            should_add = decision in ['是', 'yes', 'true', '需要', '适合']

            logger.info(f"AI判断第{page_number}页是否需要图片: {decision} -> {should_add}")
            return should_add

        except Exception as e:
            logger.error(f"AI判断是否添加图片失败: {e}")
            # 出错时默认不添加图片，避免不必要的处理
            return False
