"""
Enhanced PPT Service with real AI integration and project management
"""

import json
import re
import logging
import uuid
import asyncio
import time
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..api.models import (
    PPTGenerationRequest, PPTOutline, EnhancedPPTOutline,
    SlideContent, PPTProject, TodoBoard
)
from ..ai import get_ai_provider, AIMessage, MessageRole
from ..core.config import ai_config
from .ppt_service import PPTService
from .db_project_manager import DatabaseProjectManager
from .global_master_template_service import GlobalMasterTemplateService
from .deep_research_service import DEEPResearchService
from .research_report_generator import ResearchReportGenerator

# Configure logger for this module
logger = logging.getLogger(__name__)

class EnhancedPPTService(PPTService):
    """Enhanced PPT service with real AI integration and project management"""

    def __init__(self, provider_name: Optional[str] = None):
        super().__init__()
        self.provider_name = provider_name
        self.project_manager = DatabaseProjectManager()
        self.global_template_service = GlobalMasterTemplateService(provider_name)

        # 配置属性，用于summeryanyfile集成
        # 初始化配置（将在需要时实时更新）
        self.config = self._get_current_ai_config()

        # 初始化文件缓存管理器 - 设置缓存目录到项目根目录下的temp文件夹，每个模式的缓存分开管理
        try:
            from summeryanyfile.core.file_cache_manager import FileCacheManager
            import os
            from pathlib import Path

            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent.parent

            # 为不同模式创建分离的缓存目录
            base_cache_dir = project_root / "temp"

            # 创建分模式的缓存目录结构
            cache_dirs = {
                'summeryanyfile': base_cache_dir / "summeryanyfile_cache",
                'style_genes': base_cache_dir / "style_genes_cache",
                'ai_responses': base_cache_dir / "ai_responses_cache",
                'templates': base_cache_dir / "templates_cache"
            }

            # 确保所有缓存目录存在
            for cache_type, cache_path in cache_dirs.items():
                cache_path.mkdir(parents=True, exist_ok=True)

            # 初始化主要的文件缓存管理器（用于summeryanyfile）
            self.file_cache_manager = FileCacheManager(cache_dir=str(cache_dirs['summeryanyfile']))

            # 存储缓存目录配置供其他功能使用
            self.cache_dirs = cache_dirs

            logger.info(f"文件缓存管理器已初始化，分模式缓存目录: {cache_dirs}")
        except ImportError as e:
            logger.warning(f"无法导入文件缓存管理器: {e}")
            self.file_cache_manager = None
            self.cache_dirs = None

        # 初始化研究服务
        self.research_service = None
        self.report_generator = None
        self._initialize_research_services()

    @property
    def ai_provider(self):
        """Dynamically get AI provider to ensure latest config"""
        provider_name = self.provider_name or ai_config.default_ai_provider
        return get_ai_provider(provider_name)

    def _initialize_research_services(self):
        """Initialize research services if available"""
        try:
            self.research_service = DEEPResearchService()
            self.report_generator = ResearchReportGenerator()

            if self.research_service.is_available():
                logger.info("DEEP Research service initialized successfully")
            else:
                logger.warning("DEEP Research service not available - check Tavily API configuration")

        except Exception as e:
            logger.warning(f"Failed to initialize research services: {e}")
            self.research_service = None
            self.report_generator = None

    def reload_research_config(self):
        """Reload research service configuration"""
        if self.research_service:
            try:
                self.research_service.reload_config()
                logger.info("Research service configuration reloaded in EnhancedPPTService")
            except Exception as e:
                logger.warning(f"Failed to reload research service config: {e}")
                # If reload fails, reinitialize
                self._initialize_research_services()

    def _get_model_name_for_provider(self, provider_name: str) -> str:
        """根据provider获取正确的模型名称"""
        if provider_name == "openai":
            return ai_config.openai_model
        elif provider_name == "anthropic":
            return ai_config.anthropic_model
        elif provider_name == "ollama":
            return ai_config.ollama_model
        else:
            # 默认返回OpenAI模型
            return ai_config.openai_model

    def _get_current_ai_config(self):
        """获取当前最新的AI配置"""
        current_provider = self.provider_name or ai_config.default_ai_provider
        model_name = self._get_model_name_for_provider(current_provider)

        return {
            "llm_model": model_name,
            "llm_provider": current_provider,
            "temperature": getattr(ai_config, 'temperature', 0.7),
            "max_tokens": getattr(ai_config, 'max_tokens', 2000)
        }

    def update_ai_config(self):
        """更新AI配置到最新状态"""
        self.config = self._get_current_ai_config()
        logger.info(f"AI配置已更新: provider={self.config['llm_provider']}, model={self.config['llm_model']}")

    def _configure_summeryfile_api(self, generator):
        """配置summeryanyfile的API设置"""
        try:
            import os
            # 获取当前provider的配置
            current_provider = self.provider_name or ai_config.default_ai_provider
            provider_config = ai_config.get_provider_config(current_provider)

            # 设置通用配置参数
            if provider_config.get("max_tokens"):
                os.environ["MAX_TOKENS"] = str(provider_config["max_tokens"])
            if provider_config.get("temperature"):
                os.environ["TEMPERATURE"] = str(provider_config["temperature"])

            if current_provider == "openai":
                # 设置OpenAI API配置
                if provider_config.get("api_key"):
                    os.environ["OPENAI_API_KEY"] = provider_config["api_key"]
                if provider_config.get("base_url"):
                    os.environ["OPENAI_BASE_URL"] = provider_config["base_url"]

                logger.info(f"已配置summeryanyfile OpenAI API: model={provider_config.get('model')}, base_url={provider_config.get('base_url')}")

            elif current_provider == "anthropic":
                # 设置Anthropic API配置
                if provider_config.get("api_key"):
                    os.environ["ANTHROPIC_API_KEY"] = provider_config["api_key"]

                logger.info(f"已配置summeryanyfile Anthropic API: model={provider_config.get('model')}")

            elif current_provider == "google":
                # 设置Google API配置
                if provider_config.get("api_key"):
                    os.environ["GOOGLE_API_KEY"] = provider_config["api_key"]

                logger.info(f"已配置summeryanyfile Google API: model={provider_config.get('model')}")

            elif current_provider == "ollama":
                # 设置Ollama API配置
                if provider_config.get("base_url"):
                    os.environ["OLLAMA_BASE_URL"] = provider_config["base_url"]

                logger.info(f"已配置summeryanyfile Ollama API: model={provider_config.get('model')}, base_url={provider_config.get('base_url')}")

            logger.info(f"已配置summeryanyfile通用参数: max_tokens={provider_config.get('max_tokens')}, temperature={provider_config.get('temperature')}")

        except Exception as e:
            logger.warning(f"配置summeryanyfile API时出错: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取文件缓存统计信息

        Returns:
            缓存统计信息字典
        """
        if self.file_cache_manager:
            return self.file_cache_manager.get_cache_stats()
        else:
            return {"error": "缓存管理器未初始化"}

    def cleanup_cache(self):
        """清理过期的缓存条目"""
        # 清理summeryanyfile缓存
        if self.file_cache_manager:
            try:
                self.file_cache_manager.cleanup_expired_cache()
                logger.info("summeryanyfile缓存清理完成")
            except Exception as e:
                logger.error(f"summeryanyfile缓存清理失败: {e}")

        # 清理设计基因缓存
        self._cleanup_style_genes_cache()

        # 清理内存缓存
        if hasattr(self, '_cached_style_genes'):
            self._cached_style_genes.clear()
            logger.info("内存中的设计基因缓存已清理")

    def _cleanup_style_genes_cache(self, max_age_days: int = 7):
        """清理过期的设计基因缓存文件"""
        if not hasattr(self, 'cache_dirs') or not self.cache_dirs:
            return

        try:
            import json
            import time
            from pathlib import Path

            cache_dir = self.cache_dirs['style_genes']
            if not cache_dir.exists():
                return

            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            cleaned_count = 0

            for cache_file in cache_dir.glob("*_style_genes.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        created_at = cache_data.get('created_at', 0)

                    if current_time - created_at > max_age_seconds:
                        cache_file.unlink()
                        cleaned_count += 1
                        logger.debug(f"删除过期的设计基因缓存文件: {cache_file.name}")

                except Exception as e:
                    logger.warning(f"处理缓存文件 {cache_file} 时出错: {e}")

            if cleaned_count > 0:
                logger.info(f"设计基因缓存清理完成，删除了 {cleaned_count} 个过期文件")
            else:
                logger.info("设计基因缓存清理完成，没有过期文件需要删除")

        except Exception as e:
            logger.error(f"设计基因缓存清理失败: {e}")

    async def generate_outline(self, request: PPTGenerationRequest, page_count_settings: Dict[str, Any] = None) -> PPTOutline:
        """Generate PPT outline using real AI with optional DEEP research and page count settings"""
        try:
            research_context = ""
            research_report = None

            # Check if network mode is enabled and research service is available
            if request.network_mode and self.research_service and self.research_service.is_available():
                logger.info(f"Starting DEEP research for topic: {request.topic}")
                try:
                    # Conduct DEEP research
                    research_report = await self.research_service.conduct_deep_research(
                        topic=request.topic,
                        language=request.language
                    )

                    # Generate research context for outline generation
                    research_context = self._create_research_context(research_report)
                    logger.info("DEEP research completed successfully")

                    # Save research report if generator is available
                    if self.report_generator:
                        try:
                            report_path = self.report_generator.save_report_to_file(research_report)
                            logger.info(f"Research report saved to: {report_path}")
                        except Exception as save_error:
                            logger.warning(f"Failed to save research report: {save_error}")

                except Exception as research_error:
                    logger.warning(f"DEEP research failed, proceeding without research context: {research_error}")
                    research_context = ""

            # Create AI prompt for outline generation (with or without research context and page count settings)
            prompt = self._create_outline_prompt(request, research_context, page_count_settings)

            # Generate outline using AI
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=ai_config.max_tokens,
                temperature=ai_config.temperature
            )

            # Parse AI response to create structured outline
            outline = self._parse_ai_outline(response.content, request)

            # Add research metadata if available
            if research_report:
                outline.metadata["research_enhanced"] = True
                outline.metadata["research_duration"] = research_report.total_duration
                outline.metadata["research_sources"] = len(research_report.sources)

            # Add page count settings to metadata
            if page_count_settings:
                outline.metadata["page_count_settings"] = page_count_settings

            return outline

        except Exception as e:
            logger.error(f"Error generating AI outline: {str(e)}")
            # Fallback to original method
            return await super().generate_outline(request)
    
    async def generate_slide_content(self, slide_title: str, scenario: str, topic: str, language: str = "zh") -> str:
        """Generate slide content using AI"""
        try:
            prompt = self._create_slide_content_prompt(slide_title, scenario, topic, language)
            
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=min(ai_config.max_tokens, 500),  # Use smaller limit for slide content
                temperature=ai_config.temperature
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating slide content: {str(e)}")
            # Fallback to original method
            return self._generate_slide_content(topic, slide_title, scenario, language)
    
    async def enhance_content_with_ai(self, content: str, scenario: str, language: str = "zh") -> str:
        """Enhance existing content using AI"""
        try:
            prompt = self._create_enhancement_prompt(content, scenario, language)
            
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=min(ai_config.max_tokens, 800),  # Use smaller limit for content enhancement
                temperature=max(ai_config.temperature - 0.1, 0.1)  # Slightly lower temperature for enhancement
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error enhancing content: {str(e)}")
            return content  # Return original content if enhancement fails
    
    async def generate_speaker_notes(self, slide_content: str, scenario: str, language: str = "zh") -> str:
        """Generate speaker notes for a slide using AI"""
        try:
            prompt = self._create_speaker_notes_prompt(slide_content, scenario, language)
            
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=min(ai_config.max_tokens, 400),  # Use smaller limit for speaker notes
                temperature=ai_config.temperature
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating speaker notes: {str(e)}")
            return ""
    
    def _create_research_context(self, research_report) -> str:
        """Create comprehensive structured Markdown research context for outline generation"""
        if not research_report:
            return ""

        # 构建详细的结构化Markdown研究报告内容
        markdown_content = []

        # 标题和基本信息
        markdown_content.append(f"# {research_report.topic} - 深度研究报告")
        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")

        # 报告元信息
        markdown_content.append("## 📊 报告信息")
        markdown_content.append("")
        markdown_content.append(f"- **研究主题**: {research_report.topic}")
        markdown_content.append(f"- **报告语言**: {research_report.language}")
        markdown_content.append(f"- **生成时间**: {research_report.created_at.strftime('%Y年%m月%d日 %H:%M:%S')}")
        markdown_content.append(f"- **研究耗时**: {research_report.total_duration:.2f} 秒")
        markdown_content.append(f"- **研究步骤**: {len(research_report.steps)} 个")
        markdown_content.append(f"- **信息来源**: {len(research_report.sources)} 个")
        markdown_content.append("")

        # 执行摘要
        if research_report.executive_summary:
            markdown_content.append("## 📋 执行摘要")
            markdown_content.append("")
            markdown_content.append(research_report.executive_summary)
            markdown_content.append("")

        # 关键发现
        if research_report.key_findings:
            markdown_content.append("## 🔍 关键发现")
            markdown_content.append("")
            for i, finding in enumerate(research_report.key_findings, 1):
                markdown_content.append(f"### {i}. {finding}")
                markdown_content.append("")
            markdown_content.append("")

        # 建议与推荐
        if research_report.recommendations:
            markdown_content.append("## 💡 建议与推荐")
            markdown_content.append("")
            for i, recommendation in enumerate(research_report.recommendations, 1):
                markdown_content.append(f"### {i}. {recommendation}")
                markdown_content.append("")
            markdown_content.append("")

        # 详细研究过程和分析
        if research_report.steps:
            markdown_content.append("## 🔬 详细研究过程")
            markdown_content.append("")
            markdown_content.append("本节包含了完整的研究过程，每个步骤都包含了深入的分析和权威的信息来源。")
            markdown_content.append("")

            for step_num, step in enumerate(research_report.steps, 1):
                if step.completed and step.analysis:
                    markdown_content.append(f"### 步骤 {step_num}: {step.description}")
                    markdown_content.append("")
                    markdown_content.append(f"**🎯 研究目标**: {step.description}")
                    markdown_content.append("")
                    markdown_content.append(f"**🔍 搜索查询**: `{step.query}`")
                    markdown_content.append("")
                    markdown_content.append("**📊 研究状态**: ✅ 已完成")
                    markdown_content.append("")

                    # 详细分析结果
                    markdown_content.append("#### 📝 深度分析")
                    markdown_content.append("")
                    markdown_content.append(step.analysis)
                    markdown_content.append("")

                    # 详细的信息源列表
                    if step.results:
                        markdown_content.append("#### 📚 权威信息源")
                        markdown_content.append("")
                        markdown_content.append("以下是本研究步骤中使用的主要信息源，按相关性排序：")
                        markdown_content.append("")

                        for i, result in enumerate(step.results[:5], 1):  # 显示前5个来源
                            title = result.get('title', '未知标题')
                            url = result.get('url', '#')
                            content = result.get('content', '')
                            score = result.get('score', 0)
                            published_date = result.get('published_date', '')

                            markdown_content.append(f"**{i}. [{title}]({url})**")
                            if published_date:
                                markdown_content.append(f"   - 发布时间: {published_date}")
                            if score:
                                markdown_content.append(f"   - 相关性评分: {score:.2f}")
                            if content:
                                # 显示内容摘要（前300字符）
                                content_preview = content[:300] + "..." if len(content) > 300 else content
                                markdown_content.append(f"   - 内容摘要: {content_preview}")
                            markdown_content.append("")

                        # 如果还有更多来源，显示统计
                        if len(step.results) > 5:
                            markdown_content.append(f"*注：本步骤共找到 {len(step.results)} 个相关信息源，以上显示前5个最相关的来源。*")
                            markdown_content.append("")

                    markdown_content.append("---")
                    markdown_content.append("")

        # 综合分析和结论
        markdown_content.append("## 🎯 综合分析")
        markdown_content.append("")
        markdown_content.append("基于以上多维度的深度研究，我们可以得出以下综合性分析：")
        markdown_content.append("")

        # 重新整理关键发现作为综合分析的一部分
        if research_report.key_findings:
            markdown_content.append("### 核心洞察")
            markdown_content.append("")
            for finding in research_report.key_findings:
                markdown_content.append(f"- {finding}")
            markdown_content.append("")

        # 重新整理建议作为行动指南
        if research_report.recommendations:
            markdown_content.append("### 行动指南")
            markdown_content.append("")
            for recommendation in research_report.recommendations:
                markdown_content.append(f"- {recommendation}")
            markdown_content.append("")

        # 完整的信息源列表
        if research_report.sources:
            markdown_content.append("## 📖 完整信息源列表")
            markdown_content.append("")
            markdown_content.append("以下是本研究中使用的所有信息源：")
            markdown_content.append("")
            for i, source in enumerate(research_report.sources, 1):
                markdown_content.append(f"{i}. {source}")
            markdown_content.append("")

        # 研究方法说明
        markdown_content.append("## 🔬 研究方法说明")
        markdown_content.append("")
        markdown_content.append("本研究采用DEEP研究方法论：")
        markdown_content.append("")
        markdown_content.append("- **D (Define)**: 定义研究目标和范围")
        markdown_content.append("- **E (Explore)**: 探索多个信息维度和视角")
        markdown_content.append("- **E (Evaluate)**: 评估信息源的权威性和可靠性")
        markdown_content.append("- **P (Present)**: 呈现结构化的研究发现")
        markdown_content.append("")
        markdown_content.append(f"通过 {len(research_report.steps)} 个研究步骤，从 {len(research_report.sources)} 个权威信息源中")
        markdown_content.append(f"收集和分析了相关信息，耗时 {research_report.total_duration:.2f} 秒完成了这份综合性研究报告。")
        markdown_content.append("")

        # 结尾
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append("*本报告由 LandPPT DEEP Research 系统自动生成，基于多个权威信息源的深度分析。*")
        markdown_content.append("")
        markdown_content.append(f"*生成时间: {research_report.created_at.strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(markdown_content)


    def _create_outline_prompt(self, request: PPTGenerationRequest, research_context: str = "", page_count_settings: Dict[str, Any] = None) -> str:
        """Create prompt for AI outline generation - Enhanced with professional templates"""
        scenario_descriptions = {
            "general": "通用商务演示",
            "tourism": "旅游观光介绍",
            "education": "儿童科普教育",
            "analysis": "深入数据分析",
            "history": "历史文化主题",
            "technology": "科技技术展示",
            "business": "商业方案汇报"
        }

        scenario_desc = scenario_descriptions.get(request.scenario, "通用演示")

        # Handle page count requirements
        page_count_instruction = ""
        expected_page_count = 10  # Default page count

        if page_count_settings:
            page_count_mode = page_count_settings.get('mode', 'ai_decide')

            if page_count_mode == 'custom_range':
                min_pages = page_count_settings.get('min_pages', 8)
                max_pages = page_count_settings.get('max_pages', 15)
                page_count_instruction = f"- 页数要求：必须严格生成{min_pages}-{max_pages}页的PPT，确保页数在此范围内"
                expected_page_count = max_pages  # Use max for template
            elif page_count_mode == 'fixed':
                fixed_pages = page_count_settings.get('fixed_pages', 10)
                page_count_instruction = f"- 页数要求：必须生成恰好{fixed_pages}页的PPT"
                expected_page_count = fixed_pages
            else:
                page_count_instruction = "- 页数要求：根据内容复杂度自主决定合适的页数"
                expected_page_count = 12  # Default for AI decide
        else:
            page_count_instruction = "- 页数要求：根据内容复杂度自主决定合适的页数"
            expected_page_count = 12

        # Add research context if available
        research_section = ""
        if research_context:
            research_section = f"""

基于深度研究的背景信息：
{research_context}

请充分利用以上研究信息来丰富PPT内容，确保信息准确、权威、具有深度。"""

        # Get target audience and style information
        target_audience = getattr(request, 'target_audience', None) or '普通大众'
        ppt_style = getattr(request, 'ppt_style', None) or 'general'
        custom_style_prompt = getattr(request, 'custom_style_prompt', None)
        description = getattr(request, 'description', None)
        language = getattr(request, 'language', None)
        # Create style description
        style_descriptions = {
            "general": "通用商务风格，简洁专业",
            "conference": "学术会议风格，严谨正式",
            "custom": custom_style_prompt or "自定义风格"
        }
        style_desc = style_descriptions.get(ppt_style, "通用商务风格")

        # Add custom style prompt if provided (regardless of ppt_style)
        if custom_style_prompt and ppt_style != "custom":
            style_desc += f"，{custom_style_prompt}"

        if request.language == "zh":
            prompt = f"""你是一位专业的PPT大纲策划专家，请基于以下项目信息，生成一个**结构清晰、内容创意、专业严谨、格式规范的JSON格式PPT大纲**。

### 📌【项目信息】：
- **主题**：{request.topic}
- **应用场景**：{scenario_desc}
- **目标受众**：{target_audience}
- **PPT风格**：{style_desc}
- **特殊要求**：{request.requirements or '无'}
- **补充说明**：{description or '无'}
{research_section}

### 📄【页数要求】：
{page_count_instruction}

---

### 📋【大纲生成规则】：

1. **内容契合度要求**：
   - 所有幻灯片内容必须与上述项目信息严格匹配，确保主题明确、风格统一、内容相关。
   - 信息表达要专业可信，同时具有吸引力与传播力。

2. **页面结构规范**：
   - 必须包含以下结构：封面页、目录页、内容页（若干）、结论页。
   - 内容页应合理分层，逻辑清晰；封面和结论页需具备视觉冲击力或独特设计说明。

3. **内容点控制**：
   - 每页控制在3～6个内容要点之间。
   - 每个要点内容简洁清晰，**不超过50字符**。
   - 内容分布需均衡，避免信息堆积或重复。

4. **图表展示优化**：
   - 对适合可视化的信息，**建议并提供图表配置**，写入 `chart_config` 字段中。
   - 图表需明确类型（如柱状图、饼图、折线图等）、说明含义、配置样式及数据结构。

5. **语言风格与语境一致性**：
   - 使用统一语言（{language}），保持语境一致，适合目标受众理解与接受。

---

### 🧾【输出格式要求】：

请严格使用如下JSON格式进行输出，**使用代码块包裹，内容必须有效且结构完整**：

```json
{{
  "title": "专业且吸引人的PPT标题",
  "total_pages": {expected_page_count},
  "page_count_mode": "final",
  "slides": [
    {{
      "page_number": 1,
      "title": "页面标题",
      "content_points": ["要点1", "要点2", "要点3"],
      "slide_type": "title/content/conclusion",
      "type": "content",
      "description": "此页的简要说明与目的",
      "chart_config": {{
        "type": "bar",
        "data": {{
          "labels": ["示例A", "示例B", "示例C"],
          "datasets": [{{
            "label": "数据说明",
            "data": [80, 95, 70],
            "backgroundColor": ["#FF6B6B", "#4ECDC4", "#FFD93D"],
            "borderColor": ["#FF5252", "#26A69A", "#F4A261"],
            "borderWidth": 2
          }}]
        }},
        "options": {{
          "responsive": true,
          "plugins": {{
            "legend": {{"position": "top"}},
            "title": {{"display": true, "text": "图表标题"}}
          }},
          "scales": {{"y": {{"beginAtZero": true}}}}
        }}
      }}
    }}
  ],
  "metadata": {{
    "scenario": "{request.scenario}",
    "language": "{language}",
    "total_slides": {expected_page_count},
    "generated_with_ai": true,
    "enhanced_with_charts": true,
    "content_depth": "professional"
  }}
}}
"""
        else:
            # Add research context for English version
            english_research_section = ""
            if research_context:
                # Translate research context to English context
                english_research_section = f"""

Based on comprehensive research background:
{research_context}

Please utilize the above research information to enrich the PPT content, ensuring accuracy, authority, and depth."""

            # Handle page count requirements for English
            english_page_count_instruction = ""
            if page_count_settings:
                page_count_mode = page_count_settings.get('mode', 'ai_decide')

                if page_count_mode == 'custom_range':
                    min_pages = page_count_settings.get('min_pages', 8)
                    max_pages = page_count_settings.get('max_pages', 15)
                    english_page_count_instruction = f"- Page Count Requirement: Must strictly generate {min_pages}-{max_pages} pages PPT, ensure page count within this range"
                elif page_count_mode == 'fixed':
                    fixed_pages = page_count_settings.get('fixed_pages', 10)
                    english_page_count_instruction = f"- Page Count Requirement: Must generate exactly {fixed_pages} pages PPT"
                else:
                    english_page_count_instruction = "- Page Count Requirement: Decide appropriate page count based on content complexity (recommended 8-15 pages)"
            else:
                english_page_count_instruction = "- Page Count Requirement: Decide appropriate page count based on content complexity (recommended 8-15 pages)"

            prompt = f"""You are a **professional presentation outline designer**. Based on the following project details, please generate a **well-structured, creative, and professional JSON-format PowerPoint outline**.

### 📌【Project Details】:
- **Topic**: {request.topic}
- **Scenario**: {scenario_desc}
- **Target Audience**: {target_audience}
- **PPT Style**: {style_desc}
- **Special Requirements**: {request.requirements or 'None'}
- **Additional Notes**: {description or 'None'}
{research_section}

**Page Count Requirements:**
{english_page_count_instruction}

---

### 📋【Outline Generation Rules】:

1. **Content Relevance**:
   - All slide content must strictly align with the project details above.
   - Ensure the theme is clear, the tone is consistent, and the message is well-targeted.

2. **Slide Structure**:
   - The deck must include: **Title Slide**, **Agenda Slide**, **Content Slides**, and **Conclusion Slide**.
   - Title and Conclusion slides should be visually distinct or offer special design instructions.
   - Content slides must follow a logical and clear structure.

3. **Content Density Control**:
   - Each slide must contain **3–6 concise bullet points**.
   - Each point should be **no more than 50 characters**.
   - Distribute content evenly across slides to avoid overload or redundancy.

4. **Chart Suggestions**:
   - For any data, comparisons, or visual-friendly content, suggest a chart and include its configuration under `chart_config`.
   - Specify chart type (e.g., bar, pie, line), provide sample data, and chart options.

5. **Language & Tone**:
   - The entire outline should be in **{language}** and aligned with the communication preferences of the target audience.

---

### 🧾【Required Output Format】:

Please follow the exact JSON format below, and **wrap the result in a code block**. The JSON must be valid and complete.

```json
{{
  "title": "A compelling and professional PPT title",
  "total_pages": {expected_page_count},
  "page_count_mode": "final",
  "slides": [
    {{
      "page_number": 1,
      "title": "Slide Title",
      "content_points": ["Point 1", "Point 2", "Point 3"],
      "slide_type": "title/content/conclusion",
      "type": "content",
      "description": "Brief description of this slide",
      "chart_config": {{
        "type": "bar",
        "data": {{
          "labels": ["Metric A", "Metric B", "Metric C"],
          "datasets": [{{
            "label": "Performance Data",
            "data": [80, 95, 70],
            "backgroundColor": ["#FF6B6B", "#4ECDC4", "#FFD93D"],
            "borderColor": ["#FF5252", "#26A69A", "#F4A261"],
            "borderWidth": 2
          }}]
        }},
        "options": {{
          "responsive": true,
          "plugins": {{
            "legend": {{"position": "top"}},
            "title": {{"display": true, "text": "Chart Title"}}
          }},
          "scales": {{"y": {{"beginAtZero": true}}}}
        }}
      }}
    }}
  ],
  "metadata": {{
    "scenario": "{request.scenario}",
    "language": "{language}",
    "total_slides": {expected_page_count},
    "generated_with_ai": true,
    "enhanced_with_charts": true,
    "content_depth": "professional"
  }}
}}
"""
        
        return prompt
    
    def _create_slide_content_prompt(self, slide_title: str, scenario: str, topic: str, language: str) -> str:
        """Create prompt for slide content generation"""
        if language == "zh":
            prompt = f"""为PPT幻灯片生成内容：

PPT主题：{topic}
幻灯片标题：{slide_title}
场景类型：{scenario}

请生成这张幻灯片的具体内容，包括：
- 3-5个要点
- 每个要点的简短说明
- 适合{scenario}场景的语言风格

内容要求：
- 简洁明了，适合幻灯片展示
- 逻辑清晰，层次分明
- 语言专业但易懂
- 符合中文表达习惯

请直接输出内容，不需要额外说明。"""
        else:
            prompt = f"""Generate content for a PPT slide:

PPT Topic: {topic}
Slide Title: {slide_title}
Scenario: {scenario}

Please generate specific content for this slide, including:
- 3-5 key points
- Brief explanation for each point
- Language style appropriate for {scenario} scenario

Content requirements:
- Concise and suitable for slide presentation
- Clear logic and structure
- Professional but understandable language
- Appropriate for the target audience

Please output the content directly without additional explanations."""
        
        return prompt
    
    def _create_enhancement_prompt(self, content: str, scenario: str, language: str) -> str:
        """Create prompt for content enhancement"""
        if language == "zh":
            prompt = f"""请优化以下PPT内容，使其更适合{scenario}场景：

原始内容：
{content}

优化要求：
- 保持原有信息的完整性
- 改善语言表达和逻辑结构
- 增加适合{scenario}场景的专业术语
- 使内容更具吸引力和说服力
- 保持简洁明了的风格

请输出优化后的内容："""
        else:
            prompt = f"""Please enhance the following PPT content to make it more suitable for {scenario} scenario:

Original content:
{content}

Enhancement requirements:
- Maintain the completeness of original information
- Improve language expression and logical structure
- Add professional terminology suitable for {scenario} scenario
- Make content more attractive and persuasive
- Keep concise and clear style

Please output the enhanced content:"""
        
        return prompt
    
    def _create_speaker_notes_prompt(self, slide_content: str, scenario: str, language: str) -> str:
        """Create prompt for speaker notes generation"""
        if language == "zh":
            prompt = f"""为以下幻灯片内容生成演讲者备注：

幻灯片内容：
{slide_content}

场景：{scenario}

请生成演讲者备注，包括：
- 如何介绍这张幻灯片
- 需要强调的重点
- 可能的互动或问题
- 过渡到下一张幻灯片的方式

备注应该：
- 自然流畅，便于口头表达
- 提供比幻灯片更详细的信息
- 适合{scenario}场景的演讲风格
- 长度适中，不超过200字"""
        else:
            prompt = f"""Generate speaker notes for the following slide content:

Slide content:
{slide_content}

Scenario: {scenario}

Please generate speaker notes including:
- How to introduce this slide
- Key points to emphasize
- Possible interactions or questions
- Transition to the next slide

Notes should be:
- Natural and fluent for verbal expression
- Provide more detailed information than the slide
- Suitable for {scenario} scenario presentation style
- Moderate length, not exceeding 200 words"""
        
        return prompt
    
    def _parse_ai_outline(self, ai_response: str, request: PPTGenerationRequest) -> PPTOutline:
        """Parse AI response to create structured outline"""
        try:
            import json
            import re

            # 首先尝试解析JSON格式的响应
            json_str = None

            # 方法1: 尝试提取```json```代码块中的内容
            json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', ai_response, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
                logger.info("从```json```代码块中提取大纲JSON")
            else:
                # 方法2: 尝试提取```代码块中的内容（不带json标识）
                code_block_match = re.search(r'```\s*(\{.*?\})\s*```', ai_response, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                    logger.info("从```代码块中提取大纲JSON")
                else:
                    # 方法3: 尝试提取完整的JSON对象
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', ai_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        logger.info("使用正则表达式提取大纲JSON")

            if json_str:
                try:
                    # 清理JSON字符串
                    json_str = json_str.strip()
                    json_str = re.sub(r',\s*}', '}', json_str)  # 移除}前的多余逗号
                    json_str = re.sub(r',\s*]', ']', json_str)  # 移除]前的多余逗号

                    json_data = json.loads(json_str)
                    if 'slides' in json_data:
                        logger.info(f"Successfully parsed JSON outline with {len(json_data['slides'])} slides")

                        # 标准化slides格式以确保兼容性
                        standardized_data = self._standardize_outline_format(json_data)

                        # 确保metadata包含必要字段
                        metadata = standardized_data.get("metadata", {})
                        metadata.update({
                            "scenario": request.scenario,
                            "language": request.language,
                            "total_slides": len(standardized_data.get("slides", [])),
                            "generated_with_ai": True,
                            "ai_provider": self.provider_name
                        })

                        return PPTOutline(
                            title=standardized_data.get("title", request.topic),
                            slides=standardized_data.get("slides", []),
                            metadata=metadata
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse extracted JSON: {e}")
                    pass

            # Fallback: 解析文本格式的大纲
            logger.info("JSON解析失败，使用文本解析方式")
            lines = ai_response.strip().split('\n')
            title = request.topic
            slides = []

            # Extract title if present
            for line in lines:
                if line.startswith('标题：') or line.startswith('Title:'):
                    title = line.split('：', 1)[-1].split(':', 1)[-1].strip()
                    break

            # Parse slide structure
            page_number = 1

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for numbered items (slide structure)
                if re.match(r'^\d+\.', line):
                    # Extract slide title and description
                    parts = line.split(' - ', 1)
                    if len(parts) == 2:
                        slide_title = parts[0].split('.', 1)[1].strip()
                        slide_desc = parts[1].strip()
                    else:
                        slide_title = line.split('.', 1)[1].strip()
                        slide_desc = ""

                    # Determine slide type
                    slide_type = "content"
                    if "封面" in slide_title or "title" in slide_title.lower():
                        slide_type = "title"
                    elif "目录" in slide_title or "agenda" in slide_title.lower():
                        slide_type = "agenda"
                    elif "谢谢" in slide_title or "thank" in slide_title.lower():
                        slide_type = "thankyou"

                    # 使用与文件生成一致的格式
                    slides.append({
                        "page_number": page_number,
                        "title": slide_title,
                        "content_points": [slide_desc] if slide_desc else ["内容要点"],
                        "slide_type": slide_type,
                        "type": slide_type,  # 添加type字段以兼容不同的访问方式
                        "description": slide_desc
                    })

                    page_number += 1

            # If no slides were parsed, create default structure
            if not slides:
                slides = self._create_default_slides_compatible(title, request)

            return PPTOutline(
                title=title,
                slides=slides,
                metadata={
                    "scenario": request.scenario,
                    "language": request.language,
                    "total_slides": len(slides),
                    "generated_with_ai": True,
                    "ai_provider": self.provider_name
                }
            )

        except Exception as e:
            logger.error(f"Error parsing AI outline: {str(e)}")
            # Fallback to default structure
            return self._create_default_outline(request)
    
    def _create_default_slides(self, title: str, request: PPTGenerationRequest) -> List[Dict[str, Any]]:
        """Create default slide structure when AI parsing fails (legacy format)"""
        return [
            {
                "id": 1,
                "type": "title",
                "title": title,
                "subtitle": "专业演示" if request.language == "zh" else "Professional Presentation",
                "content": ""
            },
            {
                "id": 2,
                "type": "agenda",
                "title": "目录" if request.language == "zh" else "Agenda",
                "subtitle": "",
                "content": "• 主要内容概览\n• 核心要点分析\n• 总结与展望"
            },
            {
                "id": 3,
                "type": "content",
                "title": "主要内容" if request.language == "zh" else "Main Content",
                "subtitle": "",
                "content": f"• 关于{title}的核心要点\n• 详细分析和说明\n• 实际应用案例"
            },
            {
                "id": 4,
                "type": "thankyou",
                "title": "谢谢" if request.language == "zh" else "Thank You",
                "subtitle": "感谢聆听" if request.language == "zh" else "Thank you for your attention",
                "content": ""
            }
        ]

    def _create_default_slides_compatible(self, title: str, request: PPTGenerationRequest) -> List[Dict[str, Any]]:
        """Create default slide structure compatible with file generation format"""
        return [
            {
                "page_number": 1,
                "title": title,
                "content_points": ["专业演示" if request.language == "zh" else "Professional Presentation"],
                "slide_type": "title",
                "type": "title",
                "description": "PPT标题页"
            },
            {
                "page_number": 2,
                "title": "目录" if request.language == "zh" else "Agenda",
                "content_points": ["主要内容概览", "核心要点分析", "总结与展望"],
                "slide_type": "agenda",
                "type": "agenda",
                "description": "PPT目录页"
            },
            {
                "page_number": 3,
                "title": "主要内容" if request.language == "zh" else "Main Content",
                "content_points": [f"关于{title}的核心要点", "详细分析和说明", "实际应用案例"],
                "slide_type": "content",
                "type": "content",
                "description": "主要内容页"
            },
            {
                "page_number": 4,
                "title": "谢谢" if request.language == "zh" else "Thank You",
                "content_points": ["感谢聆听" if request.language == "zh" else "Thank you for your attention"],
                "slide_type": "thankyou",
                "type": "thankyou",
                "description": "PPT结束页"
            }
        ]
    
    def _create_default_outline(self, request: PPTGenerationRequest) -> PPTOutline:
        """Create default outline when AI generation fails"""
        slides = self._create_default_slides(request.topic, request)
        
        return PPTOutline(
            title=request.topic,
            slides=slides,
            metadata={
                "scenario": request.scenario,
                "language": request.language,
                "total_slides": len(slides),
                "generated_with_ai": False,
                "fallback_used": True
            }
        )

    # New project-based methods
    async def create_project_with_workflow(self, request: PPTGenerationRequest) -> PPTProject:
        """Create a new project with complete TODO workflow"""
        try:
            # Create project with TODO board
            project = await self.project_manager.create_project(request)

            # Start the workflow
            await self._execute_project_workflow(project.project_id, request)

            return project

        except Exception as e:
            logger.error(f"Error creating project with workflow: {str(e)}")
            raise

    async def _execute_project_workflow(self, project_id: str, request: PPTGenerationRequest):
        """Execute the complete project workflow with sequential subtask processing"""
        try:
            # Get project to check if requirements are confirmed
            project = await self.project_manager.get_project(project_id)
            if not project:
                raise ValueError("Project not found")

            # Only execute if requirements are confirmed
            if not project.confirmed_requirements:
                logger.info(f"Project {project_id} workflow waiting for requirements confirmation")
                return

            # Get TODO board to access stages and subtasks
            todo_board = await self.project_manager.get_todo_board(project_id)
            if not todo_board:
                raise ValueError("TODO board not found for project")

            # Process each stage sequentially (skip requirements confirmation stage)
            for stage_index, stage in enumerate(todo_board.stages):
                # Skip requirements confirmation stage as it's already done
                if stage.id == "requirements_confirmation":
                    continue

                logger.info(f"Starting stage {stage_index + 1}: {stage.name}")

                # Mark stage as running
                await self.project_manager.update_stage_status(
                    project_id, stage.id, "running", 0.0
                )

                # Execute the complete stage as a single task
                try:
                    stage_result = await self._execute_complete_stage(project_id, stage.id, request)
                except Exception as e:
                    logger.error(f"Error executing stage '{stage.name}': {str(e)}")
                    # Mark stage as failed but continue with next stage
                    await self.project_manager.update_stage_status(
                        project_id, stage.id, "failed", 0.0, {"error": str(e)}
                    )
                    continue
                # Wrap string result in dictionary for proper serialization
                result_dict = {"message": stage_result} if isinstance(stage_result, str) else stage_result
                await self.project_manager.update_stage_status(
                    project_id, stage.id, "completed", 100.0, result_dict
                )

                logger.info(f"Completed stage: {stage.name}")

            # Mark project as completed
            await self.project_manager.update_project_status(project_id, "completed")
            logger.info(f"Project workflow completed: {project_id}")

        except Exception as e:
            logger.error(f"Error in project workflow: {str(e)}")
            # Mark current stage as failed
            todo_board = await self.project_manager.get_todo_board(project_id)
            if todo_board and todo_board.current_stage_index < len(todo_board.stages):
                current_stage = todo_board.stages[todo_board.current_stage_index]
                await self.project_manager.update_stage_status(
                    project_id, current_stage.id, "failed", 0.0,
                    {"error": str(e)}
                )

    async def _execute_complete_stage(self, project_id: str, stage_id: str, request: PPTGenerationRequest):
        """Execute a complete stage as a single task"""
        try:
            logger.info(f"Executing complete stage: {stage_id}")

            # Get project and confirmed requirements
            project = await self.project_manager.get_project(project_id)
            if not project or not project.confirmed_requirements:
                raise ValueError("Project or confirmed requirements not found")

            confirmed_requirements = project.confirmed_requirements

            # Execute based on stage type
            if stage_id == "outline_generation":
                return await self._execute_outline_generation(project_id, confirmed_requirements, self._load_prompts_md_system_prompt())
            elif stage_id == "ppt_creation":
                return await self._execute_ppt_creation(project_id, confirmed_requirements, self._load_prompts_md_system_prompt())
            else:
                # Fallback for other stages
                return await self._execute_general_stage(project_id, stage_id, confirmed_requirements)

        except Exception as e:
            logger.error(f"Error executing complete stage '{stage_id}': {str(e)}")
            raise

    async def _execute_general_stage(self, project_id: str, stage_id: str, confirmed_requirements: Dict[str, Any]):
        """Execute a general stage task"""
        try:
            system_prompt = self._load_prompts_md_system_prompt()

            context = f"""
项目信息：
- 主题：{confirmed_requirements['topic']}
- 类型：{confirmed_requirements['type']}
- 其他说明：{confirmed_requirements.get('description', '无')}

当前阶段：{stage_id}

请根据以上信息完成当前阶段的任务。
"""

            response = await self.ai_provider.text_completion(
                prompt=context,
                system_prompt=system_prompt,
                max_tokens=ai_config.max_tokens,
                temperature=ai_config.temperature
            )

            return {"message": response.content}

        except Exception as e:
            logger.error(f"Error executing general stage '{stage_id}': {str(e)}")
            raise

    async def _complete_stage(self, project_id: str, stage_id: str,
                            request: PPTGenerationRequest) -> Dict[str, Any]:
        """Complete a stage and return its result"""
        try:
            if stage_id == "outline_generation":
                outline = await self.generate_outline(request)
                return {"outline": outline.dict()}

            elif stage_id == "theme_design":
                theme_config = await self._design_theme(request.scenario, request.language)
                return {"theme_config": theme_config}

            elif stage_id == "content_generation":
                # Get outline from previous stage
                project = await self.project_manager.get_project(project_id)
                if project and project.outline:
                    enhanced_slides = await self._generate_enhanced_content(project.outline, request)
                    return {"enhanced_slides": [slide.dict() for slide in enhanced_slides]}
                else:
                    # Fallback: generate basic outline first
                    outline = await self.generate_outline(request)
                    enhanced_slides = await self._generate_enhanced_content(outline, request)
                    return {"enhanced_slides": [slide.dict() for slide in enhanced_slides]}

            elif stage_id == "layout_verification":
                # Get slides from previous stage
                todo_board = await self.project_manager.get_todo_board(project_id)
                if todo_board:
                    for stage in todo_board.stages:
                        if stage.id == "content_generation" and stage.result:
                            slides_data = stage.result.get("enhanced_slides", [])
                            slides = [SlideContent(**slide_data) for slide_data in slides_data]
                            theme_config = {}
                            for s in todo_board.stages:
                                if s.id == "theme_design" and s.result:
                                    theme_config = s.result.get("theme_config", {})
                                    break
                            verified_slides = await self._verify_layout(slides, theme_config)
                            return {"verified_slides": [slide.dict() for slide in verified_slides]}
                return {"verified_slides": []}

            elif stage_id == "export_output":
                # Get verified slides and generate HTML
                todo_board = await self.project_manager.get_todo_board(project_id)
                if todo_board:
                    slides_data = []
                    theme_config = {}

                    for stage in todo_board.stages:
                        if stage.id == "layout_verification" and stage.result:
                            slides_data = stage.result.get("verified_slides", [])
                        elif stage.id == "theme_design" and stage.result:
                            theme_config = stage.result.get("theme_config", {})

                    if slides_data:
                        slides = [SlideContent(**slide_data) for slide_data in slides_data]
                        html_content = await self._generate_html_output(slides, theme_config)

                        # Update project with final results
                        project = await self.project_manager.get_project(project_id)
                        if project:
                            project.slides_html = html_content

                            # Save version
                            await self.project_manager.save_project_version(
                                project_id,
                                {
                                    "slides_html": html_content,
                                    "theme_config": theme_config
                                }
                            )

                        return {"html_content": html_content}

                return {"html_content": ""}

            else:
                return {"message": f"Stage {stage_id} completed"}

        except Exception as e:
            logger.error(f"Error completing stage '{stage_id}': {str(e)}")
            return {"error": str(e)}

    async def generate_outline_streaming(self, project_id: str):
        """Generate outline with streaming output"""
        try:
            project = await self.project_manager.get_project(project_id)
            if not project:
                raise ValueError("Project not found")

            # 检查是否已经有从文件生成的大纲
            file_generated_outline = None
            if project.confirmed_requirements and project.confirmed_requirements.get('file_generated_outline'):
                file_generated_outline = project.confirmed_requirements['file_generated_outline']
                logger.info(f"Project {project_id} has file-generated outline, using it")
            elif project.outline and project.outline.get('slides') and project.outline.get('metadata', {}).get('generated_with_summeryfile'):
                file_generated_outline = project.outline
                logger.info(f"Project {project_id} already has outline generated from file, using existing outline")

            if file_generated_outline:
                # 直接流式输出已有的大纲
                import json
                existing_outline = {
                    "title": file_generated_outline.get('title', project.topic),
                    "slides": file_generated_outline.get('slides', []),
                    "metadata": file_generated_outline.get('metadata', {})
                }

                # 确保元数据包含正确的标识
                if 'metadata' not in existing_outline:
                    existing_outline['metadata'] = {}
                existing_outline['metadata']['generated_with_summeryfile'] = True
                existing_outline['metadata']['generated_at'] = time.time()

                formatted_json = json.dumps(existing_outline, ensure_ascii=False, indent=2)

                # Stream the existing outline
                for i, char in enumerate(formatted_json):
                    yield f"data: {json.dumps({'content': char})}\n\n"
                    if i % 10 == 0:
                        await asyncio.sleep(0.02)  # Faster streaming for existing content

                # 保存大纲到项目中 - 直接保存结构化数据
                project.outline = existing_outline  # 直接保存结构化数据，而不是包装格式
                project.updated_at = time.time()

                # 立即保存到数据库
                try:
                    from .db_project_manager import DatabaseProjectManager
                    db_manager = DatabaseProjectManager()
                    save_success = await db_manager.save_project_outline(project_id, project.outline)

                    if save_success:
                        logger.info(f"✅ Successfully saved file-generated outline to database for project {project_id}")
                        # 同时更新内存中的项目管理器
                        self.project_manager.projects[project_id] = project
                    else:
                        logger.error(f"❌ Failed to save file-generated outline to database for project {project_id}")

                except Exception as save_error:
                    logger.error(f"❌ Exception while saving file-generated outline: {str(save_error)}")
                    import traceback
                    traceback.print_exc()

                # Update stage status
                await self._update_outline_generation_stage(project_id, existing_outline)
              # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            # Update project status to in_progress
            await self.project_manager.update_project_status(project_id, "in_progress")

            # Update TODO board stage status
            if project.todo_board:
                for stage in project.todo_board.stages:
                    if stage.id == "outline_generation":
                        stage.status = "running"
                        break

            import json
            # time 模块已经在文件顶部导入，不需要重复导入

            # 构建基于确认需求的提示词
            confirmed_requirements = project.confirmed_requirements or {}

            # 检查是否启用了联网模式并进行DEEP research
            research_context = ""
            network_mode = False
            if project.project_metadata and isinstance(project.project_metadata, dict):
                network_mode = project.project_metadata.get("network_mode", False)

            if network_mode and self.research_service and self.research_service.is_available():
                logger.info(f"🔍 Project {project_id} has network mode enabled, starting DEEP research for topic: {project.topic}")
                try:
                    # Conduct DEEP research
                    research_report = await self.research_service.conduct_deep_research(
                        topic=project.topic,
                        language="zh"  # Default to Chinese for now
                    )

                    # Generate structured Markdown research context
                    research_context = self._create_research_context(research_report)
                    logger.info(f"✅ DEEP research completed successfully for project {project_id}")

                    # Save research report if generator is available
                    if self.report_generator:
                        try:
                            report_path = self.report_generator.save_report_to_file(research_report)
                            logger.info(f"📄 Research report saved to: {report_path}")
                        except Exception as save_error:
                            logger.warning(f"Failed to save research report: {save_error}")

                    # 如果有研究内容，保存为临时文件并使用现有的文件处理流程
                    if research_context:
                        logger.info(f"🎯 Using research-based outline generation via file processing for project {project_id}")

                        # 保存研究内容为临时Markdown文件
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
                            temp_file.write(research_context)
                            temp_research_file = temp_file.name

                        logger.info(f"📄 Research content saved to temporary file: {temp_research_file}")
                        logger.info(f"📊 Research content stats: {len(research_context)} chars, {len(research_context.split())} words")

                        try:
                            # 创建文件大纲生成请求，使用现有的generate_outline_from_file方法
                            from ..api.models import FileOutlineGenerationRequest

                            file_request = FileOutlineGenerationRequest(
                                file_path=temp_research_file,
                                filename=f"research_{project.topic}.md",
                                topic=confirmed_requirements.get('topic', project.topic),
                                scenario=confirmed_requirements.get('type', project.scenario),
                                language="zh",
                                page_count_mode=confirmed_requirements.get('page_count_settings', {}).get('mode', 'ai_decide'),
                                min_pages=confirmed_requirements.get('page_count_settings', {}).get('min_pages', 8),
                                max_pages=confirmed_requirements.get('page_count_settings', {}).get('max_pages', 15),
                                fixed_pages=confirmed_requirements.get('page_count_settings', {}).get('fixed_pages', 10),
                                ppt_style=confirmed_requirements.get('ppt_style', 'general'),
                                custom_style_prompt=confirmed_requirements.get('custom_style_prompt'),
                                file_processing_mode="markitdown",  # 使用markitdown处理Markdown文件
                                content_analysis_depth="fast"  # 使用快速分析策略，适合研究报告处理
                            )

                            # 使用现有的文件处理方法生成大纲（采用快速分块策略）
                            logger.info(f"🚀 Using fast chunking strategy for research-based outline generation")
                            logger.info(f"📊 File processing config: mode={file_request.file_processing_mode}, depth={file_request.content_analysis_depth}")

                            outline_response = await self.generate_outline_from_file(file_request)

                            if outline_response.success and outline_response.outline:
                                structured_outline = outline_response.outline

                                # 添加研究增强标识
                                if 'metadata' not in structured_outline:
                                    structured_outline['metadata'] = {}
                                structured_outline['metadata']['research_enhanced'] = True
                                structured_outline['metadata']['research_duration'] = research_report.total_duration
                                structured_outline['metadata']['research_sources'] = len(research_report.sources)
                                structured_outline['metadata']['generated_from_research_file'] = True
                                structured_outline['metadata']['generated_at'] = time.time()

                                # 流式输出研究增强的大纲
                                formatted_json = json.dumps(structured_outline, ensure_ascii=False, indent=2)
                                for i, char in enumerate(formatted_json):
                                    yield f"data: {json.dumps({'content': char})}\n\n"
                                    if i % 10 == 0:
                                        await asyncio.sleep(0.05)

                                # 保存大纲
                                project.outline = structured_outline
                                project.updated_at = time.time()

                                # 保存到数据库
                                try:
                                    from .db_project_manager import DatabaseProjectManager
                                    db_manager = DatabaseProjectManager()
                                    save_success = await db_manager.save_project_outline(project_id, project.outline)
                                    if save_success:
                                        logger.info(f"✅ Successfully saved research-enhanced outline to database for project {project_id}")
                                        self.project_manager.projects[project_id] = project
                                    else:
                                        logger.error(f"❌ Failed to save research-enhanced outline to database for project {project_id}")
                                except Exception as save_error:
                                    logger.error(f"❌ Exception while saving research-enhanced outline: {str(save_error)}")

                                # 更新阶段状态
                                await self._update_outline_generation_stage(project_id, structured_outline)

                                # 发送完成信号
                                yield f"data: {json.dumps({'done': True})}\n\n"
                                return
                            else:
                                logger.warning(f"Failed to generate outline from research file, falling back to normal generation")

                        finally:
                            # 清理临时文件
                            try:
                                import os
                                if os.path.exists(temp_research_file):
                                    os.unlink(temp_research_file)
                                    logger.info(f"Cleaned up temporary research file: {temp_research_file}")
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to cleanup temporary research file: {cleanup_error}")

                except Exception as research_error:
                    logger.warning(f"DEEP research failed for project {project_id}, proceeding without research context: {research_error}")
                    research_context = ""
            else:
                if network_mode:
                    logger.warning(f"Project {project_id} has network mode enabled but research service is not available")
                else:
                    logger.info(f"Project {project_id} does not have network mode enabled")

            # 处理页数设置
            page_count_settings = confirmed_requirements.get('page_count_settings', {})
            page_count_mode = page_count_settings.get('mode', 'ai_decide')

            page_count_instruction = ""
            if page_count_mode == 'custom_range':
                min_pages = page_count_settings.get('min_pages', 8)
                max_pages = page_count_settings.get('max_pages', 15)
                page_count_instruction = f"- 页数要求：必须严格生成{min_pages}-{max_pages}页的PPT，确保页数在此范围内"
            elif page_count_mode == 'fixed':
                fixed_pages = page_count_settings.get('fixed_pages', 10)
                page_count_instruction = f"- 页数要求：必须生成恰好{fixed_pages}页的PPT"
            else:
                page_count_instruction = "- 页数要求：根据内容复杂度自主决定合适的页数（建议8-15页）"

            # Generate outline using AI - 使用字符串拼接避免f-string中的花括号冲突
            topic = confirmed_requirements.get('topic', project.topic)
            target_audience = confirmed_requirements.get('target_audience', '普通大众')
            ppt_style = confirmed_requirements.get('ppt_style', 'general')

            # Add research context if available
            research_section = ""
            if research_context:
                research_section = """

基于深度研究的背景信息：
""" + research_context + """

请充分利用以上研究信息来丰富PPT内容，确保信息准确、权威、具有深度。"""

            prompt = """
作为专业的PPT大纲生成助手，请为以下项目生成详细的PPT大纲。

项目信息：
- 主题：""" + topic + """
- 目标受众：""" + target_audience + """
- PPT风格：""" + ppt_style + """
""" + page_count_instruction + research_section + """

请严格按照以下JSON格式生成PPT大纲：

{
    "title": "PPT标题",
    "slides": [
        {
            "page_number": 1,
            "title": "页面标题",
            "content_points": ["要点1", "要点2", "要点3"],
            "slide_type": "title"
        },
        {
            "page_number": 2,
            "title": "页面标题",
            "content_points": ["要点1", "要点2", "要点3"],
            "slide_type": "content"
        }
    ]
}

slide_type可选值：
- "title": 标题页/封面页
- "content": 内容页
- "agenda": 目录页
- "thankyou": 结束页/感谢页

要求：
1. 必须返回有效的JSON格式
2. 严格遵守页数要求
3. 第一页通常是标题页，最后一页是感谢页
4. 每页至少包含2-5个内容要点
5. 页面标题要简洁明确
6. 内容要点要具体实用
7. 根据重点内容和技术亮点安排页面内容

请只返回JSON，使用```json```代码块包裹，不要包含其他文字说明。

示例格式：
```json
{
  "title": "PPT标题",
  "slides": [
    {
      "page_number": 1,
      "title": "页面标题",
      "content_points": ["要点1", "要点2"],
      "slide_type": "title"
    }
  ]
}
```
"""

            # Generate outline content directly without initial message
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=ai_config.max_tokens,
                temperature=ai_config.temperature
            )

            # Get the AI response content
            content = response.content.strip()

            # Import re for regex operations
            import re

            # 初始化structured_outline变量
            structured_outline = None

            # Try to parse as JSON first with validation and repair
            try:
                # Extract JSON from response if it contains extra text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    structured_outline = json.loads(json_str)
                else:
                    structured_outline = json.loads(content)

                # Validate and repair the JSON structure
                structured_outline = await self._validate_and_repair_outline_json(structured_outline, confirmed_requirements)

                # 验证页数是否符合要求
                actual_page_count = len(structured_outline.get('slides', []))
                if page_count_mode == 'custom_range':
                    min_pages = page_count_settings.get('min_pages', 8)
                    max_pages = page_count_settings.get('max_pages', 15)
                    if actual_page_count < min_pages or actual_page_count > max_pages:
                        logger.warning(f"Generated outline has {actual_page_count} pages, but expected {min_pages}-{max_pages} pages")
                        # 可以选择重新生成或调整，这里先记录警告
                elif page_count_mode == 'fixed':
                    fixed_pages = page_count_settings.get('fixed_pages', 10)
                    if actual_page_count != fixed_pages:
                        logger.warning(f"Generated outline has {actual_page_count} pages, but expected exactly {fixed_pages} pages")

                # 添加元数据
                structured_outline['metadata'] = {
                    'generated_with_summeryfile': False,
                    'page_count_settings': page_count_settings,
                    'actual_page_count': actual_page_count,
                    'generated_at': time.time()
                }

                # Format the JSON for display
                formatted_json = json.dumps(structured_outline, ensure_ascii=False, indent=2)

                # Stream the formatted JSON character by character
                for i, char in enumerate(formatted_json):
                    yield f"data: {json.dumps({'content': char})}\n\n"

                    # Add small delay for streaming effect
                    if i % 10 == 0:  # Every 10 characters
                        await asyncio.sleep(0.05)

                # Store the structured data directly
                project.outline = structured_outline  # 直接保存结构化数据
                project.updated_at = time.time()

                # 立即保存到数据库
                try:
                    from .db_project_manager import DatabaseProjectManager
                    db_manager = DatabaseProjectManager()
                    save_success = await db_manager.save_project_outline(project_id, project.outline)

                    if save_success:
                        logger.info(f"✅ Successfully saved outline to database during streaming for project {project_id}")
                        # 同时更新内存中的项目管理器
                        self.project_manager.projects[project_id] = project
                    else:
                        logger.error(f"❌ Failed to save outline to database during streaming for project {project_id}")

                except Exception as save_error:
                    logger.error(f"❌ Exception while saving outline during streaming: {str(save_error)}")
                    import traceback
                    traceback.print_exc()

                # 大纲生成完成后，立即生成母版模板（JSON解析成功的情况）
                await self._update_outline_generation_stage(project_id, structured_outline)

            except Exception as parse_error:
                logger.warning(f"Failed to parse AI response as JSON: {parse_error}")

                # Fallback: parse text-based outline and convert to JSON
                structured_outline = self._parse_outline_content(content, project)

                # 验证和修复fallback生成的大纲
                structured_outline = await self._validate_and_repair_outline_json(structured_outline, confirmed_requirements)

                # 添加元数据
                structured_outline['metadata'] = {
                    'generated_with_summeryfile': False,
                    'page_count_settings': page_count_settings,
                    'actual_page_count': len(structured_outline.get('slides', [])),
                    'generated_at': time.time()
                }

                formatted_json = json.dumps(structured_outline, ensure_ascii=False, indent=2)

                # Stream the formatted JSON
                for i, char in enumerate(formatted_json):
                    yield f"data: {json.dumps({'content': char})}\n\n"

                    if i % 10 == 0:
                        await asyncio.sleep(0.05)

                # Store the structured data
                project.outline = structured_outline  # 直接保存结构化数据
                project.updated_at = time.time()

                # 立即保存到数据库
                try:
                    from .db_project_manager import DatabaseProjectManager
                    db_manager = DatabaseProjectManager()
                    save_success = await db_manager.save_project_outline(project_id, project.outline)

                    if save_success:
                        logger.info(f"✅ Successfully saved fallback outline to database during streaming for project {project_id}")
                        # 同时更新内存中的项目管理器
                        self.project_manager.projects[project_id] = project
                    else:
                        logger.error(f"❌ Failed to save fallback outline to database during streaming for project {project_id}")

                except Exception as save_error:
                    logger.error(f"❌ Exception while saving fallback outline during streaming: {str(save_error)}")
                    import traceback
                    traceback.print_exc()

                # Update stage status - 确保structured_outline已定义
                if structured_outline is not None:
                    await self._update_outline_generation_stage(project_id, structured_outline)

                    # 检查是否已选择全局母版，如果没有则使用默认母版
                    logger.info(f"🎨 检查项目 {project_id} 的全局母版选择")
                    selected_template = await self._ensure_global_master_template_selected(project_id)

                    if selected_template:
                        logger.info(f"✅ 项目 {project_id} 已选择全局母版: {selected_template['template_name']}")
                    else:
                        logger.warning(f"⚠️ 项目 {project_id} 未找到可用的全局母版，将使用备用模板")
                    
                else:
                    # 如果structured_outline未定义，使用项目大纲数据
                    if project.outline and project.outline.get('slides'):
                        outline_data = {
                            "title": project.outline.get("title", project.topic),
                            "slides": project.outline.get("slides", [])
                        }
                        await self._update_outline_generation_stage(project_id, outline_data)

                    else:
                        # 创建默认的大纲数据
                        default_outline = {
                            "title": project.topic,
                            "slides": [
                                {
                                    "page_number": 1,
                                    "title": project.topic,
                                    "content_points": ["项目介绍"],
                                    "slide_type": "title"
                                }
                            ]
                        }
                        await self._update_outline_generation_stage(project_id, default_outline)
                # Send completion signal without message
                yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Error in outline streaming generation: {str(e)}")
            error_message = f'生成大纲时出现错误：{str(e)}'
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    async def _validate_and_repair_outline_json(self, outline_data: Dict[str, Any], confirmed_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """验证大纲JSON数据的正确性，如果有错误则调用AI修复，最多修复10次"""
        try:
            # 第一步：基本结构验证
            logger.info(f"outline_data: {outline_data}")
            validation_errors = self._validate_outline_structure(outline_data, confirmed_requirements)

            if not validation_errors:
                logger.info("大纲JSON验证通过，无需修复")
                return outline_data

            logger.warning(f"大纲JSON验证发现 {len(validation_errors)} 个错误，开始AI修复")

            # 第二步：调用AI修复，最多修复10次
            max_repair_attempts = 10
            current_attempt = 1

            while current_attempt <= max_repair_attempts:
                logger.info(f"第 {current_attempt} 次AI修复尝试")

                try:
                    repaired_outline = await self._repair_outline_with_ai(outline_data, validation_errors, confirmed_requirements)

                    # 验证修复后的结果
                    repair_validation_errors = self._validate_outline_structure(repaired_outline, confirmed_requirements)

                    if not repair_validation_errors:
                        logger.info(f"AI修复成功，第 {current_attempt} 次尝试通过验证")
                        return repaired_outline
                    else:
                        logger.warning(f"第 {current_attempt} 次AI修复后仍有 {len(repair_validation_errors)} 个错误")
                        validation_errors = repair_validation_errors
                        outline_data = repaired_outline

                except Exception as repair_error:
                    logger.error(f"第 {current_attempt} 次AI修复失败: {str(repair_error)}")

                current_attempt += 1

            # 如果10次修复都失败，直接输出JSON
            logger.warning("AI修复达到最大尝试次数(10次)，直接输出当前JSON")
            return outline_data

        except Exception as e:
            logger.error(f"验证和修复过程出错: {str(e)}")
            # 如果验证修复过程出错，直接输出原始JSON
            return outline_data

    def _validate_outline_structure(self, outline_data: Dict[str, Any], confirmed_requirements: Dict[str, Any]) -> List[str]:
        """验证大纲结构，返回错误列表"""
        errors = []

        try:
            # 1. 检查必需字段
            if not isinstance(outline_data, dict):
                errors.append("大纲数据必须是字典格式")
                return errors

            if 'slides' not in outline_data:
                errors.append("缺少必需字段: slides")
                return errors

            if 'title' not in outline_data:
                errors.append("缺少必需字段: title")

            # 2. 检查slides字段
            slides = outline_data.get('slides', [])
            if not isinstance(slides, list):
                errors.append("slides字段必须是列表格式")
                return errors

            if len(slides) == 0:
                errors.append("slides列表不能为空")
                return errors

            # 3. 检查页数要求
            page_count_settings = confirmed_requirements.get('page_count_settings', {})
            page_count_mode = page_count_settings.get('mode', 'ai_decide')
            actual_page_count = len(slides)

            if page_count_mode == 'custom_range':
                min_pages = page_count_settings.get('min_pages', 8)
                max_pages = page_count_settings.get('max_pages', 15)
                if actual_page_count < min_pages:
                    errors.append(f"页数不足：当前{actual_page_count}页，要求至少{min_pages}页")
                elif actual_page_count > max_pages:
                    errors.append(f"页数过多：当前{actual_page_count}页，要求最多{max_pages}页")
            elif page_count_mode == 'fixed':
                fixed_pages = page_count_settings.get('fixed_pages', 10)
                if actual_page_count != fixed_pages:
                    errors.append(f"页数不匹配：当前{actual_page_count}页，要求恰好{fixed_pages}页")

            # 4. 检查每个slide的结构
            for i, slide in enumerate(slides):
                slide_errors = self._validate_slide_structure(slide, i + 1)
                errors.extend(slide_errors)

            # 5. 检查页码连续性
            page_numbers = [slide.get('page_number', 0) for slide in slides]
            expected_numbers = list(range(1, len(slides) + 1))
            if page_numbers != expected_numbers:
                expected_str = ', '.join(map(str, expected_numbers))
                actual_str = ', '.join(map(str, page_numbers))
                errors.append(f"页码不连续：期望[{expected_str}]，实际[{actual_str}]")

            return errors

        except Exception as e:
            errors.append(f"验证过程出错: {str(e)}")
            return errors

    def _validate_slide_structure(self, slide: Dict[str, Any], slide_index: int) -> List[str]:
        """验证单个slide的结构"""
        errors = []

        try:
            if not isinstance(slide, dict):
                errors.append(f"第{slide_index}页：slide必须是字典格式")
                return errors

            # 检查必需字段
            required_fields = ['page_number', 'title', 'content_points', 'slide_type']
            for field in required_fields:
                if field not in slide:
                    errors.append(f"第{slide_index}页：缺少必需字段 {field}")

            # 检查字段类型和值
            if 'page_number' in slide:
                page_num = slide['page_number']
                if not isinstance(page_num, int) or page_num != slide_index:
                    errors.append(f"第{slide_index}页：page_number应为{slide_index}，实际为{page_num}")

            if 'title' in slide:
                title = slide['title']
                if not isinstance(title, str) or not title.strip():
                    errors.append(f"第{slide_index}页：title必须是非空字符串")

            if 'content_points' in slide:
                content_points = slide['content_points']
                if not isinstance(content_points, list):
                    errors.append(f"第{slide_index}页：content_points必须是列表格式")
                elif len(content_points) == 0:
                    errors.append(f"第{slide_index}页：content_points不能为空")
                else:
                    for j, point in enumerate(content_points):
                        if not isinstance(point, str) or not point.strip():
                            errors.append(f"第{slide_index}页：content_points[{j}]必须是非空字符串")

            if 'slide_type' in slide:
                slide_type = slide['slide_type']
                valid_types = ['title', 'content', 'agenda', 'thankyou']
                if slide_type not in valid_types:
                    valid_types_str = ', '.join(valid_types)
                    errors.append(f"第{slide_index}页：slide_type必须是{valid_types_str}中的一个，实际为{slide_type}")

            return errors

        except Exception as e:
            errors.append(f"第{slide_index}页验证出错: {str(e)}")
            return errors

    async def _repair_outline_with_ai(self, outline_data: Dict[str, Any], validation_errors: List[str], confirmed_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """使用AI修复大纲JSON数据"""
        try:
            # 构建修复提示词
            repair_prompt = self._build_repair_prompt(outline_data, validation_errors, confirmed_requirements)

            # 调用AI进行修复
            response = await self.ai_provider.text_completion(
                prompt=repair_prompt,
                max_tokens=ai_config.max_tokens,
                temperature=0.3  # 使用较低的温度以确保更准确的修复
            )

            # 解析AI返回的修复结果
            repaired_content = response.content.strip()

            # 提取JSON - 改进的提取逻辑
            import re
            json_str = None

            # 方法1: 尝试提取```json```代码块中的内容
            json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', repaired_content, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
                logger.info("从```json```代码块中提取JSON")
            else:
                # 方法2: 尝试提取```代码块中的内容（不带json标识）
                code_block_match = re.search(r'```\s*(\{.*?\})\s*```', repaired_content, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                    logger.info("从```代码块中提取JSON")
                else:
                    # 方法3: 尝试提取完整的JSON对象（非贪婪匹配）
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', repaired_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        logger.info("使用正则表达式提取JSON")
                    else:
                        # 方法4: 假设整个内容就是JSON
                        json_str = repaired_content
                        logger.info("将整个响应内容作为JSON处理")

            # 清理JSON字符串中的常见问题
            if json_str:
                # 移除可能的前后空白和换行
                json_str = json_str.strip()
                # 修复常见的JSON格式问题
                json_str = re.sub(r',\s*}', '}', json_str)  # 移除}前的多余逗号
                json_str = re.sub(r',\s*]', ']', json_str)  # 移除]前的多余逗号

            repaired_outline = json.loads(json_str)

            logger.info("AI修复完成，返回修复后的大纲")
            return repaired_outline

        except Exception as e:
            logger.error(f"AI修复过程出错: {str(e)}")
            # 如果AI修复失败，直接返回原始数据
            return outline_data

    def _build_repair_prompt(self, outline_data: Dict[str, Any], validation_errors: List[str], confirmed_requirements: Dict[str, Any]) -> str:
        """构建AI修复提示词"""

        # 获取页数要求
        page_count_settings = confirmed_requirements.get('page_count_settings', {})
        page_count_mode = page_count_settings.get('mode', 'ai_decide')

        page_count_instruction = ""
        if page_count_mode == 'custom_range':
            min_pages = page_count_settings.get('min_pages', 8)
            max_pages = page_count_settings.get('max_pages', 15)
            page_count_instruction = f"- 页数要求：必须严格生成{min_pages}-{max_pages}页的PPT"
        elif page_count_mode == 'fixed':
            fixed_pages = page_count_settings.get('fixed_pages', 10)
            page_count_instruction = f"- 页数要求：必须生成恰好{fixed_pages}页的PPT"
        else:
            page_count_instruction = "- 页数要求：根据内容复杂度自主决定合适的页数（建议8-15页）"

        import json
        current_json = json.dumps(outline_data, ensure_ascii=False, indent=2)
        errors_text = '\n'.join(["- " + str(error) for error in validation_errors])

        # 使用字符串拼接而不是f-string来避免花括号冲突
        prompt = """
作为专业的PPT大纲修复助手，请修复以下PPT大纲JSON数据中的错误。

项目信息：
- 主题：""" + confirmed_requirements.get('topic', '未知') + """
- 类型：""" + confirmed_requirements.get('type', '未知') + """
- 重点内容：""" + ', '.join(confirmed_requirements.get('focus_content', [])) + """
- 技术亮点：""" + ', '.join(confirmed_requirements.get('tech_highlights', [])) + """
- 目标受众：""" + confirmed_requirements.get('target_audience', '通用受众') + """
""" + page_count_instruction + """

当前大纲JSON（有错误）：
```json
""" + current_json + """
```

发现的错误：
""" + errors_text + """

请修复以上所有错误，并返回正确的JSON格式。修复要求：

1. **结构要求**：
   - 必须包含 "title" 和 "slides" 字段
   - slides必须是数组格式
   - 每个slide必须包含：page_number, title, content_points, slide_type

2. **页码要求**：
   - page_number必须从1开始连续递增
   - 与数组索引保持一致

3. **内容要求**：
   - title必须是非空字符串
   - content_points必须是非空字符串数组，每页至少2-5个要点
   - slide_type必须是以下之一：title, content, agenda, thankyou

4. **页数要求**：
   """ + page_count_instruction + """

5. **逻辑要求**：
   - 第一页通常是title类型
   - 最后一页通常是thankyou类型
   - 中间页面主要是content类型
   - 如果有目录页，通常在第二页

请只返回修复后的JSON，使用```json```代码块包裹，不要包含其他文字说明。确保JSON格式完全正确且符合所有要求。

示例格式：
```json
{
  "title": "修复后的标题",
  "slides": [
    {
      "page_number": 1,
      "title": "页面标题",
      "content_points": ["要点1", "要点2"],
      "slide_type": "title"
    }
  ]
}
```
"""

        return prompt




    async def _update_outline_generation_stage(self, project_id: str, outline_data: Dict[str, Any]):
        """Update outline generation stage status and save to database"""
        try:
            # 保存大纲到数据库
            from .db_project_manager import DatabaseProjectManager
            db_manager = DatabaseProjectManager()

            project = await self.project_manager.get_project(project_id)
            if not project:
                logger.error(f"❌ Project not found in memory for project {project_id}")
                return

            # 确保项目有outline数据，如果没有则使用传入的outline_data
            if not project.outline:
                logger.info(f"Project outline is None, setting outline from outline_data")
                project.outline = outline_data
                project.updated_at = time.time()

            # 保存大纲到数据库 - 使用outline_data而不是project.outline
            save_success = await db_manager.save_project_outline(project_id, outline_data)

            if save_success:
                logger.info(f"✅ Successfully saved outline to database for project {project_id}")

                # 验证保存是否成功
                saved_project = await db_manager.get_project(project_id)
                if saved_project and saved_project.outline:
                    saved_slides_count = len(saved_project.outline.get('slides', []))
                    logger.info(f"✅ Verified: outline saved with {saved_slides_count} slides")

                    # 确保内存中的项目数据也是最新的
                    project.outline = saved_project.outline
                    project.updated_at = saved_project.updated_at
                    logger.info(f"✅ Updated memory project with database outline")
                else:
                    logger.error(f"❌ Verification failed: outline not found in database")
            else:
                logger.error(f"❌ Failed to save outline to database for project {project_id}")

            # Update project manager
            await self.project_manager.update_project_status(project_id, "in_progress")

            # Update TODO board stage status
            if project.todo_board:
                for stage in project.todo_board.stages:
                    if stage.id == "outline_generation":
                        stage.status = "completed"
                        stage.result = {"outline_data": outline_data}
                        break

                # Update the project in project manager
                await self.project_manager.update_stage_status(
                    project_id, "outline_generation", "completed",
                    progress=100.0, result={"outline_data": outline_data}
                )

        except Exception as e:
            logger.error(f"Error updating outline generation stage: {str(e)}")
            import traceback
            traceback.print_exc()

    def _parse_outline_content(self, content: str, project: PPTProject) -> Dict[str, Any]:
        """Parse outline content to extract structured data for PPT generation"""
        try:
            import re
            import json

            # First try to parse the entire content as JSON
            try:
                json_data = json.loads(content)
                if isinstance(json_data, dict) and 'slides' in json_data:
                    logger.info(f"Successfully parsed complete JSON outline with {len(json_data['slides'])} slides")
                    # 标准化slides格式以确保兼容性
                    standardized_data = self._standardize_outline_format(json_data)
                    return standardized_data
            except json.JSONDecodeError:
                pass

            # 改进的JSON提取逻辑
            json_str = None

            # 方法1: 尝试提取```json```代码块中的内容
            json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
                logger.info("从```json```代码块中提取JSON")
            else:
                # 方法2: 尝试提取```代码块中的内容（不带json标识）
                code_block_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                    logger.info("从```代码块中提取JSON")
                else:
                    # 方法3: 尝试提取完整的JSON对象
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        logger.info("使用正则表达式提取JSON")

            if json_str:
                try:
                    # 清理JSON字符串
                    json_str = json_str.strip()
                    json_str = re.sub(r',\s*}', '}', json_str)  # 移除}前的多余逗号
                    json_str = re.sub(r',\s*]', ']', json_str)  # 移除]前的多余逗号

                    json_data = json.loads(json_str)
                    if 'slides' in json_data:
                        logger.info(f"Successfully extracted JSON from content with {len(json_data['slides'])} slides")
                        # 标准化slides格式以确保兼容性
                        standardized_data = self._standardize_outline_format(json_data)
                        return standardized_data
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse extracted JSON: {e}")
                    pass

            # Fallback: parse text-based outline
            lines = content.split('\n')
            slides = []
            current_slide = None
            slide_number = 1

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for slide titles (various formats)
                if (line.startswith('#') or
                    line.startswith('第') and ('页' in line or '章' in line) or
                    line.startswith('Page') or
                    re.match(r'^\d+[\.\)]\s*', line) or
                    line.endswith('：') or line.endswith(':')):

                    # Save previous slide
                    if current_slide:
                        slides.append(current_slide)

                    # Clean title
                    title = re.sub(r'^#+\s*', '', line)  # Remove markdown headers
                    title = re.sub(r'^第\d+[页章]\s*[：:]\s*', '', title)  # Remove "第X页："
                    title = re.sub(r'^Page\s*\d+\s*[：:]\s*', '', title, flags=re.IGNORECASE)  # Remove "Page X:"
                    title = re.sub(r'^\d+[\.\)]\s*', '', title)  # Remove "1. " or "1) "
                    title = title.rstrip('：:')  # Remove trailing colons

                    # Determine slide type
                    slide_type = "content"
                    if slide_number == 1 or '标题' in title or 'Title' in title or '封面' in title:
                        slide_type = "title"
                    elif '谢谢' in title or 'Thank' in title or '结束' in title or '总结' in title:
                        slide_type = "thankyou"
                    elif '目录' in title or 'Agenda' in title or '大纲' in title:
                        slide_type = "agenda"

                    current_slide = {
                        "page_number": slide_number,
                        "title": title or f"第{slide_number}页",
                        "content_points": [],
                        "slide_type": slide_type
                    }
                    slide_number += 1

                elif current_slide and (line.startswith('-') or line.startswith('•') or
                                      line.startswith('*') or re.match(r'^\d+[\.\)]\s*', line)):
                    # Content point
                    point = re.sub(r'^[-•*]\s*', '', line)
                    point = re.sub(r'^\d+[\.\)]\s*', '', point)
                    if point:
                        current_slide["content_points"].append(point)

                elif current_slide and line and not line.startswith('#'):
                    # Regular content line
                    current_slide["content_points"].append(line)

            # Add the last slide
            if current_slide:
                slides.append(current_slide)

            # If no slides were parsed, create a default structure
            if not slides:
                slides = self._create_default_slides_from_content(content, project)

            return {
                "title": project.topic,
                "slides": slides
            }

        except Exception as e:
            logger.error(f"Error parsing outline content: {str(e)}")
            # Return default structure
            return {
                "title": project.topic,
                "slides": self._create_default_slides_from_content(content, project)
            }

    def _standardize_outline_format(self, outline_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化大纲格式，确保slides字段的兼容性"""
        try:
            import re

            # 确保有基本结构
            if not isinstance(outline_data, dict):
                raise ValueError("Outline data must be a dictionary")

            title = outline_data.get("title", "PPT大纲")
            slides_data = outline_data.get("slides", [])
            metadata = outline_data.get("metadata", {})

            if not isinstance(slides_data, list):
                raise ValueError("Slides data must be a list")

            # 标准化每个slide的格式
            standardized_slides = []

            for i, slide in enumerate(slides_data):
                if not isinstance(slide, dict):
                    continue

                # 提取基本信息
                page_number = slide.get("page_number", i + 1)
                title_text = slide.get("title", f"第{page_number}页")

                # 处理content_points字段
                content_points = slide.get("content_points", [])
                if not isinstance(content_points, list):
                    content_points = []

                # 如果没有content_points，尝试从其他字段提取
                if not content_points:
                    # 尝试从content字段提取
                    content = slide.get("content", "")
                    if content:
                        lines = content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line:
                                # 移除bullet point符号
                                line = re.sub(r'^[•\-\*]\s*', '', line)
                                if line:
                                    content_points.append(line)

                    # 如果仍然没有，使用默认值
                    if not content_points:
                        content_points = ["内容要点"]

                # 处理slide_type字段
                slide_type = slide.get("slide_type", slide.get("type", "content"))

                # 智能识别slide类型
                title_lower = title_text.lower()
                if page_number == 1 or "标题" in title_lower or "title" in title_lower:
                    slide_type = "title"
                elif "目录" in title_lower or "agenda" in title_lower or "大纲" in title_lower:
                    slide_type = "agenda"
                elif "谢谢" in title_lower or "thank" in title_lower or "致谢" in title_lower:
                    slide_type = "thankyou"
                elif "总结" in title_lower or "结论" in title_lower or "conclusion" in title_lower:
                    slide_type = "conclusion"
                elif slide_type not in ["title", "content", "agenda", "thankyou", "conclusion"]:
                    slide_type = "content"

                # 构建标准化的slide
                standardized_slide = {
                    "page_number": page_number,
                    "title": title_text,
                    "content_points": content_points,
                    "slide_type": slide_type,
                    "type": slide_type,  # 添加type字段以兼容不同的访问方式
                    "description": slide.get("description", "")
                }

                # 保留chart_config如果存在
                if "chart_config" in slide and slide["chart_config"]:
                    standardized_slide["chart_config"] = slide["chart_config"]

                standardized_slides.append(standardized_slide)

            # 构建标准化的大纲
            standardized_outline = {
                "title": title,
                "slides": standardized_slides,
                "metadata": metadata
            }

            logger.info(f"Successfully standardized outline format: {title}, {len(standardized_slides)} slides")
            return standardized_outline

        except Exception as e:
            logger.error(f"Error standardizing outline format: {str(e)}")
            # 返回原始数据或默认结构
            if isinstance(outline_data, dict) and "slides" in outline_data:
                return outline_data
            else:
                return {
                    "title": "PPT大纲",
                    "slides": [
                        {
                            "page_number": 1,
                            "title": "标题页",
                            "content_points": ["演示标题"],
                            "slide_type": "title",
                            "type": "title",
                            "description": "PPT标题页"
                        }
                    ],
                    "metadata": {}
                }

    def _create_default_slides_from_content(self, content: str, project: PPTProject) -> List[Dict[str, Any]]:
        """Create default slides structure from content"""
        slides = [
            {
                "page_number": 1,
                "title": project.topic,
                "content_points": ["项目介绍", "主要内容", "核心特点"],
                "slide_type": "title"
            },
            {
                "page_number": 2,
                "title": "主要内容",
                "content_points": content.split('\n')[:5] if content else ["内容要点1", "内容要点2", "内容要点3"],
                "slide_type": "content"
            },
            {
                "page_number": 3,
                "title": "谢谢",
                "content_points": ["感谢聆听"],
                "slide_type": "thankyou"
            }
        ]
        return slides

    async def update_project_outline(self, project_id: str, outline_content: str) -> bool:
        """Update project outline content (expects JSON format)"""
        try:
            project = await self.project_manager.get_project(project_id)
            if not project:
                return False

            import json

            # Try to parse the content as JSON
            try:
                structured_outline = json.loads(outline_content)

                # Validate the JSON structure
                if 'slides' not in structured_outline:
                    raise ValueError("Invalid JSON structure: missing 'slides'")

                # 标准化大纲格式以确保兼容性
                structured_outline = self._standardize_outline_format(structured_outline)

                # Format the JSON for consistent display
                formatted_json = json.dumps(structured_outline, ensure_ascii=False, indent=2)

            except json.JSONDecodeError:
                # If not valid JSON, try to parse as text and convert to JSON
                structured_outline = self._parse_outline_content(outline_content, project)
                formatted_json = json.dumps(structured_outline, ensure_ascii=False, indent=2)

            # Update outline in the correct field
            if not project.outline:
                project.outline = {}
            project.outline["content"] = formatted_json  # Store formatted JSON
            project.outline["title"] = structured_outline.get("title", project.topic)
            project.outline["slides"] = structured_outline.get("slides", [])
            project.outline["updated_at"] = time.time()

            # 保存更新的大纲到数据库
            try:
                from .db_project_manager import DatabaseProjectManager
                db_manager = DatabaseProjectManager()
                save_success = await db_manager.save_project_outline(project_id, project.outline)

                if save_success:
                    logger.info(f"✅ Successfully saved updated outline to database for project {project_id}")
                else:
                    logger.error(f"❌ Failed to save updated outline to database for project {project_id}")

            except Exception as save_error:
                logger.error(f"❌ Exception while saving updated outline to database: {str(save_error)}")

            # Update TODO board stage result
            if project.todo_board:
                for stage in project.todo_board.stages:
                    if stage.id == "outline_generation":
                        if not stage.result:
                            stage.result = {}
                        stage.result["outline_content"] = formatted_json
                        break

            return True

        except Exception as e:
            logger.error(f"Error updating project outline: {str(e)}")
            return False

    async def confirm_project_outline(self, project_id: str) -> bool:
        """Confirm project outline and enable PPT generation"""
        try:
            project = await self.project_manager.get_project(project_id)
            if not project:
                return False

            # 确保大纲数据存在
            if not project.outline:
                logger.error(f"No outline found for project {project_id}")
                return False

            # 检查大纲是否包含slides数据
            if not project.outline.get('slides'):
                logger.error(f"No slides found in outline for project {project_id}")

                # 首先尝试从confirmed_requirements中的file_generated_outline恢复
                if (project.confirmed_requirements and
                    project.confirmed_requirements.get('file_generated_outline') and
                    isinstance(project.confirmed_requirements['file_generated_outline'], dict)):

                    file_outline = project.confirmed_requirements['file_generated_outline']
                    if file_outline.get('slides'):
                        logger.info(f"Restoring outline from file_generated_outline with {len(file_outline['slides'])} slides")
                        # 恢复完整的大纲数据，保留确认状态
                        project.outline = file_outline.copy()
                        project.outline["confirmed"] = True
                        project.outline["confirmed_at"] = time.time()
                    else:
                        logger.error(f"file_generated_outline does not contain slides data")
                        return False
                else:
                    # 尝试从数据库重新加载大纲
                    try:
                        from .db_project_manager import DatabaseProjectManager
                        db_manager = DatabaseProjectManager()
                        db_project = await db_manager.get_project(project_id)
                        if db_project and db_project.outline and db_project.outline.get('slides'):
                            project.outline = db_project.outline
                            logger.info(f"Reloaded outline from database for project {project_id}")
                        else:
                            logger.error(f"No valid outline found in database for project {project_id}")
                            return False
                    except Exception as reload_error:
                        logger.error(f"Failed to reload outline from database: {reload_error}")
                        return False

            # 保留原有的大纲数据，只添加确认状态
            project.outline["confirmed"] = True
            project.outline["confirmed_at"] = time.time()

            # 保存确认状态到数据库
            try:
                from .db_project_manager import DatabaseProjectManager
                db_manager = DatabaseProjectManager()
                save_success = await db_manager.save_project_outline(project_id, project.outline)

                if save_success:
                    logger.info(f"✅ Successfully saved outline confirmation to database for project {project_id}")
                else:
                    logger.error(f"❌ Failed to save outline confirmation to database for project {project_id}")

            except Exception as save_error:
                logger.error(f"❌ Exception while saving outline confirmation to database: {save_error}")

            # Update TODO board - mark outline as confirmed and enable PPT creation
            if project.todo_board:
                for stage in project.todo_board.stages:
                    if stage.id == "outline_generation":
                        stage.status = "completed"
                        if not stage.result:
                            stage.result = {}
                        stage.result["confirmed"] = True
                    elif stage.id == "ppt_creation":
                        stage.status = "pending"  # Enable PPT creation
                        break

            # Update project manager
            await self.project_manager.update_stage_status(
                project_id, "outline_generation", "completed",
                progress=100.0, result={"confirmed": True}
            )

            return True

        except Exception as e:
            logger.error(f"Error confirming project outline: {e}")
            return False

    async def generate_requirements_suggestions(self, project: PPTProject) -> Dict[str, Any]:
        """Generate AI suggestions for project requirements based on scenario and uploaded content"""
        try:
            # 读取上传的文件内容（如果有）
            file_content = ""
            if hasattr(project, 'uploaded_files') and project.uploaded_files:
                file_content = f"\n上传文件内容：{project.uploaded_files[:1000]}..."  # 限制长度

            prompt = f"""
作为专业的PPT生成助手，请根据以下项目信息生成个性化的需求确认建议：

项目场景：{project.scenario}
项目主题：{project.topic}
项目要求：{project.requirements or '无特殊要求'}
{file_content}

请深入分析项目内容，生成以下建议：

1. 建议的精确标题（基于主题优化，更具吸引力和专业性）
2. 展示类型选项（如：技术分享、产品介绍、学术报告等，3-5个选项）

要求：
- 标题要具体、专业、有吸引力
- 类型选项要贴合项目实际内容和场景
- 提供多样化的展示类型选择

请严格按照以下JSON格式返回，使用```json```代码块包裹：

```json
{{
    "suggested_topic": "具体的建议标题",
    "type_options": [
        "展示类型1",
        "展示类型2",
        "展示类型3",
        "展示类型4",
        "展示类型5"
    ]
}}
```
"""

            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=min(ai_config.max_tokens, 1500),  # Use smaller limit for suggestions
                temperature=min(ai_config.temperature + 0.1, 1.0)  # Slightly higher temperature for creativity
            )

            # Try to parse JSON response
            import json
            import re

            try:
                # 改进的JSON提取逻辑
                content = response.content.strip()
                json_str = None

                # 方法1: 尝试提取```json```代码块中的内容
                json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_block_match:
                    json_str = json_block_match.group(1)
                    logger.info("从```json```代码块中提取建议JSON")
                else:
                    # 方法2: 尝试提取```代码块中的内容（不带json标识）
                    code_block_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if code_block_match:
                        json_str = code_block_match.group(1)
                        logger.info("从```代码块中提取建议JSON")
                    else:
                        # 方法3: 尝试提取完整的JSON对象
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            logger.info("使用正则表达式提取建议JSON")
                        else:
                            # 方法4: 假设整个内容就是JSON
                            json_str = content
                            logger.info("将整个响应内容作为建议JSON处理")

                # 清理JSON字符串
                if json_str:
                    json_str = json_str.strip()
                    json_str = re.sub(r',\s*}', '}', json_str)  # 移除}前的多余逗号
                    json_str = re.sub(r',\s*]', ']', json_str)  # 移除]前的多余逗号

                suggestions = json.loads(json_str)

                # Validate required fields
                required_fields = ['suggested_topic', 'type_options']
                for field in required_fields:
                    if field not in suggestions:
                        raise ValueError(f"Missing required field: {field}")

                # Ensure type_options is a list
                if not isinstance(suggestions['type_options'], list):
                    suggestions['type_options'] = [suggestions['type_options']]

                return suggestions

            except Exception as parse_error:
                logger.warning(f"Failed to parse AI suggestions: {parse_error}")
                # Fallback to default suggestions based on project
                return self._get_default_suggestions(project)

        except Exception as e:
            logger.error(f"Error generating requirements suggestions: {e}")
            return self._get_default_suggestions(project)

    def _get_default_suggestions(self, project: PPTProject) -> Dict[str, Any]:
        """Get default suggestions when AI generation fails"""
        # Generate basic suggestions based on project scenario
        scenario_types = {
            "general": ["通用展示", "综合介绍", "概述报告", "基础展示"],
            "tourism": ["旅游推介", "景点介绍", "文化展示", "旅行规划"],
            "education": ["教学课件", "学术报告", "知识分享", "培训材料"],
            "analysis": ["数据分析", "研究报告", "分析总结", "调研展示"],
            "history": ["历史回顾", "文化传承", "时代变迁", "历史教育"],
            "technology": ["技术分享", "产品介绍", "创新展示", "技术方案"],
            "business": ["商业计划", "项目汇报", "业务介绍", "企业展示"]
        }

        # Get type options based on scenario
        type_options = scenario_types.get(project.scenario, scenario_types["general"])

        # Generate suggested topic based on original topic
        suggested_topic = f"{project.topic} - 专业展示"

        return {
            "suggested_topic": suggested_topic,
            "type_options": type_options
        }

    def _get_default_todo_structure(self, confirmed_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get default TODO structure based on confirmed requirements"""
        return {
            "stages": [
                {
                    "id": "outline_generation",
                    "name": "生成PPT大纲",
                    "description": "设计PPT整体结构与框架，规划各章节内容与关键点，确定核心优势和创新点的展示方式",
                    "subtasks": ["生成PPT大纲"]  # Single task, description is explanatory
                },
                {
                    "id": "ppt_creation",
                    "name": "制作PPT",
                    "description": "设计PPT封面与导航页，根据大纲制作各章节内容页面，添加视觉元素和图表美化PPT",
                    "subtasks": ["制作PPT"]  # Single task, description is explanatory
                }
            ]
        }

    async def _update_project_todo_board(self, project_id: str, todo_data: Dict[str, Any],
                                       confirmed_requirements: Dict[str, Any]):
        """Update project TODO board with custom stages (including requirements confirmation)"""
        try:
            from ..api.models import TodoStage, TodoBoard
            import time

            # Create complete stages including requirements confirmation
            stages = [
                TodoStage(
                    id="requirements_confirmation",
                    name="需求确认",
                    description="AI根据用户设定的场景和上传的文件内容提供补充信息用来确认用户的任务需求",
                    status="completed",  # This stage is completed when requirements are confirmed
                    progress=100.0,
                    subtasks=["需求确认完成"]
                )
            ]

            # Add custom stages from AI generation
            for stage_data in todo_data.get("stages", []):
                stage = TodoStage(
                    id=stage_data["id"],
                    name=stage_data["name"],
                    description=stage_data["description"],
                    subtasks=stage_data["subtasks"],
                    status="pending",  # Start as pending
                    progress=0.0
                )
                stages.append(stage)

            # Create custom TODO board
            todo_board = TodoBoard(
                task_id=project_id,
                title=confirmed_requirements['topic'],
                stages=stages
            )

            # Calculate correct overall progress
            completed_stages = sum(1 for s in stages if s.status == "completed")
            todo_board.overall_progress = (completed_stages / len(stages)) * 100

            # Set current stage index to the first non-completed stage
            todo_board.current_stage_index = 0
            for i, stage in enumerate(stages):
                if stage.status != "completed":
                    todo_board.current_stage_index = i
                    break

            # Update project manager
            self.project_manager.todo_boards[project_id] = todo_board

            # Update project with confirmed requirements
            project = await self.project_manager.get_project(project_id)
            if project:
                project.topic = confirmed_requirements['topic']
                project.requirements = f"""
类型：{confirmed_requirements['type']}
其他说明：{confirmed_requirements.get('description', '无')}
"""
                project.updated_at = time.time()

        except Exception as e:
            logger.error(f"Error updating project TODO board: {e}")
            raise

    async def confirm_requirements_and_update_workflow(self, project_id: str, confirmed_requirements: Dict[str, Any]) -> bool:
        """Confirm requirements and update the TODO board with complete workflow"""
        try:
            project = await self.project_manager.get_project(project_id)
            if not project:
                return False

            # Store confirmed requirements
            project.confirmed_requirements = confirmed_requirements
            project.status = "in_progress"
            project.updated_at = time.time()

            # 如果有文件生成的大纲，直接设置到项目的outline字段中
            file_generated_outline = confirmed_requirements.get('file_generated_outline')
            if file_generated_outline and isinstance(file_generated_outline, dict):
                logger.info(f"Setting file-generated outline to project {project_id}")
                project.outline = file_generated_outline
                project.updated_at = time.time()

            # Save confirmed requirements to database
            try:
                from .db_project_manager import DatabaseProjectManager
                db_manager = DatabaseProjectManager()

                # Update project status
                await db_manager.update_project_status(project_id, "in_progress")
                logger.info(f"Successfully updated project status in database for project {project_id}")

                # Save confirmed requirements to database
                await db_manager.save_confirmed_requirements(project_id, confirmed_requirements)
                logger.info(f"Successfully saved confirmed requirements to database for project {project_id}")

                # 如果有文件生成的大纲，也保存到数据库
                if file_generated_outline:
                    save_success = await db_manager.save_project_outline(project_id, file_generated_outline)
                    if save_success:
                        logger.info(f"✅ Successfully saved file-generated outline to database for project {project_id}")
                    else:
                        logger.error(f"❌ Failed to save file-generated outline to database for project {project_id}")

                # Update requirements confirmation stage to completed
                await db_manager.update_stage_status(
                    project_id,
                    "requirements_confirmation",
                    "completed",
                    100.0,
                    {"confirmed_at": time.time(), "requirements": confirmed_requirements}
                )
                logger.info(f"Successfully updated requirements confirmation stage to completed for project {project_id}")

            except Exception as save_error:
                logger.error(f"Failed to update project status or save requirements in database: {save_error}")
                import traceback
                traceback.print_exc()

            # Update TODO board with default workflow (无需AI生成) - 修复：添加await
            success = await self.project_manager.update_todo_board_with_confirmed_requirements(
                project_id, confirmed_requirements
            )

            # 不再启动后台工作流，让前端直接控制大纲生成
            return success

        except Exception as e:
            logger.error(f"Error confirming requirements: {e}")
            return False

    def _load_prompts_md_system_prompt(self) -> str:
        """Load system prompt from prompts.md file"""
        try:
            import os
            prompts_path = os.path.join(os.getcwd(), "prompts.md")
            if os.path.exists(prompts_path):
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract the main system prompt content
                    return content
            else:
                logger.warning("prompts.md file not found, using default system prompt")
                return self._get_default_ppt_system_prompt()
        except Exception as e:
            logger.error(f"Error loading prompts.md: {e}")
            return self._get_default_ppt_system_prompt()

    def _load_keynote_style_prompt(self) -> str:
        """Load keynote style prompt from keynote_style_prompt.md file"""
        try:
            import os
            keynote_path = os.path.join(os.getcwd(), "keynote_style_prompt.md")
            if os.path.exists(keynote_path):
                with open(keynote_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return content
            else:
                logger.warning("keynote_style_prompt.md file not found, using default keynote prompt")
                return self._get_default_keynote_prompt()
        except Exception as e:
            logger.error(f"Error loading keynote_style_prompt.md: {e}")
            return self._get_default_keynote_prompt()

    def _get_default_keynote_prompt(self) -> str:
        """Get default keynote style prompt"""
        return """
        请生成Apple风格的发布会PPT页面，具有以下特点：
        1. 黑色背景，简洁现代的设计
        2. 卡片式布局，突出重点信息
        3. 使用科技蓝或品牌色作为高亮色
        4. 大字号标题，清晰的视觉层级
        5. 响应式设计，支持多设备显示
        6. 使用Font Awesome图标和Chart.js图表
        7. 平滑的动画效果

        🌟 **特殊页面设计要求** 🌟：
        - **首页（title类型）**：必须设计得非常亮眼！使用Apple风格的渐变背景、动画效果、特殊装饰元素、超大字体标题、光效等，营造强烈的视觉冲击力
        - **结尾页（thankyou/conclusion类型）**：必须设计得令人印象深刻！使用Apple风格的特殊背景效果、发光文字、动态装饰、庆祝元素等，留下深刻的最后印象
        """

    def _get_style_prompt(self, confirmed_requirements: Dict[str, Any]) -> str:
        """Get style prompt based on confirmed requirements"""
        if not confirmed_requirements:
            return self._load_prompts_md_system_prompt()

        ppt_style = confirmed_requirements.get('ppt_style', 'general')

        if ppt_style == 'keynote':
            return self._load_keynote_style_prompt()
        elif ppt_style == 'custom':
            custom_prompt = confirmed_requirements.get('custom_style_prompt', '')
            if custom_prompt:
                return f"""
                请根据以下自定义风格要求生成PPT页面：

                {custom_prompt}

                请确保生成的HTML页面符合上述风格要求，同时保持良好的可读性和用户体验。
                """
            else:
                return self._load_prompts_md_system_prompt()
        else:
            # Default to general style (prompts.md)
            return self._load_prompts_md_system_prompt()

    def _get_default_ppt_system_prompt(self) -> str:
        """Get default PPT generation system prompt"""
        return """
你是一个专业的PPT设计师和HTML开发专家。

核心职责：
- 根据幻灯片内容生成高质量的HTML页面
- 确保设计风格的一致性和专业性
- 优化视觉表现和用户体验

设计原则：
- 内容驱动设计：让设计服务于内容表达
- 视觉层次清晰：合理运用字体、颜色、间距建立信息层级
- 用户体验优先：确保信息传达效率和阅读舒适度
- 创意与一致性平衡：在保持风格一致性的前提下展现创意
"""

    async def _execute_outline_generation(self, project_id: str, confirmed_requirements: Dict[str, Any], system_prompt: str) -> str:
        """Execute outline generation as a complete task"""
        try:
            # 处理页数设置
            page_count_settings = confirmed_requirements.get('page_count_settings', {})
            page_count_mode = page_count_settings.get('mode', 'ai_decide')

            page_count_instruction = ""
            expected_page_count = None  # Track expected page count for validation

            if page_count_mode == 'custom_range':
                min_pages = page_count_settings.get('min_pages', 8)
                max_pages = page_count_settings.get('max_pages', 15)
                # 更强调页数要求
                page_count_instruction = f"- 页数要求：必须严格生成{min_pages}-{max_pages}页的PPT。请确保生成的幻灯片数量在此范围内，不能超出或不足。"
                expected_page_count = {"min": min_pages, "max": max_pages, "mode": "range"}
                logger.info(f"Custom page count range set: {min_pages}-{max_pages} pages")
            else:
                # AI决定模式：不给出具体页数限制，让AI自行判断
                page_count_instruction = "- 页数要求：请根据主题内容的复杂度、深度和逻辑结构，自主决定最合适的页数，确保内容充实且逻辑清晰"
                expected_page_count = {"mode": "ai_decide"}
                logger.info("AI decide mode set for page count")

            # 使用字符串拼接避免f-string中的花括号冲突
            topic = confirmed_requirements['topic']
            target_audience = confirmed_requirements.get('target_audience', '普通大众')
            ppt_style = confirmed_requirements.get('ppt_style', 'general')
            custom_style = confirmed_requirements.get('custom_style_prompt', '无')
            description = confirmed_requirements.get('description', '无')

            context = """
项目信息：
- 主题：""" + topic + """
- 目标受众：""" + target_audience + """
""" + page_count_instruction + """
- PPT风格：""" + ppt_style + """
- 自定义风格说明：""" + custom_style + """
- 其他说明：""" + description + """

任务：生成完整的PPT大纲

请生成一个详细的PPT大纲，包括：
1. PPT标题
2. 各页面标题和主要内容要点
3. 逻辑结构和流程
4. 每页的内容重点
5. 根据页数要求合理安排内容分布

请以JSON格式返回大纲，使用```json```代码块包裹，格式如下：

```json
{
    "title": "PPT标题",
    "total_pages": 实际页数,
    "page_count_mode": \"""" + page_count_mode + """\",
    "slides": [
        {
            "page_number": 1,
            "title": "页面标题",
            "content_points": ["要点1", "要点2", "要点3"],
            "slide_type": "title|content|conclusion",
            "description": "页面内容描述"
        }
    ]
}
```
"""

            response = await self.ai_provider.text_completion(
                prompt=context,
                system_prompt=system_prompt,
                max_tokens=ai_config.max_tokens,
                temperature=ai_config.temperature
            )

            # Try to parse and store the outline
            import json
            import re

            try:
                # Extract JSON from the response content
                content = response.content.strip()

                # 改进的JSON提取方法
                json_str = None

                # 方法1: 尝试提取```json```代码块中的内容
                json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_block_match:
                    json_str = json_block_match.group(1)
                    logger.info("从```json```代码块中提取JSON")
                else:
                    # 方法2: 尝试提取```代码块中的内容（不带json标识）
                    code_block_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if code_block_match:
                        json_str = code_block_match.group(1)
                        logger.info("从```代码块中提取JSON")
                    else:
                        # 方法3: 尝试提取完整的JSON对象（改进的正则表达式）
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            logger.info("使用正则表达式提取JSON")
                        else:
                            # 方法4: 假设整个内容就是JSON
                            json_str = content
                            logger.info("将整个响应内容作为JSON处理")

                # 清理JSON字符串中的常见问题
                if json_str:
                    # 移除可能的前后空白和换行
                    json_str = json_str.strip()
                    # 修复常见的JSON格式问题
                    json_str = re.sub(r',\s*}', '}', json_str)  # 移除}前的多余逗号
                    json_str = re.sub(r',\s*]', ']', json_str)  # 移除]前的多余逗号

                outline_data = json.loads(json_str)

                # 验证和修复JSON数据
                outline_data = await self._validate_and_repair_outline_json(outline_data, confirmed_requirements)

                # 验证页数是否符合要求
                if expected_page_count and "slides" in outline_data:
                    actual_page_count = len(outline_data["slides"])
                    logger.info(f"Generated outline has {actual_page_count} pages")

                    if expected_page_count["mode"] == "range":
                        min_pages = expected_page_count["min"]
                        max_pages = expected_page_count["max"]

                        if actual_page_count < min_pages or actual_page_count > max_pages:
                            logger.warning(f"Generated outline has {actual_page_count} pages, but expected {min_pages}-{max_pages} pages. Adjusting...")
                            # 强制调整页数
                            outline_data = await self._adjust_outline_page_count(outline_data, min_pages, max_pages, confirmed_requirements)

                            # 验证调整后的页数
                            adjusted_page_count = len(outline_data.get("slides", []))
                            logger.info(f"Adjusted outline to {adjusted_page_count} pages")

                            if adjusted_page_count < min_pages or adjusted_page_count > max_pages:
                                logger.error(f"Failed to adjust page count to required range {min_pages}-{max_pages}")
                                # 如果调整失败，强制设置为中间值
                                target_pages = (min_pages + max_pages) // 2
                                outline_data = await self._force_page_count(outline_data, target_pages, confirmed_requirements)
                        else:
                            logger.info(f"Page count {actual_page_count} is within required range {min_pages}-{max_pages}")

                    # 添加页数信息到大纲元数据
                    if "metadata" not in outline_data:
                        outline_data["metadata"] = {}
                    outline_data["metadata"]["page_count_settings"] = expected_page_count
                    outline_data["metadata"]["actual_page_count"] = len(outline_data.get("slides", []))

                # Store outline in project (内存中)
                project = await self.project_manager.get_project(project_id)
                if project:
                    project.outline = outline_data
                    project.updated_at = time.time()
                    logger.info(f"Successfully saved outline to memory for project {project_id}")

                # Save outline to database (数据库中) - 这是关键步骤
                try:
                    from .db_project_manager import DatabaseProjectManager
                    db_manager = DatabaseProjectManager()
                    save_success = await db_manager.save_project_outline(project_id, outline_data)

                    if save_success:
                        logger.info(f"✅ Successfully saved outline to database for project {project_id}")

                        # 验证保存是否成功
                        saved_project = await db_manager.get_project(project_id)
                        if saved_project and saved_project.outline:
                            saved_slides_count = len(saved_project.outline.get('slides', []))
                            logger.info(f"✅ Verified: outline saved with {saved_slides_count} slides")
                        else:
                            logger.error(f"❌ Verification failed: outline not found in database")
                            return f"❌ 大纲保存失败：数据库验证失败"
                    else:
                        logger.error(f"❌ Failed to save outline to database for project {project_id}")
                        return f"❌ 大纲保存失败：数据库写入失败"

                except Exception as save_error:
                    logger.error(f"❌ Exception while saving outline to database: {save_error}")
                    import traceback
                    traceback.print_exc()
                    return f"❌ 大纲保存失败：{str(save_error)}"

                # 更新大纲生成阶段状态为完成
                try:
                    from .db_project_manager import DatabaseProjectManager
                    db_manager = DatabaseProjectManager()

                    await db_manager.update_stage_status(
                        project_id,
                        "outline_generation",
                        "completed",
                        100.0,
                        {
                            "outline_title": outline_data.get('title', '未知'),
                            "slides_count": len(outline_data.get('slides', [])),
                            "completed_at": time.time()
                        }
                    )
                    logger.info(f"Successfully updated outline generation stage to completed for project {project_id}")

                except Exception as stage_error:
                    logger.error(f"Failed to update outline generation stage status: {stage_error}")

                final_page_count = len(outline_data.get('slides', []))
                return f"✅ PPT大纲生成完成！\n\n标题：{outline_data.get('title', '未知')}\n页数：{final_page_count}页\n已保存到数据库\n\n{response.content}"

            except Exception as e:
                logger.error(f"Error parsing outline JSON: {e}")
                logger.error(f"Response content: {response.content[:500]}...")

                # Try to create a basic outline structure from the response
                try:
                    # Create a fallback outline structure
                    fallback_outline = {
                        "title": confirmed_requirements.get('topic', 'AI生成的PPT大纲'),
                        "slides": [
                            {
                                "page_number": 1,
                                "title": confirmed_requirements.get('topic', '标题页'),
                                "content_points": ["项目介绍", "主要内容", "核心价值"],
                                "slide_type": "title"
                            },
                            {
                                "page_number": 2,
                                "title": "主要内容",
                                "content_points": ["内容要点1", "内容要点2", "内容要点3"],
                                "slide_type": "content"
                            },
                            {
                                "page_number": 3,
                                "title": "谢谢观看",
                                "content_points": ["感谢聆听", "欢迎提问"],
                                "slide_type": "thankyou"
                            }
                        ]
                    }

                    # 验证和修复fallback大纲
                    fallback_outline = await self._validate_and_repair_outline_json(fallback_outline, confirmed_requirements)

                    # Store fallback outline in project
                    project = await self.project_manager.get_project(project_id)
                    if project:
                        project.outline = fallback_outline
                        project.updated_at = time.time()
                        logger.info(f"Saved fallback outline for project {project_id}")

                    # Save to database
                    try:
                        from .db_project_manager import DatabaseProjectManager
                        db_manager = DatabaseProjectManager()
                        save_success = await db_manager.save_project_outline(project_id, fallback_outline)

                        if save_success:
                            logger.info(f"Successfully saved fallback outline to database for project {project_id}")
                        else:
                            logger.error(f"Failed to save fallback outline to database for project {project_id}")
                    except Exception as save_error:
                        logger.error(f"Exception while saving fallback outline to database: {save_error}")

                    final_page_count = len(fallback_outline.get('slides', []))
                    return f"✅ PPT大纲生成完成！（使用备用方案）\n\n标题：{fallback_outline.get('title', '未知')}\n页数：{final_page_count}页\n已保存到数据库"

                except Exception as fallback_error:
                    logger.error(f"Error creating fallback outline: {fallback_error}")
                    return f"❌ 大纲生成失败：{str(e)}\n\n{response.content}"

        except Exception as e:
            logger.error(f"Error in outline generation: {e}")
            raise

    async def _adjust_outline_page_count(self, outline_data: Dict[str, Any], min_pages: int, max_pages: int, confirmed_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust outline page count to meet requirements"""
        try:
            current_slides = outline_data.get("slides", [])
            current_count = len(current_slides)

            if current_count < min_pages:
                # Need to add more slides
                logger.info(f"Adding slides to meet minimum requirement: {current_count} -> {min_pages}")
                outline_data = await self._expand_outline(outline_data, min_pages, confirmed_requirements)
            elif current_count > max_pages:
                # Need to reduce slides
                logger.info(f"Reducing slides to meet maximum requirement: {current_count} -> {max_pages}")
                outline_data = await self._condense_outline(outline_data, max_pages)

            return outline_data

        except Exception as e:
            logger.error(f"Error adjusting outline page count: {e}")
            return outline_data  # Return original if adjustment fails

    async def _expand_outline(self, outline_data: Dict[str, Any], target_pages: int, confirmed_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Expand outline to reach target page count"""
        try:
            slides = outline_data.get("slides", [])
            current_count = len(slides)
            needed_slides = target_pages - current_count

            # Generate additional slides based on content
            topic = confirmed_requirements.get('topic', outline_data.get('title', ''))
            focus_content = confirmed_requirements.get('focus_content', [])

            # Add content slides before the conclusion
            conclusion_slide = None
            if slides and slides[-1].get('slide_type') in ['thankyou', 'conclusion']:
                conclusion_slide = slides.pop()

            for i in range(needed_slides):
                page_number = len(slides) + 1
                if i < len(focus_content):
                    # Use focus content for new slides
                    new_slide = {
                        "page_number": page_number,
                        "title": focus_content[i],
                        "content_points": [f"{focus_content[i]}的详细介绍", "核心要点", "实际应用"],
                        "slide_type": "content",
                        "description": f"详细介绍{focus_content[i]}相关内容"
                    }
                else:
                    # Generate generic content slides
                    new_slide = {
                        "page_number": page_number,
                        "title": f"{topic} - 补充内容 {i+1}",
                        "content_points": ["补充要点1", "补充要点2", "补充要点3"],
                        "slide_type": "content",
                        "description": f"关于{topic}的补充内容"
                    }
                slides.append(new_slide)

            # Re-add conclusion slide if it existed
            if conclusion_slide:
                conclusion_slide["page_number"] = len(slides) + 1
                slides.append(conclusion_slide)

            # Update page numbers
            for i, slide in enumerate(slides):
                slide["page_number"] = i + 1

            outline_data["slides"] = slides
            return outline_data

        except Exception as e:
            logger.error(f"Error expanding outline: {e}")
            return outline_data

    async def _condense_outline(self, outline_data: Dict[str, Any], target_pages: int) -> Dict[str, Any]:
        """Condense outline to reach target page count"""
        try:
            slides = outline_data.get("slides", [])
            current_count = len(slides)

            if current_count <= target_pages:
                return outline_data

            # Keep title and conclusion slides, condense content slides
            title_slides = [s for s in slides if s.get('slide_type') in ['title', 'cover']]
            conclusion_slides = [s for s in slides if s.get('slide_type') in ['thankyou', 'conclusion']]
            content_slides = [s for s in slides if s.get('slide_type') not in ['title', 'cover', 'thankyou', 'conclusion']]

            # Calculate how many content slides we can keep
            reserved_slots = len(title_slides) + len(conclusion_slides)
            available_content_slots = target_pages - reserved_slots

            if available_content_slots > 0 and len(content_slides) > available_content_slots:
                # Keep the most important content slides
                content_slides = content_slides[:available_content_slots]

            # Rebuild slides list
            new_slides = title_slides + content_slides + conclusion_slides

            # Update page numbers
            for i, slide in enumerate(new_slides):
                slide["page_number"] = i + 1

            outline_data["slides"] = new_slides
            return outline_data

        except Exception as e:
            logger.error(f"Error condensing outline: {e}")
            return outline_data

    async def _force_page_count(self, outline_data: Dict[str, Any], target_pages: int, confirmed_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Force outline to exact page count"""
        try:
            slides = outline_data.get("slides", [])
            current_count = len(slides)

            logger.info(f"Forcing page count from {current_count} to {target_pages}")

            if current_count == target_pages:
                return outline_data

            # Keep title and conclusion slides
            title_slides = [s for s in slides if s.get('slide_type') in ['title', 'cover']]
            conclusion_slides = [s for s in slides if s.get('slide_type') in ['thankyou', 'conclusion']]
            content_slides = [s for s in slides if s.get('slide_type') not in ['title', 'cover', 'thankyou', 'conclusion']]

            # Calculate content slots needed
            reserved_slots = len(title_slides) + len(conclusion_slides)
            content_slots_needed = target_pages - reserved_slots

            if content_slots_needed <= 0:
                # Only keep title slide if no room for content
                new_slides = title_slides[:1] if title_slides else []
            else:
                if len(content_slides) > content_slots_needed:
                    # Reduce content slides
                    content_slides = content_slides[:content_slots_needed]
                elif len(content_slides) < content_slots_needed:
                    # Add more content slides
                    topic = confirmed_requirements.get('topic', outline_data.get('title', ''))
                    focus_content = confirmed_requirements.get('focus_content', [])

                    for i in range(content_slots_needed - len(content_slides)):
                        page_number = len(content_slides) + i + 1
                        if i < len(focus_content):
                            new_slide = {
                                "page_number": page_number,
                                "title": focus_content[i],
                                "content_points": [f"{focus_content[i]}的详细介绍", "核心要点", "实际应用"],
                                "slide_type": "content",
                                "description": f"详细介绍{focus_content[i]}相关内容"
                            }
                        else:
                            new_slide = {
                                "page_number": page_number,
                                "title": f"{topic} - 内容 {i+1}",
                                "content_points": ["要点1", "要点2", "要点3"],
                                "slide_type": "content",
                                "description": f"关于{topic}的内容"
                            }
                        content_slides.append(new_slide)

                # Rebuild slides list
                new_slides = title_slides + content_slides + conclusion_slides

            # Update page numbers
            for i, slide in enumerate(new_slides):
                slide["page_number"] = i + 1

            outline_data["slides"] = new_slides
            logger.info(f"Successfully forced page count to {len(new_slides)} pages")
            return outline_data

        except Exception as e:
            logger.error(f"Error forcing page count: {e}")
            return outline_data

    async def _execute_ppt_creation(self, project_id: str, confirmed_requirements: Dict[str, Any], system_prompt: str) -> str:
        """Execute PPT creation by generating HTML pages individually with streaming"""
        try:
            project = await self.project_manager.get_project(project_id)
            if not project or not project.outline:
                return "❌ 错误：未找到PPT大纲，请先完成大纲生成步骤"

            outline = project.outline
            slides = outline.get('slides', [])

            if not slides:
                return "❌ 错误：大纲中没有幻灯片信息"

            # 验证大纲页数与需求一致性
            if project.confirmed_requirements:
                page_count_settings = project.confirmed_requirements.get('page_count_settings', {})
                if page_count_settings.get('mode') == 'custom_range':
                    min_pages = page_count_settings.get('min_pages', 8)
                    max_pages = page_count_settings.get('max_pages', 15)
                    actual_pages = len(slides)

                    if actual_pages < min_pages or actual_pages > max_pages:
                        logger.warning(f"Outline has {actual_pages} pages, but requirements specify {min_pages}-{max_pages} pages")
                        return f"⚠️ 错误：大纲有{actual_pages}页，但需求要求{min_pages}-{max_pages}页。请重新生成大纲以符合页数要求。"

            # Initialize slides data - 确保与大纲页数完全一致
            project.slides_data = []
            project.updated_at = time.time()

            # 确保confirmed_requirements包含项目ID，用于模板选择
            if confirmed_requirements:
                confirmed_requirements['project_id'] = project_id

            # 验证slides数据结构
            if not slides or len(slides) == 0:
                return "❌ 错误：大纲中没有有效的幻灯片数据"

            logger.info(f"Starting PPT generation for {len(slides)} slides based on outline")

            # 确保每个slide都有必要的字段
            for i, slide in enumerate(slides):
                if not slide.get('title'):
                    slide['title'] = f"幻灯片 {i+1}"
                if not slide.get('page_number'):
                    slide['page_number'] = i + 1

            return f"🚀 开始PPT制作...\n\n将严格按照大纲为 {len(slides)} 页幻灯片逐页生成HTML内容\n大纲页数：{len(slides)}页\n请在编辑器中查看实时生成过程"

        except Exception as e:
            logger.error(f"Error in PPT creation: {e}")
            raise

    async def generate_slides_streaming(self, project_id: str):
        """Generate slides with streaming output for real-time display"""
        try:
            import json
            import time

            project = await self.project_manager.get_project(project_id)
            if not project:
                error_data = {'error': '项目未找到'}
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            # 检查并确保大纲数据正确
            outline = None
            slides = []

            # 首先尝试从项目中获取大纲
            if project.outline and isinstance(project.outline, dict):
                outline = project.outline
                slides = outline.get('slides', [])
                logger.info(f"Found outline in project with {len(slides)} slides")

            # 如果没有slides或slides为空，尝试从数据库重新加载
            if not slides:
                logger.info(f"No slides found in project outline, attempting to reload from database")
                logger.error(f"DEBUG: Full outline structure for project {project_id}:")
                logger.error(f"Outline type: {type(project.outline)}")
                if project.outline:
                    logger.error(f"Outline keys: {list(project.outline.keys()) if isinstance(project.outline, dict) else 'Not a dict'}")
                    if isinstance(project.outline, dict) and 'slides' in project.outline:
                        logger.error(f"Slides type: {type(project.outline['slides'])}, content: {project.outline['slides']}")

                try:
                    from .db_project_manager import DatabaseProjectManager
                    db_manager = DatabaseProjectManager()

                    # 重新从数据库获取项目数据
                    fresh_project = await db_manager.get_project(project_id)
                    if fresh_project and fresh_project.outline:
                        outline = fresh_project.outline
                        slides = outline.get('slides', [])
                        logger.info(f"Reloaded outline from database with {len(slides)} slides")

                        # 更新内存中的项目数据
                        project.outline = outline
                    else:
                        logger.error(f"Failed to reload project from database or outline is None")
                        if fresh_project:
                            logger.error(f"Fresh project outline type: {type(fresh_project.outline)}")

                except Exception as db_error:
                    logger.error(f"Failed to reload outline from database: {db_error}")
                    import traceback
                    logger.error(f"Database reload traceback: {traceback.format_exc()}")

            # 如果仍然没有slides，检查是否有大纲内容需要解析
            if not slides and outline and 'content' in outline:
                logger.info(f"Found outline content, attempting to parse slides")
                try:
                    # 尝试解析大纲内容
                    parsed_outline = self._parse_outline_content(outline['content'], project)
                    slides = parsed_outline.get('slides', [])
                    logger.info(f"Parsed {len(slides)} slides from outline content")

                    # 更新大纲数据
                    outline['slides'] = slides
                    project.outline = outline

                except Exception as parse_error:
                    logger.error(f"Failed to parse outline content: {parse_error}")

            # 特殊处理：如果outline直接包含slides数组但为空，尝试从content字段解析
            if not slides and outline and isinstance(outline, dict):
                # 检查是否有content字段包含JSON格式的大纲
                content_field = outline.get('content', '')
                if content_field and isinstance(content_field, str):
                    logger.info(f"Attempting to parse slides from content field")
                    try:
                        import json
                        # 尝试解析content字段中的JSON
                        content_data = json.loads(content_field)
                        if isinstance(content_data, dict) and 'slides' in content_data:
                            slides = content_data['slides']
                            logger.info(f"Successfully parsed {len(slides)} slides from content JSON")

                            # 更新outline中的slides
                            outline['slides'] = slides
                            project.outline = outline
                    except json.JSONDecodeError as json_error:
                        logger.error(f"Failed to parse content as JSON: {json_error}")
                    except Exception as content_error:
                        logger.error(f"Failed to extract slides from content: {content_error}")

            # 最后尝试：如果outline本身就是完整的大纲数据（包含title和slides）
            if not slides and outline and isinstance(outline, dict):
                # 检查outline是否直接包含slides数组
                direct_slides = outline.get('slides', [])
                if direct_slides and isinstance(direct_slides, list):
                    slides = direct_slides
                    logger.info(f"Found {len(slides)} slides directly in outline")
                # 或者检查是否有嵌套的大纲结构
                elif 'outline' in outline and isinstance(outline['outline'], dict):
                    nested_slides = outline['outline'].get('slides', [])
                    if nested_slides and isinstance(nested_slides, list):
                        slides = nested_slides
                        logger.info(f"Found {len(slides)} slides in nested outline structure")

            # 额外调试：打印outline结构以便诊断
            if not slides:
                logger.error(f"DEBUG: Full outline structure for project {project_id}:")
                logger.error(f"Outline type: {type(outline)}")
                if outline:
                    logger.error(f"Outline keys: {list(outline.keys()) if isinstance(outline, dict) else 'Not a dict'}")
                    if isinstance(outline, dict):
                        for key, value in outline.items():
                            logger.error(f"  {key}: {type(value)} - {len(value) if isinstance(value, (list, dict, str)) else value}")
                            if key == 'slides' and isinstance(value, list):
                                logger.error(f"    Slides count: {len(value)}")
                                if value:
                                    logger.error(f"    First slide: {value[0] if len(value) > 0 else 'None'}")
                            elif key == 'content' and isinstance(value, str):
                                logger.error(f"    Content preview: {value[:200]}...")

                # 尝试直接从outline中提取slides，不管结构如何
                if isinstance(outline, dict):
                    # 递归搜索slides字段
                    def find_slides_recursive(obj, path=""):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                current_path = f"{path}.{k}" if path else k
                                if k == 'slides' and isinstance(v, list) and v:
                                    logger.info(f"Found slides at path: {current_path} with {len(v)} items")
                                    return v
                                elif isinstance(v, (dict, list)):
                                    result = find_slides_recursive(v, current_path)
                                    if result:
                                        return result
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                current_path = f"{path}[{i}]" if path else f"[{i}]"
                                if isinstance(item, (dict, list)):
                                    result = find_slides_recursive(item, current_path)
                                    if result:
                                        return result
                        return None

                    found_slides = find_slides_recursive(outline)
                    if found_slides:
                        slides = found_slides
                        logger.info(f"Successfully found {len(slides)} slides through recursive search")

            # 最后的fallback：如果仍然没有slides，返回错误而不是生成默认大纲
            if not slides:
                error_message = "❌ 错误：未找到PPT大纲数据，请先完成大纲生成步骤"
                logger.error(f"No slides found for project {project_id}")
                logger.error(f"Project outline structure: {type(project.outline)}")
                if project.outline:
                    logger.error(f"Outline keys: {list(project.outline.keys()) if isinstance(project.outline, dict) else 'Not a dict'}")
                    if isinstance(project.outline, dict) and 'slides' in project.outline:
                        logger.error(f"Slides type: {type(project.outline['slides'])}, length: {len(project.outline['slides']) if isinstance(project.outline['slides'], list) else 'Not a list'}")
                error_data = {'error': error_message}
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            # 如果没有确认需求，使用默认需求配置
            if not project.confirmed_requirements:
                logger.info(f"Project {project_id} has no confirmed requirements, using default configuration")
                confirmed_requirements = {
                    "topic": project.topic,
                    "target_audience": "普通大众",
                    "focus_content": ["核心概念", "主要特点"],
                    "tech_highlights": ["技术要点", "实践应用"],
                    "page_count_settings": {"mode": "ai_decide"},
                    "ppt_style": "general",
                    "description": f"基于主题 '{project.topic}' 的PPT演示"
                }
            else:
                confirmed_requirements = project.confirmed_requirements

            # 确保我们有有效的大纲和slides数据
            if not outline:
                outline = project.outline

            if not slides:
                slides = outline.get('slides', []) if outline else []

            # 最终检查：如果仍然没有slides，返回错误
            if not slides:
                error_message = "❌ 错误：大纲中没有幻灯片信息，请检查大纲生成是否完成"
                logger.error(f"No slides found after all attempts for project {project_id}")
                error_data = {'error': error_message}
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            logger.info(f"Starting PPT generation for project {project_id} with {len(slides)} slides")

            # Load system prompt
            system_prompt = self._load_prompts_md_system_prompt()

            # Initialize slides data if not exists
            if not project.slides_data:
                project.slides_data = []

            # Generate each slide individually
            for i, slide in enumerate(slides):
                try:
                    # Check if slide already exists
                    existing_slide = None
                    if project.slides_data and i < len(project.slides_data):
                        existing_slide = project.slides_data[i]

                    # If slide exists and has content (either user-edited or AI-generated), skip generation
                    if existing_slide and existing_slide.get('html_content'):
                        if existing_slide.get('is_user_edited', False):
                            logger.info(f"Skipping slide {i+1} generation - user has edited this slide")
                            skip_message = f'第{i+1}页已被用户编辑，跳过重新生成'
                        else:
                            logger.info(f"Skipping slide {i+1} generation - slide already exists")
                            skip_message = f'第{i+1}页已存在，跳过生成'

                        # Send skip message
                        skip_data = {
                            'type': 'slide_skipped',
                            'current': i + 1,
                            'total': len(slides),
                            'message': skip_message,
                            'slide_data': existing_slide
                        }
                        yield f"data: {json.dumps(skip_data)}\n\n"
                        continue

                    # Send progress update
                    slide_title = slide.get('title', '')
                    progress_data = {
                        'type': 'progress',
                        'current': i + 1,
                        'total': len(slides),
                        'message': f'正在生成第{i+1}页：{slide_title}...'
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    logger.info(f"Generating slide {i+1}/{len(slides)}: {slide_title}")

                    # Generate HTML for this slide with context
                    html_content = await self._generate_single_slide_html_with_prompts(
                        slide, confirmed_requirements, system_prompt, i + 1, len(slides), slides, project.slides_data, project_id
                    )
                    logger.debug(f"Successfully generated slide {i+1}/{len(slides)}: {html_content}")

                    # Create slide data
                    slide_data = {
                        "page_number": i + 1,
                        "title": slide.get('title', f'第{i+1}页'),
                        "html_content": html_content,
                        "is_user_edited": False  # Mark as AI-generated
                    }

                    # Update project slides data
                    while len(project.slides_data) <= i:
                        project.slides_data.append(None)
                    project.slides_data[i] = slide_data

                    # 立即保存当前页面到数据库，确保实时同步和独立的创建时间
                    try:
                        from .db_project_manager import DatabaseProjectManager
                        db_manager = DatabaseProjectManager()

                        # 更新项目的slides_data和updated_at
                        project.updated_at = time.time()

                        # 保存单个slide到数据库，保持独立的创建时间
                        await db_manager.save_single_slide(project_id, i, slide_data)
                        logger.info(f"Successfully saved slide {i+1} to database for project {project_id}")
                    except Exception as save_error:
                        logger.error(f"Failed to save slide {i+1} to database: {save_error}")
                        # 继续生成，不因保存失败而中断

                    # Send slide data
                    slide_response = {'type': 'slide', 'slide_data': slide_data}
                    yield f"data: {json.dumps(slide_response)}\n\n"

                except Exception as e:
                    logger.error(f"Error generating slide {i+1}: {e}")
                    # Send error for this slide
                    error_slide = {
                        "page_number": i + 1,
                        "title": slide.get('title', f'第{i+1}页'),
                        "html_content": f"<div style='padding: 50px; text-align: center; color: red;'>生成失败：{str(e)}</div>"
                    }

                    while len(project.slides_data) <= i:
                        project.slides_data.append(None)
                    project.slides_data[i] = error_slide

                    error_response = {'type': 'slide', 'slide_data': error_slide}
                    yield f"data: {json.dumps(error_response)}\n\n"

            # Generate combined HTML
            project.slides_html = self._combine_slides_to_full_html(
                project.slides_data, outline.get('title', project.title)
            )
            project.status = "completed"
            project.updated_at = time.time()

            # Update project status and stage completion (slides already saved individually)
            try:
                from .db_project_manager import DatabaseProjectManager
                db_manager = DatabaseProjectManager()

                # Update project with final slides_html and slides_data (without recreating individual slides)
                await db_manager.update_project_data(project_id, {
                    "slides_html": project.slides_html,
                    "slides_data": project.slides_data,
                    "status": "completed",
                    "updated_at": time.time()
                })
                logger.info(f"Successfully updated project data for project {project_id}")

                # Update PPT creation stage status to completed
                await db_manager.update_stage_status(
                    project_id,
                    "ppt_creation",
                    "completed",
                    100.0,
                    {"slides_count": len(slides), "completed_at": time.time()}
                )
                logger.info(f"Successfully updated PPT creation stage to completed for project {project_id}")

            except Exception as save_error:
                logger.error(f"Failed to update project status in database: {save_error}")
                # Continue anyway, as the data is still in memory

            # Send completion message
            complete_message = f'✅ PPT制作完成！成功生成 {len(slides)} 页幻灯片'
            complete_response = {'type': 'complete', 'message': complete_message}
            yield f"data: {json.dumps(complete_response)}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming PPT generation: {e}")
            error_message = f'生成过程中出现错误：{str(e)}'
            error_response = {'type': 'error', 'message': error_message}
            yield f"data: {json.dumps(error_response)}\n\n"

    async def _execute_general_subtask(self, project_id: str, stage, subtask: str, confirmed_requirements: Dict[str, Any], system_prompt: str) -> str:
        """Execute general subtask"""
        context = f"""
项目信息：
- 主题：{confirmed_requirements['topic']}
- 类型：{confirmed_requirements['type']}
- 重点展示内容：{confirmed_requirements['focus_content']}
- 技术亮点：{confirmed_requirements['tech_highlights']}
- 目标受众：{confirmed_requirements['target_audience']}
- 其他说明：{confirmed_requirements.get('description', '无')}

当前阶段：{stage.name}
当前子任务：{subtask}

请根据以上信息执行当前子任务。
"""

        response = await self.ai_provider.text_completion(
            prompt=context,
            system_prompt=system_prompt,
            max_tokens=ai_config.max_tokens,
            temperature=ai_config.temperature
        )

        return response.content

    async def _generate_single_slide_html_with_prompts(self, slide_data: Dict[str, Any], confirmed_requirements: Dict[str, Any],
                                                     system_prompt: str, page_number: int, total_pages: int,
                                                     all_slides: List[Dict[str, Any]] = None, existing_slides_data: List[Dict[str, Any]] = None, project_id: str = None) -> str:
        """Generate HTML for a single slide using prompts.md and first step information with template selection"""
        try:
            # 使用传入的项目ID或从confirmed_requirements获取
            if not project_id:
                project_id = confirmed_requirements.get('project_id')

            selected_template = None

            # 如果有项目ID，尝试获取选择的全局母版模板
            if project_id:
                try:
                    selected_template = await self.get_selected_global_template(project_id)
                    if selected_template:
                        logger.info(f"为第{page_number}页使用全局母版: {selected_template['template_name']}")
                except Exception as e:
                    logger.warning(f"获取全局母版失败，使用默认生成方式: {e}")

            # 如果有选中的全局母版，使用模板生成
            if selected_template:
                return await self._generate_slide_with_template(
                    slide_data, selected_template, page_number, total_pages, confirmed_requirements
                )

            # 否则使用原有的生成方式，但应用新的设计基因缓存和统一创意指导
            # 获取或提取设计基因（只在第一页提取一次）
            style_genes = await self._get_or_extract_style_genes(project_id, "", page_number)

            # 生成统一的创意设计指导
            unified_design_guide = await self._generate_unified_design_guide(slide_data, page_number, total_pages)

            # Build context information for better coherence
            context_info = self._build_slide_context(page_number, total_pages)

            # Create detailed context based on first step information and HTML style prompts
            context = f"""
根据项目信息，为第{page_number}页生成完整的HTML代码。

项目信息：
- 主题：{confirmed_requirements.get('topic', '')}
- 目标受众：{confirmed_requirements.get('target_audience', '')}
- 其他说明：{confirmed_requirements.get('description', '无')}

当前页面信息：
{slide_data}

**严格内容约束（重要）**：
1. **严格基于提供内容**：只能使用上述提供的标题和内容要点，不得添加、修改或编造任何内容
2. **内容完整性**：必须包含所有提供的内容要点，不得遗漏任何一个
3. **展示方式**：
   - 根据内容特点选择最合适的展示方式
   - 可以创新性地组合多种视觉元素
   - 支持Chart.js、ECharts、D3.js图表、进度条、时间线、统计卡片、对比表格等任何您认为合适的组件
   - **布局多样性**：通过颜色、字体、间距、图标等元素创造视觉变化，而非依赖图表
4. **禁止编造**：严禁添加任何未在内容要点中明确提到的信息、数据或描述
5. **布局优化**：
   - 当内容要点少于3个时，使用居中布局，增大字体和间距
   - 当内容要点为1个时，确保该内容在页面正中央显示
   - 使用flexbox布局实现完美居中（justify-content: center, align-items: center）
   - 适当增加padding确保内容不会过于紧凑

{context_info}

**设计平衡要求（一致性与创新并重）**：
1. 使用16:9的响应式PPT尺寸，适配不同屏幕大小
2. 页面右下角显示页码：{page_number}/{total_pages}
3. **必须保持一致的核心元素**：
   - 遵循提供的设计风格模板中的核心约束
   - 保持主色调和字体系统的统一
   - 维持整体视觉品牌的连贯性
4. **鼓励创新的设计空间**：
   - 根据内容特点创新布局结构
   - 灵活运用视觉元素增强表达效果
   - 适度融入当前设计趋势
   - 优化信息层次和用户体验

**内容连贯性要求**：
1. 内容应与前后页面逻辑连贯
2. 避免重复前面页面的内容
3. 为后续页面做好铺垫
4. 保持整体叙述的流畅性

**技术规范**：
- 生成完整的HTML页面（包含<!DOCTYPE html>、head、body，不包含style标签）
- 使用Tailwind CSS或内联CSS，确保美观的设计
- 使用16:9响应式设计，适配不同屏幕尺寸
- 使用CSS的aspect-ratio属性保持16:9比例
- 使用clamp()函数实现响应式字体大小
- 使用百分比和vw/vh单位实现响应式布局
- 内容布局清晰，重点突出
- 确保文字清晰可读，颜色搭配协调
- 根据页面类型调整布局：
  - title页面：居中大标题设计
  - content页面：标题+内容区域布局
  - conclusion页面：总结性布局

**富文本支持**：
- 支持数学公式（使用MathJax）、代码高亮（使用Prism.js）、图表（使用Chart.js）等富文本元素
- 根据内容需要自动添加相应的库和样式

**严格的页面尺寸和高度控制**：
- **页面尺寸**：html {{ height: 100%; display: flex; align-items: center; justify-content: center; }} body {{ width: 1280px; height: 720px; position: relative; overflow: hidden; }}
- **内容高度分配**：
  * 标题和页码与参考模板保持一致(包括字体、大小、位置和颜色)
  * 主内容区域：580-620px（充分利用可用空间，根据内容动态调整）
- **空间充分利用原则**：
  * **垂直空间最大化**：确保内容区域占据页面的85-90%高度
  * **内容自适应扩展**：根据内容数量和类型，动态调整各区域高度
  * **避免大量留白**：合理分配空间，避免底部出现过多空余区域
- **内容点数量自适应布局**：
  * 1-2个内容点：大字体(1.8-2.2rem)，充分利用垂直空间，居中显示，增加行间距
  * 3-4个内容点：中等字体(1.4-1.8rem)，适中间距，合理分布
  * 5-6个内容点：标准字体(1.2-1.5rem)，紧凑间距，网格布局
  * 7个以上：小字体(1.0-1.3rem)，最紧凑布局或分栏显示，充分利用空间
- **图表高度优化**：图表容器高度根据可用空间动态调整（200-350px），确保图表清晰可见
- **溢出处理**：绝对禁止滚动条，内容过多时优先使用分栏、网格布局或适当压缩字体

**核心设计基因（必须保持）**：
{style_genes}

**统一创意设计指导**：
{unified_design_guide}

**重要输出格式要求：**
- 必须使用markdown代码块格式返回HTML代码
- 格式：```html\n[HTML代码]\n```
- HTML代码必须以<!DOCTYPE html>开始，以</html>结束
- 不要在代码块前后添加任何解释文字
- 确保代码块标记正确且完整
- 严格遵循上述风格要求生成HTML页面
"""

            # Try to generate HTML with retry mechanism for incomplete responses
            html_content = await self._generate_html_with_retry(
                context, system_prompt, slide_data, page_number, total_pages, max_retries=5
            )

            return html_content

        except Exception as e:
            logger.error(f"Error generating single slide HTML with prompts: {e}")
            # Return a fallback HTML
            return self._generate_fallback_slide_html(slide_data, page_number, total_pages)

    async def _generate_slide_with_template(self, slide_data: Dict[str, Any], template: Dict[str, Any],
                                          page_number: int, total_pages: int,
                                          confirmed_requirements: Dict[str, Any]) -> str:
        """使用选定的模板生成幻灯片HTML - AI参考模板风格生成新HTML"""
        try:
            # 获取模板HTML作为风格参考
            template_html = template['html_template']
            template_name = template.get('template_name', '未知模板')

            logger.info(f"使用模板 {template_name} 作为风格参考生成第{page_number}页")

            # 构建创意模板参考上下文
            context = await self._build_creative_template_context(
                slide_data, template_html, template_name, page_number, total_pages, confirmed_requirements
            )

            # 使用AI生成风格一致但内容创新的HTML
            system_prompt = self._load_prompts_md_system_prompt()
            html_content = await self._generate_html_with_retry(
                context, system_prompt, slide_data, page_number, total_pages, max_retries=5
            )

            if html_content:
                logger.info(f"成功使用模板 {template_name} 风格生成第{page_number}页")
                return html_content
            else:
                logger.warning(f"模板风格生成失败，回退到默认生成方式")
                # 回退到原有生成方式
                return await self._generate_fallback_slide_html(slide_data, page_number, total_pages)

        except Exception as e:
            logger.error(f"使用模板风格生成幻灯片失败: {e}")
            # 回退到原有生成方式
            return await self._generate_fallback_slide_html(slide_data, page_number, total_pages)


    async def _build_creative_template_context(self, slide_data: Dict[str, Any], template_html: str,
                                       template_name: str, page_number: int, total_pages: int,
                                       confirmed_requirements: Dict[str, Any]) -> str:
        """构建创意模板参考上下文，平衡风格一致性与创意多样性（优化版本）"""

        # 获取项目ID，检查是否已缓存设计基因
        project_id = confirmed_requirements.get('project_id')
        style_genes = None

        # 设计基因只在第一页提取一次，后续都使用第一页的
        style_genes = await self._get_or_extract_style_genes(project_id, template_html, page_number)

        # 生成统一的创意设计指导（合并创意变化指导和内容驱动的设计建议）
        unified_design_guide = await self._generate_unified_design_guide(slide_data, page_number, total_pages)

        # 获取实际内容要点
        slide_title = slide_data.get('title', f'第{page_number}页')
        slide_type = slide_data.get('slide_type', 'content')

        # Build context information for better coherence
        context_info = self._build_slide_context(page_number, total_pages)

        # 获取项目信息
        project_topic = confirmed_requirements.get('topic', '')
        project_type = confirmed_requirements.get('type', '')
        project_audience = confirmed_requirements.get('target_audience', '')
        project_style = confirmed_requirements.get('ppt_style', 'general')
        context = f"""
你是一位富有创意的设计师，需要为第{page_number}页创建一个既保持风格一致性又充满创意的PPT页面。

**严格内容约束**：
- 页面标题：{slide_title}
- 页面类型：{slide_type}
- 总页数：{total_pages}

**完整页面数据参考**：
{slide_data}

**参考模板（标题和页码完全保持原样）**：
```html
{template_html}
```

{context_info}

**重要约束**：
1. **严格基于提供的内容**：只能使用上述提供的slide_data中的内容，不得添加、修改或编造任何内容
2. **内容完整性**：必须包含所有提供的内容要点，不得遗漏
3. **页面完整性**: 必须确保页面中的文本和图表都完全可见，不能有任何内容被裁剪或隐藏

**严格的页面尺寸和高度控制**：
- **页面尺寸**：html {{ height: 100%; display: flex; align-items: center; justify-content: center; }} body {{ width: 1280px; height: 720px; position: relative; overflow: hidden; }}
- **内容高度分配**：
  * 标题和页码与参考模板保持一致(包括字体、大小、位置和颜色)
  * 主内容区域：580-620px（充分利用可用空间，根据内容动态调整）
- **空间充分利用原则**：
  * **垂直空间最大化**：确保内容区域占据页面的85-90%高度
  * **内容自适应扩展**：根据内容数量和类型，动态调整各区域高度
  * **避免大量留白**：合理分配空间，避免底部出现过多空余区域
- **内容点数量自适应布局**：
  * 1-2个内容点：大字体(1.8-2.2rem)，充分利用垂直空间，居中显示，增加行间距
  * 3-4个内容点：中等字体(1.4-1.8rem)，适中间距，合理分布
  * 5-6个内容点：标准字体(1.2-1.5rem)，紧凑间距，网格布局
  * 7个以上：小字体(1.0-1.3rem)，最紧凑布局或分栏显示，充分利用空间
- **图表高度优化**：图表容器高度根据可用空间动态调整，确保图表清晰可见，不溢出页面和被遮挡
- **溢出处理**：绝对禁止滚动条，内容过多时优先使用分栏、网格布局或适当压缩字体

**展示方式**：
- 根据内容特点选择最合适的展示方式
- 可以创新性地组合多种视觉元素
- 支持Chart.js、ECharts、D3.js图表、进度条、时间线、统计卡片、对比表格等任何您认为合适的组件
- **布局多样性**：使用网格布局、卡片组合、图标配文字、图片配文字等多种方式增加视觉变化
- **空间充分利用策略**：
  * 使用flex布局的justify-content: space-between或space-around确保内容均匀分布
  * 对于少量内容，增加元素尺寸和间距，充分利用垂直空间
  * 对于大量内容，使用多列布局或网格系统，避免内容堆积在上半部分
  * 合理使用padding和margin，确保内容区域占满可用空间
  * 图表和视觉元素适当放大，提升视觉效果和空间利用率

**核心设计基因（必须保持）**：
{style_genes}

**统一创意设计指导**：
{unified_design_guide}

**项目背景**：
- 主题：{project_topic}
- 类型：{project_type}
- 目标受众：{project_audience}
- PPT风格：{project_style}

**设计哲学**：
1. **一致性原则** - 严格遵循核心设计基因，确保品牌识别度
2. **创新性原则** - 在一致性框架内大胆创新，避免千篇一律
3. **内容适配原则** - 让设计服务于内容，而非内容迁就设计
4. **用户体验原则** - 优化信息传达效率和视觉舒适度
5. **空间最大化原则** - 充分利用页面的每一寸空间，避免大量留白浪费
   * 内容应该垂直分布均匀，占满可用高度
   * 使用合适的字体大小和间距，确保内容填充整个内容区域
   * 底部不应该有超过50px的空余空间



**创意要求**：
- 在保持核心设计基因的前提下，创造独特的布局结构
- 根据内容特点选择最佳的信息展示方式
- 使用创新的视觉元素增强表达效果
- 确保每一页都有独特的视觉亮点
- **空间利用创意要求**：
  * 设计时必须考虑内容如何填满整个可用空间
  * 使用flex布局的flex-grow属性让内容区域自动扩展
  * 对于内容较少的页面，通过增大字体、图标、间距等方式充分利用空间
  * 避免所有内容都集中在页面上半部分，要有意识地分布到整个页面
  * 底部区域也要合理利用，可以放置装饰元素或次要信息

**富文本支持**：
- 支持数学公式（使用MathJax）、代码高亮（使用Prism.js）、图表（使用Chart.js）等富文本元素
- 根据内容需要自动添加相应的库和样式

**技术规范**：
- 生成完整的HTML页面（包含<!DOCTYPE html>、head、body）
- 使用Tailwind CSS或内联CSS，确保美观的设计
- 页面尺寸严格控制：html {{ height: 100%; display: flex; align-items: center; justify-content: center; }} body {{ width: 1280px; height: 720px; position: relative; overflow: hidden; }}
- 支持使用Chart.js和Font Awesome库
- 页码显示为：{page_number}/{total_pages}
- **空间利用优化**：
  * 主内容区域必须充分利用可用高度（580-620px）
  * 避免内容区域底部出现大量空白
  * 根据内容数量动态调整字体大小和间距
  * 使用flex布局确保内容垂直分布均匀
- 图表容器高度动态设置，确保所有图表内容完全可见且充分利用空间
- 所有内容元素都必须在1280x720范围内，不能出现滚动条

**重要输出格式要求**：
- 必须使用markdown代码块格式返回HTML代码
- 格式：```html\\n[HTML代码]\\n```
- HTML代码必须以<!DOCTYPE html>开始，以</html>结束
- 不要在代码块前后添加任何解释文字
"""

        return context

    async def _extract_style_genes(self, template_html: str) -> str:
        """使用AI从模板中提取核心设计基因"""
        try:
            # 限制模板代码长度，避免token过多
            template_code = template_html

            prompt = f"""
作为专业的UI/UX设计师，请分析以下HTML模板代码，提取其核心设计基因。

**模板代码：**
```html
{template_code}
```

请从以下维度分析并提取设计基因：

1. **色彩系统**：主色调、辅助色、背景色、文字色等
2. **字体系统**：字体族、字重、字号层次等
3. **布局方式**：Flexbox、Grid、定位方式等
4. **间距系统**：padding、margin的规律和比例
5. **视觉元素**：圆角、阴影、边框、渐变等
6. **组件风格**：卡片、按钮、图标的设计特征
7. **整体风格**：现代简约、商务专业、创意活泼等

要求：
- 提取最核心、最具识别性的设计特征
- 每个特征要具体、可操作
- 避免过于细节的描述，聚焦关键要素
- 以简洁的要点形式返回，每个要点以"- "开头

请返回核心设计基因分析：
"""

            # 调用AI分析
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=min(ai_config.max_tokens, 600),
                temperature=0.3  # 较低温度确保分析准确性
            )

            ai_genes = response.content.strip()

            # 如果AI分析失败，回退到基础提取
            if not ai_genes or len(ai_genes) < 50:
                return self._extract_fallback_style_genes(template_html)

            return ai_genes

        except Exception as e:
            logger.warning(f"AI提取设计基因失败: {e}")
            # 回退到基础提取
            return self._extract_fallback_style_genes(template_html)

    def _extract_fallback_style_genes(self, template_html: str) -> str:
        """回退的基础设计基因提取"""
        import re

        genes = []

        try:
            # 提取主要颜色方案
            colors = re.findall(r'(?:background|color)[^:]*:\s*([^;]+)', template_html, re.IGNORECASE)
            if colors:
                unique_colors = list(set(colors))[:3]
                genes.append(f"- 核心色彩：{', '.join(unique_colors)}")

            # 提取字体系统
            fonts = re.findall(r'font-family[^:]*:\s*([^;]+)', template_html, re.IGNORECASE)
            if fonts:
                genes.append(f"- 字体系统：{fonts[0]}")

            # 提取布局特征
            if 'display: flex' in template_html:
                genes.append("- 布局方式：Flexbox弹性布局")
            elif 'display: grid' in template_html:
                genes.append("- 布局方式：Grid网格布局")

            # 提取设计元素
            design_elements = []
            if 'border-radius' in template_html:
                design_elements.append("圆角设计")
            if 'box-shadow' in template_html:
                design_elements.append("阴影效果")
            if 'gradient' in template_html:
                design_elements.append("渐变背景")

            if design_elements:
                genes.append(f"- 设计元素：{', '.join(design_elements)}")

            # 提取间距模式
            paddings = re.findall(r'padding[^:]*:\s*([^;]+)', template_html, re.IGNORECASE)
            if paddings:
                genes.append(f"- 间距模式：{paddings[0]}")

        except Exception as e:
            logger.warning(f"基础提取设计基因时出错: {e}")
            genes.append("- 使用现代简洁的设计风格")

        return "\n".join(genes) if genes else "- 使用现代简洁的设计风格"

    async def _get_or_extract_style_genes(self, project_id: str, template_html: str, page_number: int) -> str:
        """获取或提取设计基因，只在第一页提取一次，后续复用"""
        import json
        import hashlib
        from pathlib import Path

        # 如果没有项目ID，直接提取
        if not project_id:
            if page_number == 1:
                return await self._extract_style_genes(template_html)
            else:
                return "- 使用现代简洁的设计风格\n- 保持页面整体一致性\n- 采用清晰的视觉层次"

        # 检查内存缓存
        if hasattr(self, '_cached_style_genes') and project_id in self._cached_style_genes:
            logger.info(f"从内存缓存获取项目 {project_id} 的设计基因")
            return self._cached_style_genes[project_id]

        # 检查文件缓存（如果有缓存目录配置）
        style_genes = None
        if hasattr(self, 'cache_dirs') and self.cache_dirs:
            cache_file = self.cache_dirs['style_genes'] / f"{project_id}_style_genes.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        style_genes = cache_data.get('style_genes')
                        logger.info(f"从文件缓存获取项目 {project_id} 的设计基因")
                except Exception as e:
                    logger.warning(f"读取设计基因缓存文件失败: {e}")

        # 如果没有缓存且是第一页，提取设计基因
        if not style_genes and page_number == 1:
            style_genes = await self._extract_style_genes(template_html)

            # 缓存到内存
            if not hasattr(self, '_cached_style_genes'):
                self._cached_style_genes = {}
            self._cached_style_genes[project_id] = style_genes

            # 缓存到文件（如果有缓存目录配置）
            if hasattr(self, 'cache_dirs') and self.cache_dirs:
                try:
                    cache_file = self.cache_dirs['style_genes'] / f"{project_id}_style_genes.json"
                    cache_data = {
                        'project_id': project_id,
                        'style_genes': style_genes,
                        'created_at': time.time(),
                        'template_hash': hashlib.md5(template_html.encode()).hexdigest()[:8]
                    }
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"第一页提取并缓存项目 {project_id} 的设计基因到文件")
                except Exception as e:
                    logger.warning(f"保存设计基因缓存文件失败: {e}")

            logger.info(f"第一页提取并缓存项目 {project_id} 的设计基因")

        elif not style_genes and page_number > 1:
            # 如果不是第一页且没有缓存的设计基因，使用默认设计基因
            style_genes = "- 使用现代简洁的设计风格\n- 保持页面整体一致性\n- 采用清晰的视觉层次"
            logger.warning(f"第{page_number}页未找到缓存的设计基因，使用默认设计基因（设计基因应在第一页提取）")

        return style_genes or "- 使用现代简洁的设计风格\n- 保持页面整体一致性\n- 采用清晰的视觉层次"

    async def _generate_unified_design_guide(self, slide_data: Dict[str, Any], page_number: int, total_pages: int) -> str:
        """生成统一的创意设计指导（合并创意变化指导和内容驱动的设计建议）"""
        try:
            # 构建AI提示词
            prompt = f"""
作为资深的PPT设计师，请为以下幻灯片生成全面的创意设计指导，包含创意变化指导和内容驱动的设计建议：

**完整幻灯片数据：**
{slide_data}

**页面位置：**第{page_number}页（共{total_pages}页）

请从以下角度生成统一的设计指导：

**A. 页面定位与创意策略**：
1. **页面定位策略**：基于页面在整个演示中的位置（开头/中间/结尾）
2. **视觉层次设计**：根据内容密度和重要性安排视觉层次
3. **创新布局建议**：在保持一致性基础上的创新布局方案

**B. 内容驱动的设计建议**：
4. **视觉组件建议**：根据内容特点推荐最合适的视觉组件（图表、卡片、时间线等）
5. **布局建议**：基于内容数量和复杂度推荐最佳布局方式
6. **数据可视化建议**：如果内容包含数据，推荐最合适的图表类型

**C. 视觉元素与交互体验**：
7. **视觉元素运用**：图标、色彩、形状等视觉元素的创意运用
8. **色彩和风格建议**：根据内容主题推荐合适的色彩搭配和设计风格
9. **交互体验优化**：提升用户阅读和理解体验的设计建议

要求：
- 建议要具体可操作，避免空泛描述
- 考虑内容特点和页面功能，每个建议都要说明选择理由
- 平衡创新性与一致性，建议要有创新性，避免千篇一律
- 每个建议以"- "开头，按照上述分类组织

请生成统一的创意设计指导：
"""

            # 调用AI生成指导
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=min(ai_config.max_tokens, 1000),  # 增加token限制以容纳更全面的指导
                temperature=0.7  # 适中温度平衡创意性和实用性
            )

            ai_guide = response.content.strip()

            # 如果AI生成失败，回退到基础指导
            if not ai_guide or len(ai_guide) < 50:
                return self._generate_fallback_unified_guide(slide_data, page_number, total_pages)

            return ai_guide

        except Exception as e:
            logger.warning(f"AI生成统一设计指导失败: {e}")
            # 回退到基础指导
            return self._generate_fallback_unified_guide(slide_data, page_number, total_pages)

    def _generate_fallback_unified_guide(self, slide_data: Dict[str, Any], page_number: int, total_pages: int) -> str:
        """生成回退的统一设计指导"""
        slide_type = slide_data.get('slide_type', 'content')
        content_points = slide_data.get('content_points', [])
        title = slide_data.get('title', '')

        guides = []

        # A. 页面定位与创意策略
        guides.append("**A. 页面定位与创意策略**")
        if page_number == 1:
            guides.extend([
                "- 开场页面：可以使用大胆的视觉冲击力，设置演示基调",
                "- 标题排版：尝试非对称布局、创意字体层次、动态视觉元素",
                "- 背景色保持统一：可以微小调整背景图案或渐变方向"
            ])
        elif page_number == total_pages:
            guides.extend([
                "- 结尾页面：设计总结性视觉框架，呼应开头元素",
                "- 行动号召：使用突出的视觉引导，如按钮、箭头等",
                "- 联系信息：创新的信息展示方式"
            ])
        else:
            guides.extend([
                "- 内容页面：根据信息密度调整布局复杂度",
                "- 渐进变化：在保持一致性基础上适度演进视觉风格",
                "- 重点突出：使用视觉层次强调关键信息"
            ])

        # B. 内容驱动的设计建议
        guides.append("\n**B. 内容驱动的设计建议**")
        if slide_type == 'title':
            guides.extend([
                "- 视觉组件：使用大型标题卡片、品牌标识、装饰性图形元素",
                "- 布局建议：采用居中对称布局，突出主标题的重要性"
            ])
        elif slide_type == 'content':
            if len(content_points) > 5:
                guides.extend([
                    "- 视觉组件：考虑分栏布局、卡片式设计或折叠展示",
                    "- 布局建议：使用网格布局或多列布局优化空间利用"
                ])
            elif len(content_points) <= 3:
                guides.extend([
                    "- 视觉组件：可以使用大型图标、插图或图表增强视觉效果",
                    "- 布局建议：采用宽松布局，增加字体大小和留白空间"
                ])
            guides.append("- 内容组织：尝试时间线、流程图、对比表格等创新方式")

        # C. 视觉元素与交互体验
        guides.append("\n**C. 视觉元素与交互体验**")
        guides.extend([
            "- 视觉元素：根据内容主题选择合适的图标和色彩搭配",
            "- 色彩建议：保持与整体设计基因一致的色彩方案",
            "- 交互体验：确保信息层次清晰，便于快速阅读和理解"
        ])

        # 根据标题内容添加特定建议
        if any(keyword in title.lower() for keyword in ['数据', '统计', '分析', 'data', 'analysis']):
            guides.append("- 数据可视化：推荐使用柱状图、饼图或折线图展示数据")

        return "\n".join(guides)







    def _build_slide_context(self, page_number: int, total_pages: int) -> str:
        """Build context information for slide generation with style consistency and innovation balance"""
        context_parts = []

        if page_number == 1 or page_number == total_pages:
            context_parts.append("**🌟 特殊页面设计要求 🌟**")

            if page_number == 1:
                context_parts.extend([
                    "这是首页，在保持和参考模板一直的风格前提下，采用亮眼具有吸引力的设计。要求：",
                    "- 使用渐变背景或特殊背景效果（如动态图案、光效等）",
                    "- 标题要有强烈的视觉冲击力：大字体、渐变色彩、文字阴影或发光效果",
                    "- 添加装饰性动画元素：浮动点、旋转圆圈、闪烁效果等",
                    "- 使用现代的设计元素：几何图形、线条装饰、光圈效果",
                    "- 整体色彩要丰富但协调，营造专业而吸引人的第一印象",
                    "- 可以添加CSS动画让页面更生动",
                    "- 不要出现图表"
                ])
            elif page_number == total_pages:
                context_parts.extend([
                    "这是结尾页，在保持和参考模板一直的风格前提下，采用亮眼具有吸引力的设计。要求：",
                    "- 使用特殊的背景效果：星空、光圈、渐变等营造庆祝感",
                    "- 标题要有庆祝和总结的感觉：发光效果、渐变文字、特殊字体",
                    "- 添加动态装饰元素：旋转圆圈、闪烁星星、彩色点缀、弹跳动画",
                    "- 呼应参考模板的设计元素，形成完整的视觉闭环",
                    "- 营造感谢和总结的温暖氛围",
                    "- 使用丰富的色彩和动画效果留下深刻的最后印象",
                    "- 不要出现图表"
                ])

            context_parts.append("")

        return "\n".join(context_parts) if context_parts else ""

    def _extract_style_template(self, existing_slides: List[Dict[str, Any]]) -> List[str]:
        """Extract a comprehensive style template from existing slides"""
        if not existing_slides:
            return []

        template_parts = []

        # Analyze all existing slides to extract common patterns
        color_schemes = []
        font_families = []
        layout_patterns = []
        design_elements = []

        for slide in existing_slides:
            html_content = slide.get('html_content', '')
            if html_content:
                # Extract style information
                style_info = self._extract_detailed_style_info(html_content)
                if style_info.get('colors'):
                    color_schemes.extend(style_info['colors'])
                if style_info.get('fonts'):
                    font_families.extend(style_info['fonts'])
                if style_info.get('layout'):
                    layout_patterns.append(style_info['layout'])
                if style_info.get('design_elements'):
                    design_elements.extend(style_info['design_elements'])

        # Build style template
        template_parts.append("**核心设计约束（必须保持一致）：**")

        # Color scheme
        if color_schemes:
            unique_colors = list(set(color_schemes))[:5]  # Top 5 colors
            template_parts.append(f"- 主色调：{', '.join(unique_colors)}")

        # Typography
        if font_families:
            unique_fonts = list(set(font_families))[:3]  # Top 3 fonts
            template_parts.append(f"- 字体系统：{', '.join(unique_fonts)}")

        # Layout patterns
        if layout_patterns:
            common_layout = self._analyze_common_layout(layout_patterns)
            template_parts.append(f"- 布局模式：{common_layout}")

        # Design elements
        if design_elements:
            unique_elements = list(set(design_elements))[:4]  # Top 4 elements
            template_parts.append(f"- 设计元素：{', '.join(unique_elements)}")

        template_parts.append("")
        template_parts.append("**可创新的设计空间：**")
        template_parts.append("- 内容布局结构（在保持整体风格下可调整）")
        template_parts.append("- 图标和装饰元素的选择和位置")
        template_parts.append("- 动画和交互效果的创新")
        template_parts.append("- 内容展示方式的优化（图表、列表、卡片等）")
        template_parts.append("- 视觉层次的重新组织")

        return template_parts

    def _extract_detailed_style_info(self, html_content: str) -> Dict[str, List[str]]:
        """Extract detailed style information from HTML content"""
        import re

        style_info = {
            'colors': [],
            'fonts': [],
            'layout': '',
            'design_elements': []
        }

        try:
            # Extract colors (more comprehensive)
            color_patterns = [
                r'color[^:]*:\s*([^;]+)',
                r'background[^:]*:\s*([^;]+)',
                r'border[^:]*:\s*([^;]+)',
                r'#[0-9a-fA-F]{3,6}',
                r'rgb\([^)]+\)',
                r'rgba\([^)]+\)'
            ]

            for pattern in color_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                style_info['colors'].extend([m.strip() for m in matches if m.strip()])

            # Extract fonts
            font_matches = re.findall(r'font-family[^:]*:\s*([^;]+)', html_content, re.IGNORECASE)
            style_info['fonts'] = [f.strip().replace('"', '').replace("'", '') for f in font_matches]

            # Analyze layout
            if 'display: flex' in html_content:
                style_info['layout'] = 'Flexbox布局'
            elif 'display: grid' in html_content:
                style_info['layout'] = 'Grid布局'
            elif 'position: absolute' in html_content:
                style_info['layout'] = '绝对定位布局'
            else:
                style_info['layout'] = '流式布局'

            # Extract design elements
            if 'border-radius' in html_content:
                style_info['design_elements'].append('圆角设计')
            if 'box-shadow' in html_content:
                style_info['design_elements'].append('阴影效果')
            if 'gradient' in html_content:
                style_info['design_elements'].append('渐变背景')
            if 'transform' in html_content:
                style_info['design_elements'].append('变换效果')
            if 'opacity' in html_content or 'rgba' in html_content:
                style_info['design_elements'].append('透明效果')

        except Exception as e:
            logger.warning(f"Error extracting detailed style info: {e}")

        return style_info

    def _analyze_common_layout(self, layout_patterns: List[str]) -> str:
        """Analyze common layout patterns"""
        if not layout_patterns:
            return "标准流式布局"

        # Count occurrences
        layout_counts = {}
        for layout in layout_patterns:
            layout_counts[layout] = layout_counts.get(layout, 0) + 1

        # Return most common layout
        return max(layout_counts.items(), key=lambda x: x[1])[0]

    def _get_innovation_guidelines(self, slide_type: str, page_number: int, total_pages: int) -> List[str]:
        """Get innovation guidelines based on slide type and position"""
        guidelines = []

        # Position-based innovation
        if page_number == 1:
            guidelines.extend([
                "- 标题页：可以创新的开场设计，如独特的标题排版、引人注目的视觉元素",
                "- 考虑使用大胆的视觉冲击力，为整个演示定下基调"
            ])
        elif page_number == total_pages:
            guidelines.extend([
                "- 结尾页：可以设计总结性的视觉元素，如回顾要点的创新布局",
                "- 考虑使用呼应开头的设计元素，形成完整的视觉闭环"
            ])
        else:
            guidelines.extend([
                "- 内容页：可以根据内容特点选择最适合的展示方式",
                "- 考虑使用渐进式的视觉变化，保持观众的注意力"
            ])

        # Content-based innovation
        content_innovations = {
            'title': [
                "- 可以尝试非对称布局、创意字体排列、背景图案变化",
                "- 考虑添加微妙的动画效果或视觉引导元素"
            ],
            'content': [
                "- 可以创新内容组织方式：卡片式、时间线、流程图、对比表格等",
                "- 考虑使用图标、插图、数据可视化来增强信息传达",
                "- 可以尝试分栏布局、重点突出框、引用样式等"
            ],
            'conclusion': [
                "- 可以设计总结性的视觉框架：要点回顾、行动号召、联系方式展示",
                "- 考虑使用视觉化的总结方式，如思维导图、关键词云等"
            ]
        }

        if slide_type in content_innovations:
            guidelines.extend(content_innovations[slide_type])
        else:
            guidelines.extend(content_innovations['content'])  # Default to content guidelines

        # General innovation principles
        guidelines.extend([
            "",
            "**创新原则：**",
            "- 在保持风格一致性的前提下，大胆尝试新的视觉表达方式",
            "- 根据内容的重要性和复杂度调整视觉层次",
            "- 考虑观众的阅读习惯和认知负荷",
            "- 确保创新不影响信息的清晰传达",
            "- 可以适度使用当前流行的设计趋势，但要与整体风格协调"
        ])

        return guidelines

    def _extract_style_info(self, html_content: str) -> List[str]:
        """Extract style information from HTML content for consistency reference"""
        import re
        style_info = []

        try:
            # Extract background colors
            bg_colors = re.findall(r'background[^:]*:\s*([^;]+)', html_content, re.IGNORECASE)
            if bg_colors:
                style_info.append(f"背景色调：{bg_colors[0][:50]}")

            # Extract color schemes
            colors = re.findall(r'color[^:]*:\s*([^;]+)', html_content, re.IGNORECASE)
            if colors:
                unique_colors = list(set(colors[:3]))  # Get first 3 unique colors
                style_info.append(f"主要颜色：{', '.join(unique_colors)}")

            # Extract font families
            fonts = re.findall(r'font-family[^:]*:\s*([^;]+)', html_content, re.IGNORECASE)
            if fonts:
                style_info.append(f"字体：{fonts[0][:50]}")

            # Extract font sizes
            font_sizes = re.findall(r'font-size[^:]*:\s*([^;]+)', html_content, re.IGNORECASE)
            if font_sizes:
                unique_sizes = list(set(font_sizes[:3]))  # Get first 3 unique sizes
                style_info.append(f"字体大小：{', '.join(unique_sizes)}")

            # Extract border radius for design style
            border_radius = re.findall(r'border-radius[^:]*:\s*([^;]+)', html_content, re.IGNORECASE)
            if border_radius:
                style_info.append(f"圆角样式：{border_radius[0]}")

            # Extract box shadow for depth effect
            box_shadow = re.findall(r'box-shadow[^:]*:\s*([^;]+)', html_content, re.IGNORECASE)
            if box_shadow:
                style_info.append(f"阴影效果：{box_shadow[0][:50]}")

            # Extract layout patterns
            if 'display: flex' in html_content:
                style_info.append("布局方式：Flexbox布局")
            elif 'display: grid' in html_content:
                style_info.append("布局方式：Grid布局")

            # Extract padding/margin patterns
            paddings = re.findall(r'padding[^:]*:\s*([^;]+)', html_content, re.IGNORECASE)
            if paddings:
                style_info.append(f"内边距：{paddings[0]}")

        except Exception as e:
            logger.warning(f"Error extracting style info: {e}")

        return style_info[:8]  # Limit to 8 most important style elements

    def _validate_html_completeness(self, html_content: str) -> Dict[str, Any]:
        """
        Validate HTML format correctness and tag closure using BeautifulSoup and lxml.

        This validator checks for:
        1. Presence of essential elements (<!DOCTYPE>, <html>, <head>, <body>) as warnings
        2. Correct structural order (<head> before <body>) as a warning
        3. Well-formedness and tag closure using strict parsing, reported as errors
        4. Unescaped special characters ('<' or '>') in text content as a warning

        Returns:
            Dict with 'is_complete', 'errors', 'warnings', 'missing_elements' keys
        """
        from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
        import warnings

        validation_result = {
            'is_complete': False,
            'errors': [],
            'warnings': [],
            'missing_elements': []  # 添加missing_elements字段
        }

        if not html_content or not html_content.strip():
            validation_result['errors'].append('HTML内容为空或仅包含空白字符')
            return validation_result

        # --- Primary Validation using Strict Parsing ---
        # This is the most reliable way to find malformed HTML and unclosed tags
        self._check_html_well_formedness(html_content, validation_result)

        # --- Secondary Validation using BeautifulSoup for structural best practices ---
        # This part runs even if there are syntax errors to provide more feedback
        try:
            # Suppress BeautifulSoup warnings about markup that looks like a file path
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
                # Use 'html.parser' for better compatibility, fallback to 'lxml' if available
                try:
                    soup = BeautifulSoup(html_content, 'lxml')
                except:
                    soup = BeautifulSoup(html_content, 'html.parser')

            # 1. Check for DOCTYPE declaration (Missing element)
            if not html_content.strip().lower().startswith('<!doctype'):
                validation_result['missing_elements'].append('doctype')

            # 2. Check for essential structural elements (Missing elements)
            essential_tags = {'html', 'head', 'body'}
            for tag_name in essential_tags:
                if not soup.find(tag_name):
                    validation_result['missing_elements'].append(tag_name)

            # 3. Check for correct structure order: <head> before <body> (Warning)
            head_tag = soup.find('head')
            body_tag = soup.find('body')

            if head_tag and body_tag:
                # Check if body tag has a preceding sibling named 'head'
                if not body_tag.find_previous_sibling('head'):
                    validation_result['warnings'].append('HTML结构顺序不正确：<body>标签出现在<head>标签之前')

            # 4. Check for unescaped special characters in text content (Warning)
            # soup.get_text() extracts only human-readable text
            text_content = soup.get_text()
            if '<' in text_content or '>' in text_content:
                validation_result['warnings'].append('文本内容中可能包含未转义的特殊字符（\'<\'或\'>\'）')

        except Exception as e:
            # Catch potential errors from BeautifulSoup itself
            validation_result['errors'].append(f'BeautifulSoup解析过程中发生意外错误: {e}')

        # Final determination of validity is based on the absence of critical errors
        # missing_elements are treated as warnings only, not errors
        validation_result['is_complete'] = len(validation_result['errors']) == 0

        return validation_result

    def _check_html_well_formedness(self, html_content: str, validation_result: Dict[str, Any]) -> None:
        """
        Uses lxml's strict parser to check if the HTML is well-formed.
        This is the definitive check for syntax errors like unclosed tags.
        Modifies the validation_result dictionary in place.
        """
        try:
            # Try to import lxml for strict parsing
            from lxml import etree

            # Encode the string to bytes for the lxml parser
            encoded_html = html_content.encode('utf-8')
            # Create a parser that does NOT recover from errors. This makes it strict.
            parser = etree.HTMLParser(recover=False, encoding='utf-8')
            etree.fromstring(encoded_html, parser)

        except ImportError:
            # lxml not available, fall back to basic regex checks
            logger.warning("lxml not available, using basic HTML validation")
            self._basic_html_syntax_check(html_content, validation_result)

        except Exception as e:
            # This error is triggered by unclosed tags, malformed tags, etc.
            # It's the most reliable indicator of a syntax problem.
            validation_result['errors'].append(f'HTML语法错误: {str(e)}')

    def _auto_fix_html_with_parser(self, html_content: str) -> str:
        """
        使用 lxml 的恢复解析器自动修复 HTML 错误

        Args:
            html_content: 原始 HTML 内容

        Returns:
            修复后的 HTML 内容，如果修复失败则返回原始内容
        """
        try:
            from lxml import etree

            # 首先检查原始 HTML 是否已经是有效的
            try:
                # 尝试严格解析
                encoded_html = html_content.encode('utf-8')
                strict_parser = etree.HTMLParser(recover=False, encoding='utf-8')
                etree.fromstring(encoded_html, strict_parser)
                # 如果严格解析成功，说明 HTML 已经是有效的，直接返回
                logger.debug("HTML 已经是有效的，无需修复")
                return html_content
            except:
                # 严格解析失败，需要修复
                pass

            # 创建一个启用恢复功能的解析器
            parser = etree.HTMLParser(recover=True, encoding='utf-8')
            tree = etree.fromstring(encoded_html, parser)

            # 保留 DOCTYPE 声明（如果存在）
            doctype_match = None
            import re
            doctype_pattern = r'<!DOCTYPE[^>]*>'
            doctype_match = re.search(doctype_pattern, html_content, re.IGNORECASE)

            # 将修复后的树转换回字符串
            fixed_html = etree.tostring(tree, encoding='unicode', method='html', pretty_print=True)

            # 如果原始 HTML 有 DOCTYPE，添加回去
            if doctype_match:
                doctype = doctype_match.group(0)
                if not fixed_html.lower().startswith('<!doctype'):
                    fixed_html = doctype + '\n' + fixed_html

            logger.info("使用 lxml 解析器自动修复 HTML 成功")
            return fixed_html

        except ImportError:
            logger.warning("lxml 不可用，无法使用解析器自动修复")
            return html_content

        except Exception as e:
            logger.warning(f"解析器自动修复失败: {str(e)}")
            return html_content

    def _basic_html_syntax_check(self, html_content: str, validation_result: Dict[str, Any]) -> None:
        """
        Basic HTML syntax checking when lxml is not available.
        Uses regex patterns to detect common HTML syntax errors.
        """
        import re
        from collections import Counter

        # Check for malformed tags (tags containing other tags)
        malformed_tags = re.findall(r'<[^>]*<[^>]*>', html_content)
        if malformed_tags:
            validation_result['errors'].append('发现格式错误的标签')

        # Check for unclosed critical HTML tags using tag counting
        # Define critical HTML tags that must be properly closed
        critical_tags = {'html', 'head', 'body', 'div', 'p', 'span'}

        # Find all opening and closing tags
        open_tags = re.findall(r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*>', html_content)
        close_tags = re.findall(r'</([a-zA-Z][a-zA-Z0-9]*)>', html_content)

        # Self-closing tags that don't need closing tags
        self_closing_tags = {'meta', 'link', 'img', 'br', 'hr', 'input', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr'}

        # Filter to only check critical tags, excluding self-closing tags
        open_tags_filtered = [tag.lower() for tag in open_tags
                             if tag.lower() in critical_tags and tag.lower() not in self_closing_tags]
        close_tags_lower = [tag.lower() for tag in close_tags if tag.lower() in critical_tags]

        # Count occurrences of each tag
        open_tag_counts = Counter(open_tags_filtered)
        close_tag_counts = Counter(close_tags_lower)

        # Check for unclosed critical tags
        unclosed_critical_tags = []
        for tag, open_count in open_tag_counts.items():
            close_count = close_tag_counts.get(tag, 0)
            if open_count > close_count:
                unclosed_critical_tags.append(f"{tag}({open_count - close_count}个未闭合)")

        if unclosed_critical_tags:
            validation_result['errors'].append(f'未闭合的关键HTML标签: {", ".join(unclosed_critical_tags)}')



    async def _generate_html_with_retry(self, context: str, system_prompt: str, slide_data: Dict[str, Any],
                                      page_number: int, total_pages: int, max_retries: int = 3) -> str:
        """Generate HTML with retry mechanism for incomplete responses"""

        for attempt in range(max_retries):
            try:
                logger.info(f"Generating HTML for slide {page_number}, attempt {attempt + 1}/{max_retries}")

                # Add retry-specific instructions to the context
                retry_context = context
                if attempt > 0:
                    retry_context += f"""

**重要提醒（第{attempt + 1}次尝试）：**
- 前面的尝试可能生成了不完整的HTML，请确保这次生成完整的HTML文档
- 必须包含完整的HTML结构：<!DOCTYPE html>, <html>, <head>, <body>等标签
- 确保所有标签都正确闭合
- 使用markdown代码块格式：```html\n[完整HTML代码]\n```
- 不要截断HTML代码，确保以</html>结束
"""

                # Use the existing ai_config from imports

                # Generate HTML
                response = await self.ai_provider.text_completion(
                    prompt=retry_context,
                    system_prompt=system_prompt,
                    max_tokens=min(ai_config.max_tokens, 16000),  # Increase token limit for retries
                    temperature=max(0.1, ai_config.temperature)  # Reduce temperature for retries
                )

                # Clean and extract HTML
                try:
                    html_content = self._clean_html_response(response.content)
                    if not html_content or len(html_content.strip()) < 50:
                        logger.warning(f"AI returned empty or too short HTML content for slide {page_number}")
                        continue
                except Exception as e:
                    logger.error(f"Error cleaning HTML response for slide {page_number}: {e}")
                    continue

                # Validate HTML completeness
                validation_result = self._validate_html_completeness(html_content)

                logger.info(f"HTML validation result for slide {page_number}, attempt {attempt + 1}: "
                          f"Complete: {validation_result['is_complete']}, "
                          f"Errors: {len(validation_result['errors'])}, "
                          f"Missing elements: {len(validation_result['missing_elements'])}")

                if validation_result['is_complete']:
                    # Log any missing elements as warnings only
                    if validation_result['missing_elements']:
                        logger.warning(f"Missing elements (warnings only): {', '.join(validation_result['missing_elements'])}")
                    logger.info(f"Successfully generated complete HTML for slide {page_number} on attempt {attempt + 1}")
                    return html_content
                else:
                    # Log validation issues
                    if validation_result['missing_elements']:
                        logger.warning(f"Missing elements (warnings only): {', '.join(validation_result['missing_elements'])}")
                    if validation_result['errors']:
                        logger.error(f"Validation errors: {'; '.join(validation_result['errors'])}")

                    # Only try to fix HTML with parser if there are actual errors (not just missing elements)
                    if validation_result['errors']:
                        # Try automatic parser-based fix
                        logger.info(f"🔧 Attempting automatic parser fix for slide {page_number}")
                        parser_fixed_html = self._auto_fix_html_with_parser(html_content)

                        # If parser actually changed something, return the fixed HTML directly
                        if parser_fixed_html != html_content:  # Only if parser actually changed something
                            logger.info(f"✅ Successfully fixed HTML with parser for slide {page_number}, returning fixed result")
                            return parser_fixed_html
                        else:
                            logger.info(f"🔧 Parser did not change HTML for slide {page_number}")

                        # If parser fix didn't change anything, retry generation
                        if attempt < max_retries - 1:
                            logger.info(f"🔄 HTML has errors after parser fix, retrying fresh generation for slide {page_number}...")
                            continue
                        else:
                            # Last attempt failed, use fallback
                            logger.warning(f"❌ All generation and parser fix attempts failed, using fallback for slide {page_number}")
                            return self._generate_fallback_slide_html(slide_data, page_number, total_pages)
                    else:
                        # No actual errors, just missing elements (warnings), so don't try to fix
                        logger.info(f"✅ HTML is valid with only missing element warnings for slide {page_number}")
                        return html_content

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in HTML generation attempt {attempt + 1} for slide {page_number}: {error_msg}")

                # 特殊处理JSON解析错误和其他AI响应错误
                if "Expecting value" in error_msg or "JSON" in error_msg:
                    logger.warning(f"JSON parsing error detected, this might be due to malformed AI response")
                    # 对于JSON错误，直接重试而不尝试修复
                    if attempt < max_retries - 1:
                        logger.info("Waiting 1 second before retry due to JSON parsing error...")
                        await asyncio.sleep(1)
                        continue

                if attempt == max_retries - 1:
                    # Last attempt failed with exception
                    logger.error(f"All attempts failed with errors, using fallback for slide {page_number}")
                    return self._generate_fallback_slide_html(slide_data, page_number, total_pages)
                continue

        # This should not be reached, but just in case
        return self._generate_fallback_slide_html(slide_data, page_number, total_pages)

    def _fix_incomplete_html(self, html_content: str, slide_data: Dict[str, Any],
                           page_number: int, total_pages: int) -> str:
        """Try to fix incomplete HTML by adding missing elements"""
        import re

        html_content = html_content.strip()

        # If HTML is completely empty or too short, return fallback
        if len(html_content) < 50:
            return self._generate_fallback_slide_html(slide_data, page_number, total_pages)

        # Check and add DOCTYPE if missing
        if not html_content.lower().startswith('<!doctype'):
            html_content = '<!DOCTYPE html>\n' + html_content

        # Check and add html tags if missing
        if not re.search(r'<html[^>]*>', html_content, re.IGNORECASE):
            html_content = html_content.replace('<!DOCTYPE html>', '<!DOCTYPE html>\n<html lang="zh-CN">')

        if not re.search(r'</html>', html_content, re.IGNORECASE):
            html_content += '\n</html>'

        # Check and add head section if missing
        if not re.search(r'<head[^>]*>', html_content, re.IGNORECASE):
            head_section = '''<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
</head>'''.format(slide_data.get('title', f'第{page_number}页'))

            # Insert head after html tag
            html_content = re.sub(r'(<html[^>]*>)', r'\1\n' + head_section, html_content, flags=re.IGNORECASE)
        else:
            # Check if head section is missing closing tag
            if not re.search(r'</head>', html_content, re.IGNORECASE):
                # Find the head opening tag and add missing elements
                head_match = re.search(r'<head[^>]*>', html_content, re.IGNORECASE)
                if head_match:
                    head_start = head_match.end()
                    # Check if charset is missing
                    if not re.search(r'<meta[^>]*charset[^>]*>', html_content, re.IGNORECASE):
                        charset_meta = '\n    <meta charset="UTF-8">'
                        html_content = html_content[:head_start] + charset_meta + html_content[head_start:]

                    # Add closing head tag before body
                    if '<body' in html_content.lower():
                        html_content = re.sub(r'(<body[^>]*>)', r'</head>\n\1', html_content, flags=re.IGNORECASE)
                    else:
                        # Add closing head tag after title or at the end of head content
                        if '</title>' in html_content.lower():
                            html_content = re.sub(r'(</title>)', r'\1\n</head>', html_content, flags=re.IGNORECASE)
                        else:
                            # Find a good place to close head
                            html_content = re.sub(r'(<html[^>]*>.*?<head[^>]*>.*?)(<body|$)', r'\1\n</head>\n\2', html_content, flags=re.IGNORECASE | re.DOTALL)

        # Check and add body tags if missing
        if not re.search(r'<body[^>]*>', html_content, re.IGNORECASE):
            # Find where to insert body tag (after </head> or after <html>)
            if '</head>' in html_content.lower():
                html_content = re.sub(r'(</head>)', r'\1\n<body>', html_content, flags=re.IGNORECASE)
            else:
                html_content = re.sub(r'(<html[^>]*>)', r'\1\n<body>', html_content, flags=re.IGNORECASE)

        if not re.search(r'</body>', html_content, re.IGNORECASE):
            # Insert </body> before </html>
            html_content = re.sub(r'(</html>)', r'</body>\n\1', html_content, flags=re.IGNORECASE)

        return html_content



    def _clean_html_response(self, raw_content: str) -> str:
        """Clean and extract HTML content from AI response with robust markdown handling"""
        import re

        if not raw_content:
            logger.warning("Received empty response from AI")
            return ""

        content = raw_content.strip()
        logger.debug(f"Raw AI response length: {len(content)}, preview: {content[:200]}...")

        # Check if response is suspiciously short or contains error indicators
        if len(content) < 100:
            logger.warning(f"AI response is very short ({len(content)} chars), might be incomplete")

        if any(error_indicator in content.lower() for error_indicator in ['error', 'sorry', 'cannot', 'unable']):
            logger.warning("AI response contains error indicators")

        # Step 1: Look for markdown code blocks first (most reliable)
        # Pattern to match ```html...``` blocks
        html_block_pattern = r'```html\s*\n(.*?)\n```'
        html_match = re.search(html_block_pattern, content, re.DOTALL | re.IGNORECASE)

        if html_match:
            extracted_html = html_match.group(1).strip()
            logger.debug("Found HTML in markdown code block")
            return extracted_html

        # Step 2: Look for generic code blocks ```...```
        generic_block_pattern = r'```\s*\n(.*?)\n```'
        generic_match = re.search(generic_block_pattern, content, re.DOTALL)

        if generic_match:
            potential_html = generic_match.group(1).strip()
            # Check if it looks like HTML
            if (potential_html.lower().startswith('<!doctype html') or
                potential_html.lower().startswith('<html')):
                logger.debug("Found HTML in generic code block")
                return potential_html

        # Step 3: Remove common AI response prefixes and try direct extraction
        prefixes_to_remove = [
            "这是生成的HTML代码：",
            "以下是HTML代码：",
            "HTML代码如下：",
            "生成的完整HTML页面：",
            "Here's the HTML code:",
            "The HTML code is:",
            "```html",
            "```",
        ]

        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()

        # Remove trailing markdown markers
        if content.endswith('```'):
            content = content[:-3].strip()

        # Step 4: Extract HTML using DOCTYPE or html tag patterns
        # Look for complete HTML document with DOCTYPE
        doctype_pattern = r'<!DOCTYPE html.*?</html>'
        doctype_match = re.search(doctype_pattern, content, re.DOTALL | re.IGNORECASE)

        if doctype_match:
            extracted_html = doctype_match.group(0)
            logger.debug("Found HTML using DOCTYPE pattern")
            return extracted_html

        # Look for html tag without DOCTYPE
        html_pattern = r'<html.*?</html>'
        html_match = re.search(html_pattern, content, re.DOTALL | re.IGNORECASE)

        if html_match:
            extracted_html = html_match.group(0)
            logger.debug("Found HTML using html tag pattern")
            return extracted_html

        # Step 5: Line-by-line extraction as fallback
        lines = content.split('\n')
        html_lines = []
        in_html = False

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Skip empty lines and common non-HTML prefixes
            if not line_stripped or line_stripped.startswith('#') or line_stripped.startswith('//'):
                continue

            # Start collecting when we see HTML start
            if line_lower.startswith('<!doctype') or line_lower.startswith('<html'):
                in_html = True
                html_lines.append(line)
                continue

            # Collect lines if we're in HTML
            if in_html:
                html_lines.append(line)

                # Stop when we see HTML end
                if line_lower.strip().endswith('</html>'):
                    break

        if html_lines:
            extracted_html = '\n'.join(html_lines)
            logger.debug("Found HTML using line-by-line extraction")
            return extracted_html

        # Step 6: If all else fails, check if content looks like HTML at all
        if '<' in content and '>' in content:
            logger.warning("Could not extract HTML using any method, but content contains HTML tags, returning cleaned content")
            return content
        else:
            logger.error("Content does not appear to contain HTML, returning empty string")
            return ""


    
    def _generate_fallback_slide_html(self, slide_data: Dict[str, Any], page_number: int, total_pages: int) -> str:
        """Generate fallback HTML for a slide with improved content visibility and special designs for title/thankyou slides"""
        title = slide_data.get('title', f'第{page_number}页')
        content_points = slide_data.get('content_points', [])
        slide_type = slide_data.get('slide_type', 'content')

        if slide_type == 'title':
            # 特殊设计的首页 - 亮眼的视觉效果
            content_html = f"""
            <div style="
                text-align: center;
                width: 100%;
                aspect-ratio: 16/9;
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin: 0 auto;
                box-sizing: border-box;
                position: relative;
                max-width: 1200px;
                padding: 3% 5%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                overflow: hidden;
            ">
                <!-- 动态背景装饰 -->
                <div style="
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
                    background-size: 50px 50px;
                    animation: float 20s ease-in-out infinite;
                    z-index: 1;
                "></div>

                <!-- 光效装饰 -->
                <div style="
                    position: absolute;
                    top: 20%;
                    right: 10%;
                    width: 200px;
                    height: 200px;
                    background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
                    border-radius: 50%;
                    z-index: 1;
                "></div>

                <div style="
                    position: absolute;
                    bottom: 30%;
                    left: 15%;
                    width: 150px;
                    height: 150px;
                    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
                    border-radius: 50%;
                    z-index: 1;
                "></div>

                <!-- 主要内容 -->
                <div style="position: relative; z-index: 2;">
                    <h1 style="
                        font-size: clamp(2rem, 5vw, 4rem);
                        color: #ffffff;
                        margin-bottom: clamp(30px, 4vh, 50px);
                        line-height: 1.2;
                        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
                        font-weight: 700;
                        letter-spacing: 1px;
                        background: linear-gradient(45deg, #ffffff, #f8f9fa);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                    ">{title}</h1>

                    <div style="
                        width: 80px;
                        height: 4px;
                        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
                        margin: 0 auto clamp(20px, 3vh, 30px) auto;
                        border-radius: 2px;
                    "></div>

                    <p style="
                        font-size: clamp(1.2rem, 3vw, 2rem);
                        color: rgba(255,255,255,0.9);
                        line-height: 1.4;
                        font-weight: 300;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    ">专业演示文稿</p>

                    <!-- 装饰性元素 -->
                    <div style="
                        margin-top: clamp(30px, 4vh, 50px);
                        display: flex;
                        justify-content: center;
                        gap: 15px;
                    ">
                        <div style="
                            width: 12px;
                            height: 12px;
                            background: rgba(255,255,255,0.6);
                            border-radius: 50%;
                            animation: pulse 2s ease-in-out infinite;
                        "></div>
                        <div style="
                            width: 12px;
                            height: 12px;
                            background: rgba(255,255,255,0.4);
                            border-radius: 50%;
                            animation: pulse 2s ease-in-out infinite 0.5s;
                        "></div>
                        <div style="
                            width: 12px;
                            height: 12px;
                            background: rgba(255,255,255,0.6);
                            border-radius: 50%;
                            animation: pulse 2s ease-in-out infinite 1s;
                        "></div>
                    </div>
                </div>

                <!-- 页码 -->
                <div style="
                    position: absolute;
                    bottom: 15px;
                    right: 20px;
                    color: rgba(255,255,255,0.8);
                    font-size: clamp(10px, 1.5vw, 14px);
                    font-weight: 500;
                    background: rgba(0,0,0,0.2);
                    padding: 6px 12px;
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                    z-index: 3;
                ">
                    第{page_number}页 / 共{total_pages}页
                </div>

                <style>
                    @keyframes float {{
                        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                        50% {{ transform: translateY(-20px) rotate(180deg); }}
                    }}
                    @keyframes pulse {{
                        0%, 100% {{ opacity: 0.6; transform: scale(1); }}
                        50% {{ opacity: 1; transform: scale(1.2); }}
                    }}
                </style>
            </div>
            """
        elif slide_type in ['thankyou', 'conclusion']:
            # 特殊设计的结尾页 - 亮眼的总结效果
            content_html = f"""
            <div style="
                text-align: center;
                width: 100%;
                aspect-ratio: 16/9;
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin: 0 auto;
                box-sizing: border-box;
                position: relative;
                max-width: 1200px;
                padding: 3% 5%;
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 50%, #9b59b6 100%);
                overflow: hidden;
            ">
                <!-- 星空背景效果 -->
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-image:
                        radial-gradient(2px 2px at 20px 30px, rgba(255,255,255,0.8), transparent),
                        radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.6), transparent),
                        radial-gradient(1px 1px at 90px 40px, rgba(255,255,255,0.9), transparent),
                        radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.7), transparent),
                        radial-gradient(2px 2px at 160px 30px, rgba(255,255,255,0.8), transparent);
                    background-repeat: repeat;
                    background-size: 200px 100px;
                    animation: sparkle 3s ease-in-out infinite;
                    z-index: 1;
                "></div>

                <!-- 光圈装饰 -->
                <div style="
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 300px;
                    height: 300px;
                    border: 2px solid rgba(255,255,255,0.2);
                    border-radius: 50%;
                    animation: rotate 10s linear infinite;
                    z-index: 1;
                "></div>

                <div style="
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 200px;
                    height: 200px;
                    border: 1px solid rgba(255,255,255,0.3);
                    border-radius: 50%;
                    animation: rotate 8s linear infinite reverse;
                    z-index: 1;
                "></div>

                <!-- 主要内容 -->
                <div style="position: relative; z-index: 2;">
                    <h1 style="
                        font-size: clamp(2.5rem, 6vw, 4.5rem);
                        color: #ffffff;
                        margin-bottom: clamp(20px, 3vh, 30px);
                        line-height: 1.2;
                        text-shadow: 0 4px 12px rgba(0,0,0,0.4);
                        font-weight: 700;
                        letter-spacing: 2px;
                        background: linear-gradient(45deg, #ffffff, #f39c12, #e74c3c);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        animation: glow 2s ease-in-out infinite alternate;
                    ">{title}</h1>

                    <!-- 装饰性分割线 -->
                    <div style="
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        margin: clamp(20px, 3vh, 30px) 0;
                    ">
                        <div style="
                            width: 50px;
                            height: 2px;
                            background: linear-gradient(90deg, transparent, #ffffff, transparent);
                        "></div>
                        <div style="
                            width: 20px;
                            height: 20px;
                            background: radial-gradient(circle, #ffffff 30%, transparent 30%);
                            margin: 0 15px;
                            border-radius: 50%;
                        "></div>
                        <div style="
                            width: 50px;
                            height: 2px;
                            background: linear-gradient(90deg, transparent, #ffffff, transparent);
                        "></div>
                    </div>

                    <p style="
                        font-size: clamp(1.2rem, 3vw, 1.8rem);
                        color: rgba(255,255,255,0.9);
                        line-height: 1.4;
                        font-weight: 300;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                        margin-bottom: clamp(30px, 4vh, 40px);
                    ">感谢您的聆听</p>

                    <!-- 内容要点（如果有） -->"""

            # 处理内容要点的显示
            if content_points:
                content_html += '''
                    <div style="
                        margin-top: clamp(20px, 3vh, 30px);
                        text-align: left;
                        max-width: 600px;
                        margin-left: auto;
                        margin-right: auto;
                    ">'''
                for point in content_points[:3]:
                    content_html += f'''
                        <div style="
                            background: rgba(255,255,255,0.1);
                            padding: 12px 20px;
                            margin: 10px 0;
                            border-radius: 25px;
                            border-left: 4px solid #f39c12;
                            color: rgba(255,255,255,0.9);
                            font-size: clamp(0.9rem, 2vw, 1.2rem);
                            backdrop-filter: blur(5px);
                        ">{point}</div>'''
                content_html += '''
                    </div>'''

            content_html += """

                    <!-- 结尾装饰 -->
                    <div style="
                        margin-top: clamp(30px, 4vh, 40px);
                        display: flex;
                        justify-content: center;
                        gap: 20px;
                    ">
                        <div style="
                            width: 8px;
                            height: 8px;
                            background: #e74c3c;
                            border-radius: 50%;
                            animation: bounce 1.5s ease-in-out infinite;
                        "></div>
                        <div style="
                            width: 8px;
                            height: 8px;
                            background: #f39c12;
                            border-radius: 50%;
                            animation: bounce 1.5s ease-in-out infinite 0.3s;
                        "></div>
                        <div style="
                            width: 8px;
                            height: 8px;
                            background: #27ae60;
                            border-radius: 50%;
                            animation: bounce 1.5s ease-in-out infinite 0.6s;
                        "></div>
                        <div style="
                            width: 8px;
                            height: 8px;
                            background: #3498db;
                            border-radius: 50%;
                            animation: bounce 1.5s ease-in-out infinite 0.9s;
                        "></div>
                    </div>
                </div>

                <!-- 页码 -->
                <div style="
                    position: absolute;
                    bottom: 15px;
                    right: 20px;
                    color: rgba(255,255,255,0.8);
                    font-size: clamp(10px, 1.5vw, 14px);
                    font-weight: 500;
                    background: rgba(0,0,0,0.2);
                    padding: 6px 12px;
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                    z-index: 3;
                ">
                    第{page_number}页 / 共{total_pages}页
                </div>

                <style>
                    @keyframes sparkle {{
                        0%, 100% {{ opacity: 0.8; }}
                        50% {{ opacity: 1; }}
                    }}
                    @keyframes rotate {{
                        from {{ transform: translate(-50%, -50%) rotate(0deg); }}
                        to {{ transform: translate(-50%, -50%) rotate(360deg); }}
                    }}
                    @keyframes glow {{
                        0% {{ text-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
                        100% {{ text-shadow: 0 4px 20px rgba(255,255,255,0.3), 0 0 30px rgba(255,255,255,0.2); }}
                    }}
                    @keyframes bounce {{
                        0%, 100% {{ transform: translateY(0); }}
                        50% {{ transform: translateY(-10px); }}
                    }}
                </style>
            </div>
            """
        else:
            points_html = ""
            if content_points:
                points_html = "<div style='max-height: 60vh; overflow-y: auto; padding-right: 10px;'><ul style='font-size: clamp(0.9rem, 2.5vw, 1.4rem); line-height: 1.5; margin: 0; padding-left: 1.5em;'>"
                for point in content_points:
                    points_html += f"<li style='margin-bottom: 0.8em; word-wrap: break-word;'>{point}</li>"
                points_html += "</ul></div>"

            content_html = f"""
            <div style="padding: 3% 5%; width: 100%; aspect-ratio: 16/9; box-sizing: border-box; margin: 0 auto; position: relative; max-width: 1200px; display: flex; flex-direction: column;">
                <h1 style="font-size: clamp(1.5rem, 4vw, 3rem); color: #2c3e50; margin-bottom: clamp(15px, 2vh, 25px); border-bottom: 3px solid #3498db; padding-bottom: 10px; line-height: 1.2; flex-shrink: 0;">{title}</h1>
                <div style="flex: 1; overflow: hidden; display: flex; flex-direction: column;">
                    {points_html}
                </div>
                <div style="position: absolute; bottom: 15px; right: 20px; color: #95a5a6; font-size: clamp(10px, 1.5vw, 14px); font-weight: 500; background: rgba(255,255,255,0.8); padding: 4px 8px; border-radius: 4px; z-index: 10;">
                    第{page_number}页 / 共{total_pages}页
                </div>
            </div>
            """

        return f"""
<!DOCTYPE html>
<html lang="zh-CN" style="height: 100%; display: flex; align-items: center; justify-content: center;">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #2c3e50;
            width: 1280px;
            height: 720px;
            position: relative;
            overflow: hidden;
        }}
    </style>
</head>
<body>
    {content_html}
</body>
</html>
        """

    def _combine_slides_to_full_html(self, slides_data: List[Dict[str, Any]], title: str) -> str:
        """Combine individual slides into a full presentation HTML and save to temp files"""
        try:
            # 验证输入数据
            if not slides_data:
                logger.warning("No slides data provided for combining")
                return self._generate_empty_presentation_html(title)

            if not title:
                title = "未命名演示"

            # Create temp directory for this presentation
            presentation_id = f"presentation_{uuid.uuid4().hex[:8]}"
            temp_dir = Path(tempfile.gettempdir()) / "landppt" / presentation_id
            temp_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Combining {len(slides_data)} slides into full HTML presentation")

            # Save individual slide HTML files
            slide_files = []
            for i, slide in enumerate(slides_data):
                # 安全地获取页码，如果没有则使用索引+1
                page_number = slide.get('page_number', i + 1)
                slide_filename = f"slide_{page_number}.html"
                slide_path = temp_dir / slide_filename

                # 确保HTML内容存在
                html_content = slide.get('html_content', '<div>空内容</div>')

                # Write slide HTML to file
                with open(slide_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                # Create relative path for HTTP access
                relative_path = f"{presentation_id}/{slide_filename}"
                slide_files.append({
                    'page_number': page_number,
                    'filename': slide_filename,
                    'relative_path': relative_path
                })

            # Generate slides HTML using base64 data URLs to avoid encoding issues
            slides_html = ""
            for i, slide in enumerate(slides_data):
                # 安全地获取页码和HTML内容
                page_number = slide.get('page_number', i + 1)
                html_content = slide.get('html_content', '<div>空内容</div>')

                # Encode HTML content as base64 data URL
                encoded_html = self._encode_html_to_base64(html_content)
                data_url = f"data:text/html;charset=utf-8;base64,{encoded_html}"

                slides_html += f'''
                <div class="slide" id="slide-{page_number}" style="display: {'block' if i == 0 else 'none'};">
                    <iframe src="{data_url}"
                            style="width: 100%; height: 100%; border: none;"></iframe>
                </div>
                '''

            return f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: #000;
        }}
        .slide {{
            width: 100%;
            max-width: 1200px;
            aspect-ratio: 16/9;
            position: relative;
            margin: 0 auto;
        }}
        .navigation {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 25px;
        }}
        .nav-btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 15px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
        }}
        .nav-btn:hover {{
            background: #2980b9;
        }}
        .nav-btn:disabled {{
            background: #95a5a6;
            cursor: not-allowed;
        }}
        .slide-counter {{
            color: white;
            margin: 0 15px;
        }}
    </style>
</head>
<body>
    {slides_html}

    <div class="navigation">
        <button class="nav-btn" onclick="previousSlide()">⬅️ 上一页</button>
        <span class="slide-counter" id="slideCounter">1 / {len(slides_data)}</span>
        <button class="nav-btn" onclick="nextSlide()">下一页 ➡️</button>
    </div>

    <script>
        let currentSlide = 0;
        const totalSlides = {len(slides_data)};

        // No need for initialization - iframes already have src set to file paths

        function showSlide(index) {{
            document.querySelectorAll('.slide').forEach(slide => slide.style.display = 'none');
            const targetSlide = document.getElementById('slide-' + (index + 1));
            if (targetSlide) {{
                targetSlide.style.display = 'block';
            }}
            document.getElementById('slideCounter').textContent = (index + 1) + ' / ' + totalSlides;
        }}

        function nextSlide() {{
            if (currentSlide < totalSlides - 1) {{
                currentSlide++;
                showSlide(currentSlide);
            }}
        }}

        function previousSlide() {{
            if (currentSlide > 0) {{
                currentSlide--;
                showSlide(currentSlide);
            }}
        }}

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowRight') nextSlide();
            if (e.key === 'ArrowLeft') previousSlide();
        }});

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            showSlide(0);
        }});
    </script>
</body>
</html>
            '''

        except Exception as e:
            logger.error(f"Error combining slides to full HTML: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_empty_presentation_html(title)

    def _generate_empty_presentation_html(self, title: str) -> str:
        """Generate empty presentation HTML as fallback"""
        return f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: #f0f0f0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }}
        .empty-message {{
            text-align: center;
            color: #666;
            font-size: 24px;
        }}
    </style>
</head>
<body>
    <div class="empty-message">
        <h1>暂无幻灯片内容</h1>
        <p>请先生成幻灯片内容</p>
    </div>
</body>
</html>
        '''

    def _encode_html_for_iframe(self, html_content: str) -> str:
        """Encode HTML content for iframe src"""
        import urllib.parse
        return urllib.parse.quote(html_content)

    def _encode_html_to_base64(self, html_content: str) -> str:
        """Encode HTML content to base64 for safe JavaScript transmission"""
        import base64
        return base64.b64encode(html_content.encode('utf-8')).decode('ascii')

    async def _design_theme(self, scenario: str, language: str) -> Dict[str, Any]:
        """Design theme configuration based on scenario"""
        theme_configs = {
            "general": {
                "primary_color": "#3498db",
                "secondary_color": "#2c3e50",
                "accent_color": "#e74c3c",
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "font_family": "Arial, sans-serif",
                "style": "professional"
            },
            "tourism": {
                "primary_color": "#27ae60",
                "secondary_color": "#16a085",
                "accent_color": "#f39c12",
                "background": "linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)",
                "font_family": "Georgia, serif",
                "style": "vibrant"
            },
            "education": {
                "primary_color": "#9b59b6",
                "secondary_color": "#8e44ad",
                "accent_color": "#f1c40f",
                "background": "linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%)",
                "font_family": "Comic Sans MS, cursive",
                "style": "playful"
            },
            "analysis": {
                "primary_color": "#34495e",
                "secondary_color": "#2c3e50",
                "accent_color": "#e67e22",
                "background": "linear-gradient(135deg, #636e72 0%, #2d3436 100%)",
                "font_family": "Helvetica, sans-serif",
                "style": "analytical"
            },
            "history": {
                "primary_color": "#8b4513",
                "secondary_color": "#a0522d",
                "accent_color": "#daa520",
                "background": "linear-gradient(135deg, #d63031 0%, #74b9ff 100%)",
                "font_family": "Times New Roman, serif",
                "style": "classical"
            },
            "technology": {
                "primary_color": "#6c5ce7",
                "secondary_color": "#a29bfe",
                "accent_color": "#00cec9",
                "background": "linear-gradient(135deg, #00cec9 0%, #6c5ce7 100%)",
                "font_family": "Roboto, sans-serif",
                "style": "modern"
            },
            "business": {
                "primary_color": "#1f4e79",
                "secondary_color": "#2980b9",
                "accent_color": "#f39c12",
                "background": "linear-gradient(135deg, #2980b9 0%, #1f4e79 100%)",
                "font_family": "Arial, sans-serif",
                "style": "corporate"
            }
        }

        return theme_configs.get(scenario, theme_configs["general"])

    def _normalize_slide_type(self, slide_type: str) -> str:
        """Normalize slide type to supported values"""
        type_mapping = {
            "agenda": "agenda",
            "section": "section",
            "conclusion": "conclusion",
            "thankyou": "thankyou",
            "title": "title",
            "content": "content",
            "image": "image",
            "chart": "chart",
            "list": "list",
            # Handle any other types by mapping to content
            "overview": "content",
            "summary": "conclusion",
            "intro": "content",
            "ending": "thankyou"
        }
        return type_mapping.get(slide_type, "content")

    async def _generate_enhanced_content(self, outline: PPTOutline, request: PPTGenerationRequest) -> List[SlideContent]:
        """Generate enhanced content for each slide"""
        enhanced_slides = []

        for slide_data in outline.slides:
            try:
                # Generate detailed content using AI
                content = await self.generate_slide_content(
                    slide_data["title"],
                    request.scenario,
                    request.topic,
                    request.language
                )

                # Create enhanced slide content
                slide_content = SlideContent(
                    type=self._normalize_slide_type(slide_data.get("type", "content")),
                    title=slide_data["title"],
                    subtitle=slide_data.get("subtitle", ""),
                    content=content,
                    bullet_points=self._extract_bullet_points(content),
                    image_suggestions=await self._suggest_images(slide_data["title"], request.scenario),
                    layout="default"
                )

                enhanced_slides.append(slide_content)

            except Exception as e:
                logger.error(f"Error generating content for slide {slide_data['title']}: {e}")
                # Fallback to basic content
                slide_content = SlideContent(
                    type=self._normalize_slide_type(slide_data.get("type", "content")),
                    title=slide_data["title"],
                    subtitle=slide_data.get("subtitle", ""),
                    content=slide_data.get("content", ""),
                    layout="default"
                )
                enhanced_slides.append(slide_content)

        return enhanced_slides

    async def _verify_layout(self, slides: List[SlideContent], theme_config: Dict[str, Any]) -> List[SlideContent]:
        """Verify and optimize slide layouts"""
        verified_slides = []

        for slide in slides:
            # Create a copy to avoid modifying original
            verified_slide = SlideContent(**slide.model_dump())

            # Apply layout optimizations based on content type
            if slide.type == "title":
                verified_slide.layout = "title_layout"
            elif slide.type == "agenda":
                verified_slide.layout = "agenda_layout"
            elif slide.type == "section":
                verified_slide.layout = "section_layout"
            elif slide.type == "conclusion":
                verified_slide.layout = "conclusion_layout"
            elif slide.type == "thankyou":
                verified_slide.layout = "thankyou_layout"
            elif slide.type == "content" and slide.bullet_points:
                verified_slide.layout = "bullet_layout"
            elif slide.type == "content" and slide.image_suggestions:
                verified_slide.layout = "image_content_layout"
            elif slide.type == "list":
                verified_slide.layout = "list_layout"
            elif slide.type == "chart":
                verified_slide.layout = "chart_layout"
            elif slide.type == "image":
                verified_slide.layout = "image_layout"
            else:
                verified_slide.layout = "default_layout"

            # Ensure content length is appropriate
            if verified_slide.content and len(verified_slide.content) > 500:
                verified_slide.content = verified_slide.content[:500] + "..."

            verified_slides.append(verified_slide)

        return verified_slides

    async def _generate_html_output(self, slides: List[SlideContent], theme_config: Dict[str, Any]) -> str:
        """Generate HTML output for slides"""
        try:
            # Convert SlideContent to dict format for parent class
            slides_dict = []
            for i, slide in enumerate(slides):
                slide_dict = {
                    "id": i + 1,
                    "type": slide.type,
                    "title": slide.title,
                    "subtitle": slide.subtitle or "",
                    "content": slide.content or "",
                    "bullet_points": slide.bullet_points or [],
                    "layout": slide.layout
                }
                slides_dict.append(slide_dict)

            # Create a temporary outline for the parent class method
            from ..api.models import PPTOutline
            temp_outline = PPTOutline(
                title="Generated PPT",
                slides=slides_dict,
                metadata={"theme_config": theme_config}
            )

            # Use parent class method to generate HTML
            html_content = await self.generate_slides_from_outline(temp_outline, "general")
            return html_content

        except Exception as e:
            logger.error(f"Error generating HTML output: {e}")
            # Fallback to basic HTML
            return self._generate_basic_html(slides, theme_config)

    def _extract_bullet_points(self, content: str) -> List[str]:
        """Extract bullet points from content"""
        if not content:
            return []

        bullet_points = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                bullet_points.append(line[1:].strip())
            elif re.match(r'^\d+\.', line):
                bullet_points.append(line.split('.', 1)[1].strip())

        return bullet_points[:5]  # Limit to 5 bullet points

    async def _suggest_images(self, slide_title: str, scenario: str) -> List[str]:
        """Suggest images for a slide based on title and scenario"""
        # This would integrate with image search APIs in a real implementation
        # For now, return scenario-based suggestions
        image_suggestions = {
            "general": ["business-meeting.jpg", "professional-chart.jpg", "office-space.jpg"],
            "tourism": ["landscape.jpg", "travel-destination.jpg", "cultural-site.jpg"],
            "education": ["classroom.jpg", "learning-materials.jpg", "students.jpg"],
            "analysis": ["data-visualization.jpg", "analytics-dashboard.jpg", "research.jpg"],
            "history": ["historical-artifact.jpg", "ancient-building.jpg", "timeline.jpg"],
            "technology": ["innovation.jpg", "digital-technology.jpg", "futuristic.jpg"],
            "business": ["corporate-building.jpg", "business-strategy.jpg", "team-meeting.jpg"]
        }

        return image_suggestions.get(scenario, image_suggestions["general"])

    def _generate_basic_html(self, slides: List[SlideContent], theme_config: Dict[str, Any]) -> str:
        """Generate basic HTML as fallback"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>PPT Presentation</title>",
            "<style>",
            "body { margin: 0; padding: 0; font-family: " + theme_config.get('font_family', 'Arial, sans-serif') + "; }",
            ".presentation-container { width: 1280px; height: 720px; margin: 0 auto; position: relative; }",
            ".slide { width: 1280px; height: 720px; background: " + theme_config.get('background', '#f0f0f0') + "; padding: 40px; box-sizing: border-box; position: relative; }",
            ".title { color: " + theme_config.get('primary_color', '#333') + "; font-size: 2em; margin-bottom: 20px; }",
            ".content { color: " + theme_config.get('secondary_color', '#666') + "; font-size: 1.2em; line-height: 1.6; }",
            ".page-number { position: absolute; bottom: 20px; right: 20px; color: #999; font-size: 0.9em; }",
            "@media (max-width: 1280px) { .presentation-container, .slide { width: 100vw; height: 56.25vw; max-height: 100vh; } }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='presentation-container'>"
        ]

        for i, slide in enumerate(slides):
            html_parts.extend([
                f"<div class='slide' id='slide-{i+1}'>",
                f"<h1 class='title'>{slide.title}</h1>",
                f"<div class='content'>{slide.content or ''}</div>",
                f"<div class='page-number'>{i+1}</div>",
                "</div>"
            ])

        html_parts.extend(["</div>", "</body>", "</html>"])

        return "\n".join(html_parts)

    # Project management integration methods
    async def get_project_todo_board(self, project_id: str) -> Optional[TodoBoard]:
        """Get TODO board for a project"""
        return await self.project_manager.get_todo_board(project_id)

    async def update_project_stage(self, project_id: str, stage_id: str, status: str,
                                 progress: float = None, result: Dict[str, Any] = None) -> bool:
        """Update project stage status"""
        return await self.project_manager.update_stage_status(
            project_id, stage_id, status, progress, result
        )

    async def reset_stages_from(self, project_id: str, stage_id: str) -> bool:
        """Reset all stages from the specified stage onwards"""
        try:
            project = await self.project_manager.get_project(project_id)
            if not project or not project.todo_board:
                return False

            # Find the stage index
            stage_index = -1
            for i, stage in enumerate(project.todo_board.stages):
                if stage.id == stage_id:
                    stage_index = i
                    break

            if stage_index == -1:
                logger.error(f"Stage {stage_id} not found in project {project_id}")
                return False

            # Reset all stages from the specified stage onwards
            for i in range(stage_index, len(project.todo_board.stages)):
                stage = project.todo_board.stages[i]
                stage.status = "pending"
                stage.progress = 0.0
                stage.result = None
                stage.updated_at = time.time()

            # Update current stage index
            project.todo_board.current_stage_index = stage_index

            # Recalculate overall progress
            completed_stages = sum(1 for s in project.todo_board.stages if s.status == "completed")
            project.todo_board.overall_progress = (completed_stages / len(project.todo_board.stages)) * 100
            project.todo_board.updated_at = time.time()

            # Clear related project data based on the stage being reset
            if stage_id == "outline_generation":
                # Reset outline and all subsequent data
                project.outline = None
                project.slides_html = None
                project.slides_data = None
            elif stage_id == "ppt_creation":
                # Reset only PPT data, keep outline
                project.slides_html = None
                project.slides_data = None

            project.updated_at = time.time()

            # 保存重置后的项目状态到数据库
            try:
                from .db_project_manager import DatabaseProjectManager
                db_manager = DatabaseProjectManager()

                # 更新项目状态
                await db_manager.update_project_status(project_id, "in_progress")

                # 重置相关阶段状态到数据库
                for i in range(stage_index, len(project.todo_board.stages)):
                    stage = project.todo_board.stages[i]
                    await db_manager.update_stage_status(
                        project_id,
                        stage.id,
                        "pending",
                        0.0,
                        None
                    )

                # 如果重置了大纲生成阶段，清除数据库中的大纲和幻灯片数据
                if stage_id == "outline_generation":
                    # 清除大纲数据
                    await db_manager.save_project_outline(project_id, None)
                    # 清除幻灯片数据
                    await db_manager.save_project_slides(project_id, "", [])
                elif stage_id == "ppt_creation":
                    # 只清除幻灯片数据，保留大纲
                    await db_manager.save_project_slides(project_id, "", [])

                logger.info(f"Successfully saved reset stages to database for project {project_id}")

            except Exception as save_error:
                logger.error(f"Failed to save reset stages to database: {save_error}")
                # 继续执行，因为内存中的数据已经重置

            logger.info(f"Reset stages from {stage_id} onwards for project {project_id}")
            return True

        except Exception as e:
            logger.error(f"Error resetting stages from {stage_id}: {e}")
            return False

    async def start_workflow_from_stage(self, project_id: str, stage_id: str) -> bool:
        """Start workflow execution from a specific stage"""
        try:
            project = await self.project_manager.get_project(project_id)
            if not project:
                return False

            # Check if requirements are confirmed (needed for all stages except requirements_confirmation)
            if stage_id != "requirements_confirmation" and not project.confirmed_requirements:
                logger.error(f"Cannot start from stage {stage_id}: requirements not confirmed")
                return False

            # Start the workflow from the specified stage
            # This will be handled by the existing workflow execution logic
            # For now, just mark the stage as ready to start
            await self.project_manager.update_stage_status(
                project_id, stage_id, "pending", 0.0
            )

            logger.info(f"Workflow ready to start from stage {stage_id} for project {project_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting workflow from stage {stage_id}: {e}")
            return False

    async def regenerate_slide(self, project_id: str, slide_index: int,
                             request: PPTGenerationRequest) -> Optional[SlideContent]:
        """Regenerate a specific slide"""
        try:
            project = await self.project_manager.get_project(project_id)
            if not project or not project.outline:
                return None

            if slide_index >= len(project.outline.slides):
                return None

            slide_data = project.outline.slides[slide_index]

            # Generate new content
            content = await self.generate_slide_content(
                slide_data["title"],
                request.scenario,
                request.topic,
                request.language
            )

            # Create new slide content
            new_slide = SlideContent(
                type=self._normalize_slide_type(slide_data.get("type", "content")),
                title=slide_data["title"],
                subtitle=slide_data.get("subtitle", ""),
                content=content,
                bullet_points=self._extract_bullet_points(content),
                image_suggestions=await self._suggest_images(slide_data["title"], request.scenario),
                layout="default"
            )

            return new_slide

        except Exception as e:
            logger.error(f"Error regenerating slide: {e}")
            return None

    async def lock_slide(self, project_id: str, slide_index: int) -> bool:
        """Lock a slide to prevent regeneration"""
        # This would be implemented with proper slide state management
        # For now, return True as placeholder
        return True

    async def unlock_slide(self, project_id: str, slide_index: int) -> bool:
        """Unlock a slide to allow regeneration"""
        # This would be implemented with proper slide state management
        # For now, return True as placeholder
        return True

    def _standardize_summeryfile_outline(self, summeryfile_outline: Dict[str, Any]) -> Dict[str, Any]:
        """
        将summeryanyfile生成的大纲格式标准化为LandPPT格式

        Args:
            summeryfile_outline: summeryanyfile生成的大纲数据

        Returns:
            标准化后的LandPPT格式大纲
        """
        try:
            # 提取基本信息
            title = summeryfile_outline.get("title", "PPT大纲")
            slides_data = summeryfile_outline.get("slides", [])
            metadata = summeryfile_outline.get("metadata", {})

            # 转换slides格式
            standardized_slides = []

            for slide in slides_data:
                # 优先使用content_points字段，如果没有则尝试从content字段提取
                content_points = slide.get("content_points", [])

                # 如果content_points为空或不是列表，尝试从content字段提取
                if not content_points or not isinstance(content_points, list):
                    content = slide.get("content", "")
                    content_points = []

                    if content:
                        # 分割content为要点列表
                        lines = content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line:
                                # 移除bullet point符号
                                line = re.sub(r'^[•\-\*]\s*', '', line)
                                if line:
                                    content_points.append(line)

                # 如果仍然没有content_points，使用默认值
                if not content_points:
                    content_points = ["内容要点"]

                # 标准化slide_type
                slide_type = slide.get("slide_type", slide.get("type", "content"))
                page_number = slide.get("page_number", slide.get("id", 1))
                title_text = slide.get("title", "").lower()

                # 更智能的slide_type识别 - 保留summeryanyfile的原始类型
                if slide_type not in ["title", "content", "agenda", "thankyou", "conclusion"]:
                    if page_number == 1 or "标题" in title_text or "title" in title_text:
                        slide_type = "title"
                    elif "目录" in title_text or "agenda" in title_text or "大纲" in title_text:
                        slide_type = "agenda"
                    elif "谢谢" in title_text or "thank" in title_text or "致谢" in title_text:
                        slide_type = "thankyou"
                    elif "总结" in title_text or "结论" in title_text or "conclusion" in title_text or "summary" in title_text:
                        slide_type = "conclusion"
                    else:
                        slide_type = "content"
                else:
                    # 即使已经有slide_type，也要检查特殊页面类型
                    if ("目录" in title_text or "agenda" in title_text or "大纲" in title_text) and slide_type == "content":
                        slide_type = "agenda"
                    elif ("谢谢" in title_text or "thank" in title_text or "致谢" in title_text) and slide_type == "content":
                        slide_type = "thankyou"
                    elif ("总结" in title_text or "结论" in title_text or "conclusion" in title_text or "summary" in title_text) and slide_type == "content":
                        slide_type = "conclusion"

                # 映射slide_type到enhanced_ppt_service期望的type字段
                type_mapping = {
                    "title": "title",
                    "content": "content",
                    "conclusion": "thankyou",
                    "agenda": "agenda"
                }
                mapped_type = type_mapping.get(slide_type, "content")

                # 构建标准化的slide对象
                standardized_slide = {
                    "page_number": slide.get("page_number", slide.get("id", len(standardized_slides) + 1)),
                    "title": slide.get("title", f"第{len(standardized_slides) + 1}页"),
                    "content_points": content_points,
                    "slide_type": slide_type,  # 保留原始字段
                    "type": mapped_type,  # 添加enhanced_ppt_service期望的type字段
                    "description": slide.get("description", "")  # 保留描述字段
                }

                # 如果原始slide包含chart_config，则保留
                if "chart_config" in slide and slide["chart_config"]:
                    standardized_slide["chart_config"] = slide["chart_config"]

                standardized_slides.append(standardized_slide)

            # 构建标准化的metadata
            standardized_metadata = {
                "generated_with_summeryfile": True,
                "page_count_settings": {
                    "mode": metadata.get("page_count_mode", "ai_decide"),
                    "min_pages": None,
                    "max_pages": None,
                    "fixed_pages": None
                },
                "actual_page_count": len(standardized_slides),
                "generated_at": time.time(),
                "original_metadata": metadata  # 保留原始元数据
            }

            # 如果原始metadata中有页数设置，尝试转换
            if "total_pages" in metadata:
                standardized_metadata["page_count_settings"]["expected_pages"] = metadata["total_pages"]

            # 构建标准化的大纲
            standardized_outline = {
                "title": title,
                "slides": standardized_slides,
                "metadata": standardized_metadata
            }

            logger.info(f"Successfully standardized summeryfile outline: {title}, {len(standardized_slides)} slides")
            return standardized_outline

        except Exception as e:
            logger.error(f"Error standardizing summeryfile outline: {e}")
            # 返回默认结构
            return {
                "title": "PPT大纲",
                "slides": [
                    {
                        "page_number": 1,
                        "title": "标题页",
                        "content_points": ["演示标题", "演示者", "日期"],
                        "slide_type": "title",
                        "type": "title",  # 添加type字段
                        "description": "PPT标题页"
                    }
                ],
                "metadata": {
                    "generated_with_summeryfile": True,
                    "page_count_settings": {"mode": "ai_decide"},
                    "actual_page_count": 1,
                    "generated_at": time.time(),
                    "error": str(e)
                }
            }

    async def generate_outline_from_file(self, request) -> Dict[str, Any]:
        """使用summeryanyfile从文件生成PPT大纲"""
        # 导入必要的模块
        from ..api.models import FileOutlineGenerationResponse

        try:
            # 尝试使用summeryanyfile生成大纲
            logger.info(f"开始使用summeryanyfile从文件生成PPT大纲: {request.filename}")

            try:
                # 导入summeryanyfile模块
                from summeryanyfile.generators.ppt_generator import PPTOutlineGenerator
                from summeryanyfile.core.models import ProcessingConfig, ChunkStrategy

                # 获取最新的AI配置
                current_ai_config = self._get_current_ai_config()
                logger.info(f"使用最新AI配置: provider={current_ai_config['llm_provider']}, model={current_ai_config['llm_model']}")

                # 创建配置 - 使用最新的AI配置
                config = ProcessingConfig(
                    max_slides=self._get_max_slides_from_request(request),
                    chunk_size=self._get_chunk_size_from_request(request),
                    chunk_strategy=self._get_chunk_strategy_from_request(request),
                    llm_model=current_ai_config["llm_model"],
                    llm_provider=current_ai_config["llm_provider"],
                    temperature=current_ai_config["temperature"],
                    max_tokens=current_ai_config["max_tokens"],
                    target_language=request.language  # 使用用户在表单中选择的语言
                )

                # 根据file_processing_mode设置use_magic_pdf参数
                use_magic_pdf = request.file_processing_mode == "magic_pdf"
                logger.info(f"文件处理模式: {request.file_processing_mode}, 使用Magic-PDF: {use_magic_pdf}")

                # 创建生成器并传递API配置和文件处理模式
                # 设置缓存目录到项目根目录下的temp文件夹
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent.parent
                cache_dir = project_root / "temp" / "summeryanyfile_cache"

                generator = PPTOutlineGenerator(config, use_magic_pdf=use_magic_pdf, cache_dir=str(cache_dir))

                # 设置API配置到LLM管理器
                self._configure_summeryfile_api(generator)

                # 从文件生成大纲
                logger.info(f"正在使用summeryanyfile处理文件: {request.file_path}")
                outline = await generator.generate_from_file(request.file_path)

                logger.info(f"summeryanyfile生成成功: {outline.title}, 共{outline.total_pages}页")

                # 转换为LandPPT格式 - 使用新的标准化函数
                summeryfile_dict = outline.to_dict()
                landppt_outline = self._standardize_summeryfile_outline(summeryfile_dict)

                # 验证和修复文件生成的大纲
                # 构建confirmed_requirements用于验证
                confirmed_requirements = {
                    'topic': request.topic or landppt_outline.get('title', '文档演示'),
                    'target_audience': getattr(request, 'target_audience', '通用受众'),
                    'focus_content': getattr(request, 'focus_content', []),
                    'tech_highlights': getattr(request, 'tech_highlights', []),
                    'page_count_settings': {
                        'mode': request.page_count_mode,
                        'min_pages': getattr(request, 'min_pages', None),
                        'max_pages': getattr(request, 'max_pages', None),
                        'fixed_pages': getattr(request, 'fixed_pages', None)
                    }
                }

                landppt_outline = await self._validate_and_repair_outline_json(landppt_outline, confirmed_requirements)

                # 获取文件信息
                file_info = {
                    "filename": request.filename,
                    "file_path": request.file_path,
                    "processing_mode": request.file_processing_mode,
                    "analysis_depth": request.content_analysis_depth,
                    "used_summeryanyfile": True
                }

                # 获取处理统计信息
                processing_stats = {
                    "total_pages": outline.total_pages,
                    "page_count_mode": request.page_count_mode,
                    "slides_count": len(outline.slides),
                    "processing_time": "完成",
                    "generator": "summeryanyfile"
                }

                return FileOutlineGenerationResponse(
                    success=True,
                    outline=landppt_outline,
                    file_info=file_info,
                    processing_stats=processing_stats,
                    message=f"成功使用summeryanyfile从文件 {request.filename} 生成PPT大纲，共{len(outline.slides)}页"
                )

            except ImportError as ie:
                logger.warning(f"summeryanyfile模块不可用: {ie}，使用简化版本")
                return await self._generate_outline_from_file_fallback(request)
            except Exception as se:
                logger.error(f"summeryanyfile生成失败: {se}，使用简化版本")
                return await self._generate_outline_from_file_fallback(request)

        except Exception as e:
            logger.error(f"从文件生成大纲失败: {e}")
            return FileOutlineGenerationResponse(
                success=False,
                error=str(e),
                message=f"从文件生成大纲失败: {str(e)}"
            )

    def _convert_summeryfile_outline_to_landppt(self, summery_outline, request) -> Dict[str, Any]:
        """将summeryanyfile的大纲格式转换为LandPPT格式"""
        try:
            slides = []

            for i, slide in enumerate(summery_outline.slides):
                # 转换幻灯片类型
                slide_type = "content"
                if slide.slide_type == "title":
                    slide_type = "title"
                elif slide.slide_type == "agenda":
                    slide_type = "agenda"
                elif slide.slide_type == "conclusion":
                    slide_type = "thankyou"

                # 构建内容点
                content_points = slide.content_points if hasattr(slide, 'content_points') else []
                if isinstance(content_points, list):
                    content = "\n".join([f"• {point}" for point in content_points])
                else:
                    content = str(content_points)

                landppt_slide = {
                    "id": i + 1,
                    "type": slide_type,
                    "title": slide.title,
                    "subtitle": getattr(slide, 'subtitle', ''),
                    "content": content,
                    "page_number": getattr(slide, 'page_number', i + 1),
                    "description": getattr(slide, 'description', ''),
                    "slide_type": slide_type,
                    "content_points": slide.content_points if hasattr(slide, 'content_points') else []
                }

                slides.append(landppt_slide)

            # 构建完整的大纲
            landppt_outline = {
                "title": summery_outline.title,
                "slides": slides,
                "metadata": {
                    "scenario": request.scenario,
                    "language": "zh",
                    "total_slides": len(slides),
                    "generated_with_summeryfile": True,
                    "file_source": request.filename,
                    "page_count_mode": summery_outline.page_count_mode,
                    "total_pages": summery_outline.total_pages,
                    "ppt_style": request.ppt_style,
                    "focus_content": request.focus_content,
                    "tech_highlights": request.tech_highlights,
                    "target_audience": request.target_audience
                }
            }

            return landppt_outline

        except Exception as e:
            logger.error(f"大纲格式转换失败: {e}")
            # 返回基本格式
            return {
                "title": request.topic or "文档演示",
                "slides": [
                    {
                        "id": 1,
                        "type": "title",
                        "title": request.topic or "文档演示",
                        "subtitle": "基于文档内容生成",
                        "content": ""
                    }
                ],
                "metadata": {
                    "scenario": request.scenario,
                    "language": "zh",
                    "total_slides": 1,
                    "generated_with_summeryfile": False,
                    "error": str(e)
                }
            }

    def _get_max_slides_from_request(self, request) -> int:
        """根据请求获取最大幻灯片数量"""
        if request.page_count_mode == "fixed":
            return request.fixed_pages or 20
        elif request.page_count_mode == "range":
            return request.max_pages or 20
        else:  # ai_decide
            return 25  # 默认最大值

    def _get_chunk_size_from_request(self, request) -> int:
        """根据请求获取分块大小"""
        if request.content_analysis_depth == "fast":
            return 1500  # 快速分块，适合研究报告的快速处理
        elif request.content_analysis_depth == "deep":
            return 4000
        else:  # standard
            return 3000

    def _get_chunk_strategy_from_request(self, request):
        """根据请求获取分块策略"""
        try:
            from summeryanyfile.core.models import ChunkStrategy

            if request.content_analysis_depth == "fast":
                return ChunkStrategy.FAST
            elif request.content_analysis_depth == "deep":
                return ChunkStrategy.HYBRID
            else:  # standard
                return ChunkStrategy.PARAGRAPH
        except ImportError:
            return "paragraph"  # 回退值

    async def _generate_outline_from_file_fallback(self, request):
        """当summeryanyfile不可用时的回退方法"""
        from ..api.models import FileOutlineGenerationResponse

        logger.info(f"使用简化版本从文件生成PPT大纲: {request.filename}")

        try:
            # 读取文件内容
            with open(request.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(request.file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except:
                with open(request.file_path, 'r', encoding='latin-1') as f:
                    content = f.read()

        # 创建基于文件内容的PPT大纲
        landppt_outline = self._create_outline_from_file_content(content, request)

        # 验证和修复fallback生成的大纲
        # 构建confirmed_requirements用于验证
        confirmed_requirements = {
            'topic': request.topic or landppt_outline.get('title', '文档演示'),
            'target_audience': getattr(request, 'target_audience', '通用受众'),
            'focus_content': getattr(request, 'focus_content', []),
            'tech_highlights': getattr(request, 'tech_highlights', []),
            'page_count_settings': {
                'mode': request.page_count_mode,
                'min_pages': getattr(request, 'min_pages', None),
                'max_pages': getattr(request, 'max_pages', None),
                'fixed_pages': getattr(request, 'fixed_pages', None)
            }
        }

        landppt_outline = await self._validate_and_repair_outline_json(landppt_outline, confirmed_requirements)

        # 获取文件信息
        file_info = {
            "filename": request.filename,
            "file_path": request.file_path,
            "processing_mode": request.file_processing_mode,
            "analysis_depth": request.content_analysis_depth,
            "used_summeryanyfile": False
        }

        # 获取处理统计信息
        slides_count = len(landppt_outline.get('slides', []))
        processing_stats = {
            "total_pages": slides_count,
            "page_count_mode": request.page_count_mode,
            "slides_count": slides_count,
            "processing_time": "完成",
            "generator": "fallback"
        }

        logger.info(f"简化版本大纲生成成功: {landppt_outline.get('title', '未知')}, 共{slides_count}页")

        return FileOutlineGenerationResponse(
            success=True,
            outline=landppt_outline,
            file_info=file_info,
            processing_stats=processing_stats,
            message=f"成功从文件 {request.filename} 生成PPT大纲（简化版本），共{slides_count}页"
        )

    def _create_outline_from_file_content(self, content: str, request) -> Dict[str, Any]:
        """从文件内容创建PPT大纲（简化版本）"""
        try:
            # 提取标题
            lines = content.strip().split('\n')
            title = request.topic or lines[0].strip() if lines else "文档演示"

            # 简单的内容分析
            sections = []
            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 检测标题（数字开头或特殊字符）
                if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                    line.startswith(('#', '##', '###')) or
                    len(line) < 50 and not line.endswith('。')):

                    if current_section:
                        sections.append(current_section)

                    current_section = {
                        "title": line.replace('#', '').replace('1.', '').replace('2.', '').replace('3.', '').strip(),
                        "content": []
                    }
                elif current_section:
                    current_section["content"].append(line)

            if current_section:
                sections.append(current_section)

            # 创建幻灯片
            slides = []

            # 标题页
            slides.append({
                "page_number": 1,
                "title": title,
                "content_points": ["基于文档内容生成", "演示者", "日期"],
                "slide_type": "title"
            })

            # 目录页
            if len(sections) > 1:
                agenda_points = [section['title'] for section in sections[:8]]
                slides.append({
                    "page_number": 2,
                    "title": "目录",
                    "content_points": agenda_points,
                    "slide_type": "agenda"
                })

            # 内容页
            for i, section in enumerate(sections[:10], start=len(slides) + 1):
                content_points = section["content"][:5] if section["content"] else ["内容要点1", "内容要点2"]
                slides.append({
                    "page_number": i,
                    "title": section["title"],
                    "content_points": content_points,
                    "slide_type": "content"
                })

            # 结束页
            slides.append({
                "page_number": len(slides) + 1,
                "title": "谢谢",
                "content_points": ["感谢聆听", "欢迎提问"],
                "slide_type": "thankyou"
            })

            # 根据页数设置调整
            if request.page_count_mode == "fixed" and request.fixed_pages:
                target_pages = request.fixed_pages
                if len(slides) > target_pages:
                    slides = slides[:target_pages]
                elif len(slides) < target_pages:
                    # 添加更多内容页
                    for i in range(len(slides), target_pages):
                        slides.append({
                            "page_number": i + 1,
                            "title": f"补充内容 {i - 1}",
                            "content_points": ["待补充的内容要点", "根据需要添加详细信息"],
                            "slide_type": "content"
                        })

            return {
                "title": title,
                "slides": slides,
                "metadata": {
                    "scenario": request.scenario,
                    "language": "zh",
                    "total_slides": len(slides),
                    "generated_with_file": True,
                    "file_source": request.filename,
                    "page_count_mode": request.page_count_mode,
                    "total_pages": len(slides),
                    "ppt_style": request.ppt_style,
                    "focus_content": request.focus_content,
                    "tech_highlights": request.tech_highlights,
                    "target_audience": request.target_audience
                }
            }

        except Exception as e:
            logger.error(f"从文件内容创建大纲失败: {e}")
            # 返回基本格式
            return {
                "title": request.topic or "文档演示",
                "slides": [
                    {
                        "page_number": 1,
                        "title": request.topic or "文档演示",
                        "content_points": ["基于文档内容生成", "演示者", "日期"],
                        "slide_type": "title"
                    }
                ],
                "metadata": {
                    "scenario": request.scenario,
                    "language": "zh",
                    "total_slides": 1,
                    "generated_with_file": False,
                    "error": str(e)
                }
            }

    async def _ensure_global_master_template_selected(self, project_id: str) -> Optional[Dict[str, Any]]:
        """确保项目已选择全局母版模板，如果没有则使用默认模板"""
        try:
            # 检查项目是否已有选择的模板（可以在项目元数据中存储）
            project = await self.project_manager.get_project(project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return None

            # 检查项目元数据中是否已有选择的模板ID
            selected_template_id = None
            if hasattr(project, 'project_metadata') and project.project_metadata:
                selected_template_id = project.project_metadata.get('selected_global_template_id')

            # 如果已有选择的模板，获取模板信息
            if selected_template_id:
                template = await self.global_template_service.get_template_by_id(selected_template_id)
                if template and template.get('is_active', True):
                    logger.info(f"Project {project_id} using selected template: {template['template_name']}")
                    return template

            # 如果没有选择或选择的模板不可用，使用默认模板
            default_template = await self.global_template_service.get_default_template()
            if default_template:
                # 将默认模板ID保存到项目元数据中
                await self._save_selected_template_to_project(project_id, default_template['id'])
                logger.info(f"Project {project_id} using default template: {default_template['template_name']}")
                return default_template

            logger.warning(f"No global master template available for project {project_id}")
            return None

        except Exception as e:
            logger.error(f"Error ensuring global master template for project {project_id}: {e}")
            return None

    async def _save_selected_template_to_project(self, project_id: str, template_id: int):
        """将选择的模板ID保存到项目元数据中"""
        try:
            project = await self.project_manager.get_project(project_id)
            if project:
                # 更新项目元数据
                project_metadata = project.project_metadata or {}
                project_metadata['selected_global_template_id'] = template_id

                # 保存更新的元数据
                await self.project_manager.update_project_metadata(project_id, project_metadata)
                logger.info(f"Saved selected template {template_id} to project {project_id}")

        except Exception as e:
            logger.error(f"Error saving selected template to project {project_id}: {e}")

    async def select_global_template_for_project(self, project_id: str, template_id: Optional[int] = None) -> Dict[str, Any]:
        """为项目选择全局母版模板"""
        try:
            if template_id:
                # 验证模板是否存在且可用
                template = await self.global_template_service.get_template_by_id(template_id)
                if not template:
                    raise ValueError(f"Template {template_id} not found")
                if not template.get('is_active', True):
                    raise ValueError(f"Template {template_id} is not active")
            else:
                # 使用默认模板
                template = await self.global_template_service.get_default_template()
                if not template:
                    raise ValueError("No default template available")
                template_id = template['id']

            # 保存选择到项目
            await self._save_selected_template_to_project(project_id, template_id)

            # 增加模板使用次数
            await self.global_template_service.increment_template_usage(template_id)

            return {
                "success": True,
                "message": "Template selected successfully",
                "selected_template": template
            }

        except Exception as e:
            logger.error(f"Error selecting global template for project {project_id}: {e}")
            return {
                "success": False,
                "message": str(e),
                "selected_template": None
            }

    async def get_selected_global_template(self, project_id: str) -> Optional[Dict[str, Any]]:
        """获取项目选择的全局母版模板"""
        try:
            project = await self.project_manager.get_project(project_id)
            if not project:
                return None

            # 从项目元数据中获取选择的模板ID
            selected_template_id = None
            if hasattr(project, 'project_metadata') and project.project_metadata:
                selected_template_id = project.project_metadata.get('selected_global_template_id')

            if selected_template_id:
                return await self.global_template_service.get_template_by_id(selected_template_id)

            return None

        except Exception as e:
            logger.error(f"Error getting selected global template for project {project_id}: {e}")
            return None

    def clear_cached_style_genes(self, project_id: Optional[str] = None):
        """清理缓存的设计基因"""
        if not hasattr(self, '_cached_style_genes'):
            return

        if project_id:
            # 清理特定项目的缓存
            if project_id in self._cached_style_genes:
                del self._cached_style_genes[project_id]
                logger.info(f"清理项目 {project_id} 的设计基因缓存")
        else:
            # 清理所有缓存
            self._cached_style_genes.clear()
            logger.info("清理所有设计基因缓存")

    def get_cached_style_genes_info(self) -> Dict[str, Any]:
        """获取缓存的设计基因信息"""
        if not hasattr(self, '_cached_style_genes'):
            return {"cached_projects": [], "total_count": 0}

        return {
            "cached_projects": list(self._cached_style_genes.keys()),
            "total_count": len(self._cached_style_genes)
        }
