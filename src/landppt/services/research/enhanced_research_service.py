"""
Enhanced Research Service with SearXNG and Deep Content Analysis

This service integrates multiple search providers (Tavily, SearXNG) with deep content
extraction and analysis using LangChain and BeautifulSoup for comprehensive research.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ...ai import get_ai_provider, AIMessage, MessageRole
from ...core.config import ai_config
from ..deep_research_service import DEEPResearchService, ResearchReport, ResearchStep
from .searxng_provider import SearXNGContentProvider, SearXNGSearchResponse
from .content_extractor import WebContentExtractor, ExtractedContent

logger = logging.getLogger(__name__)


@dataclass
class EnhancedResearchStep:
    """Enhanced research step with multiple data sources"""
    step_number: int
    query: str
    description: str
    tavily_results: Optional[List[Dict[str, Any]]] = None
    searxng_results: Optional[SearXNGSearchResponse] = None
    extracted_content: Optional[List[ExtractedContent]] = None
    analysis: str = ""
    completed: bool = False
    duration: float = 0.0


@dataclass
class EnhancedResearchReport:
    """Enhanced research report with comprehensive data"""
    topic: str
    language: str
    steps: List[EnhancedResearchStep]
    executive_summary: str
    key_findings: List[str]
    recommendations: List[str]
    sources: List[str]
    content_analysis: Dict[str, Any]
    created_at: datetime
    total_duration: float
    provider_stats: Dict[str, Any]


class EnhancedResearchService:
    """Enhanced research service with multiple providers and deep content analysis"""
    
    def __init__(self):
        self.deep_research_service = DEEPResearchService()
        self.searxng_provider = SearXNGContentProvider()
        self.content_extractor = WebContentExtractor()
        
        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        
    @property
    def ai_provider(self):
        """Get AI provider"""
        return get_ai_provider()
    
    def is_available(self) -> bool:
        """Check if enhanced research service is available"""
        # At least one search provider must be available
        tavily_available = self.deep_research_service.is_available()
        searxng_available = self.searxng_provider.is_available()
        ai_available = self.ai_provider is not None
        
        return ai_available and (tavily_available or searxng_available)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available search providers"""
        providers = []
        if self.deep_research_service.is_available():
            providers.append('tavily')
        if self.searxng_provider.is_available():
            providers.append('searxng')
        return providers
    
    async def conduct_enhanced_research(self, topic: str, language: str = "zh") -> EnhancedResearchReport:
        """
        Conduct comprehensive enhanced research with multiple providers
        
        Args:
            topic: Research topic
            language: Language for research and report
            
        Returns:
            EnhancedResearchReport with comprehensive findings
        """
        start_time = time.time()
        logger.info(f"Starting enhanced research for topic: {topic}")
        
        try:
            # Step 1: Generate research plan
            research_plan = await self._generate_research_plan(topic, language)
            
            # Step 2: Execute research steps with multiple providers
            research_steps = []
            provider_stats = {'tavily': 0, 'searxng': 0, 'content_extraction': 0}
            
            for i, step_plan in enumerate(research_plan, 1):
                step = await self._execute_enhanced_research_step(
                    i, step_plan, topic, language, provider_stats
                )
                research_steps.append(step)
                
                # Add delay between steps
                if i < len(research_plan):
                    await asyncio.sleep(1)
            
            # Step 3: Analyze all collected content
            content_analysis = await self._analyze_collected_content(research_steps, topic, language)
            
            # Step 4: Generate comprehensive report
            report = await self._generate_enhanced_report(
                topic, language, research_steps, content_analysis, 
                time.time() - start_time, provider_stats
            )
            
            logger.info(f"Enhanced research completed in {report.total_duration:.2f} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Enhanced research failed: {e}")
            raise
    
    async def _generate_research_plan(self, topic: str, language: str) -> List[Dict[str, str]]:
        """Generate research plan using AI"""
        prompt = f"""
作为专业研究员，请为以下主题制定详细的研究计划：

主题：{topic}
语言：{language}

请生成6个不同角度的研究步骤，每个步骤应该：
1. 有明确的研究目标
2. 包含具体的搜索查询
3. 覆盖不同的信息维度
4. 适合多种搜索引擎

请严格按照以下JSON格式返回：

```json
[
    {{
        "query": "具体的搜索查询词",
        "description": "这个步骤的研究目标和预期收获"
    }}
]
```

要求：
- 查询词要具体、专业，能获取高质量信息
- 覆盖基础概念、现状分析、趋势预测、案例研究、专家观点、技术细节等维度
- 适合{language}语言环境的搜索习惯
- 每个查询都应该能产生独特且有价值的信息
"""
        
        try:
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=min(ai_config.max_tokens, 2000),
                temperature=0.3
            )
            # Extract text content from AIResponse object
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            import json
            import re

            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(1))
                if isinstance(plan_data, list) and len(plan_data) > 0:
                    return plan_data

            # Fallback parsing
            try:
                plan_data = json.loads(response_text)
                if isinstance(plan_data, list):
                    return plan_data
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Failed to generate AI research plan: {e}")
        
        # Fallback to default plan
        return self._get_default_research_plan(topic, language)
    
    def _get_default_research_plan(self, topic: str, language: str) -> List[Dict[str, str]]:
        """Fallback research plan"""
        if language == "zh":
            return [
                {"query": f"{topic} 基础概念 定义 原理", "description": "了解主题的基本概念和核心原理"},
                {"query": f"{topic} 现状 发展趋势 2024 最新", "description": "分析当前发展现状和最新趋势"},
                {"query": f"{topic} 技术细节 实现方法 架构", "description": "深入了解技术实现和架构细节"},
                {"query": f"{topic} 案例研究 实践应用 成功案例", "description": "收集实际案例和应用实践"},
                {"query": f"{topic} 专家观点 研究报告 学术论文", "description": "获取专家观点和权威研究"},
                {"query": f"{topic} 未来发展 前景预测 挑战机遇", "description": "探索未来发展方向和挑战机遇"}
            ]
        else:
            return [
                {"query": f"{topic} fundamentals concepts definition", "description": "Understanding basic concepts and core principles"},
                {"query": f"{topic} current status trends 2024 latest", "description": "Analyzing current status and latest trends"},
                {"query": f"{topic} technical details implementation architecture", "description": "Deep dive into technical implementation and architecture"},
                {"query": f"{topic} case studies practical applications examples", "description": "Collecting real-world cases and applications"},
                {"query": f"{topic} expert opinions research reports papers", "description": "Gathering expert opinions and authoritative research"},
                {"query": f"{topic} future development predictions challenges opportunities", "description": "Exploring future directions and challenges"}
            ]
    
    async def _execute_enhanced_research_step(self, step_number: int, step_plan: Dict[str, str],
                                            topic: str, language: str, 
                                            provider_stats: Dict[str, int]) -> EnhancedResearchStep:
        """Execute a single enhanced research step with multiple providers"""
        step_start_time = time.time()
        logger.info(f"Executing enhanced research step {step_number}: {step_plan['query']}")
        
        step = EnhancedResearchStep(
            step_number=step_number,
            query=step_plan['query'],
            description=step_plan['description']
        )
        
        # Determine which providers to use based on configuration
        use_tavily = ai_config.research_provider in ['tavily', 'both'] and self.deep_research_service.is_available()
        use_searxng = ai_config.research_provider in ['searxng', 'both'] and self.searxng_provider.is_available()
        
        # Execute searches with available providers
        search_tasks = []
        
        if use_tavily:
            search_tasks.append(self._search_with_tavily(step_plan['query'], language))
        
        if use_searxng:
            search_tasks.append(self._search_with_searxng(step_plan['query'], language))
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process search results
        tavily_results = None
        searxng_results = None
        
        result_index = 0
        if use_tavily:
            if not isinstance(search_results[result_index], Exception):
                tavily_results = search_results[result_index]
                provider_stats['tavily'] += 1
            result_index += 1
        
        if use_searxng:
            if result_index < len(search_results) and not isinstance(search_results[result_index], Exception):
                searxng_results = search_results[result_index]
                provider_stats['searxng'] += 1
        
        step.tavily_results = tavily_results
        step.searxng_results = searxng_results
        
        # Extract content from URLs if enabled
        if ai_config.research_enable_content_extraction:
            urls = self._collect_urls_from_results(tavily_results, searxng_results)
            if urls:
                step.extracted_content = await self.content_extractor.extract_multiple(
                    urls[:10], max_concurrent=3  # Limit to top 10 URLs
                )
                provider_stats['content_extraction'] += len(step.extracted_content or [])
        
        # Analyze collected data
        step.analysis = await self._analyze_step_data(step, topic, language)
        step.completed = True
        step.duration = time.time() - step_start_time
        
        logger.info(f"Completed enhanced research step {step_number} in {step.duration:.2f}s")
        return step

    async def _search_with_tavily(self, query: str, language: str) -> Optional[List[Dict[str, Any]]]:
        """Search using Tavily provider"""
        try:
            return await self.deep_research_service._tavily_search(query, language)
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return None

    async def _search_with_searxng(self, query: str, language: str) -> Optional[SearXNGSearchResponse]:
        """Search using SearXNG provider"""
        try:
            return await self.searxng_provider.search(query, language)
        except Exception as e:
            logger.warning(f"SearXNG search failed: {e}")
            return None

    def _collect_urls_from_results(self, tavily_results: Optional[List[Dict[str, Any]]],
                                 searxng_results: Optional[SearXNGSearchResponse]) -> List[str]:
        """Collect URLs from all search results"""
        urls = []

        # Collect from Tavily results
        if tavily_results:
            for result in tavily_results:
                url = result.get('url')
                if url and url not in urls:
                    urls.append(url)

        # Collect from SearXNG results
        if searxng_results:
            for result in searxng_results.results:
                if result.url and result.url not in urls:
                    urls.append(result.url)

        return urls

    async def _analyze_step_data(self, step: EnhancedResearchStep, topic: str, language: str) -> str:
        """Analyze data collected in a research step"""

        # Prepare content for analysis
        content_parts = []

        # Add Tavily results
        if step.tavily_results:
            content_parts.append("=== Tavily搜索结果 ===")
            for i, result in enumerate(step.tavily_results[:5], 1):
                content_parts.append(f"{i}. {result.get('title', 'No title')}")
                content_parts.append(f"   URL: {result.get('url', 'No URL')}")
                content_parts.append(f"   内容: {result.get('content', 'No content')[:300]}...")
                content_parts.append("")

        # Add SearXNG results
        if step.searxng_results:
            content_parts.append("=== SearXNG搜索结果 ===")
            for i, result in enumerate(step.searxng_results.results[:5], 1):
                content_parts.append(f"{i}. {result.title}")
                content_parts.append(f"   URL: {result.url}")
                content_parts.append(f"   内容: {result.content[:300]}...")
                content_parts.append("")

        # Add extracted content
        if step.extracted_content:
            content_parts.append("=== 深度内容提取 ===")
            for i, content in enumerate(step.extracted_content[:3], 1):
                content_parts.append(f"{i}. {content.title}")
                content_parts.append(f"   URL: {content.url}")
                content_parts.append(f"   内容: {content.content[:500]}...")
                content_parts.append("")

        combined_content = "\n".join(content_parts)

        # Generate analysis
        analysis_prompt = f"""
作为专业研究分析师，请分析以下搜索结果并提供深入见解：

研究主题：{topic}
研究步骤：{step.description}
搜索查询：{step.query}

搜索结果：
{combined_content}

请提供：
1. 关键信息总结
2. 重要发现和洞察
3. 数据质量评估
4. 与研究主题的相关性分析

要求：
- 分析要深入、专业
- 突出最有价值的信息
- 指出信息的可靠性
- 语言使用{language}
"""

        try:
            analysis_response = await self.ai_provider.text_completion(
                prompt=analysis_prompt,
                max_tokens=min(ai_config.max_tokens, 1000),
                temperature=0.3
            )
            # Extract text content from AIResponse object
            analysis = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
            return analysis
        except Exception as e:
            logger.warning(f"Failed to generate step analysis: {e}")
            return f"分析步骤 {step.step_number}: {step.description}\n查询: {step.query}\n收集到相关信息，等待进一步分析。"

    async def _analyze_collected_content(self, steps: List[EnhancedResearchStep],
                                       topic: str, language: str) -> Dict[str, Any]:
        """Analyze all collected content for patterns and insights"""

        # Collect all content
        all_content = []
        total_sources = 0
        content_stats = {
            'tavily_results': 0,
            'searxng_results': 0,
            'extracted_pages': 0,
            'total_words': 0
        }

        for step in steps:
            if step.tavily_results:
                content_stats['tavily_results'] += len(step.tavily_results)
                total_sources += len(step.tavily_results)

            if step.searxng_results:
                content_stats['searxng_results'] += len(step.searxng_results.results)
                total_sources += len(step.searxng_results.results)

            if step.extracted_content:
                content_stats['extracted_pages'] += len(step.extracted_content)
                for content in step.extracted_content:
                    content_stats['total_words'] += content.word_count

            all_content.append(step.analysis)

        # Generate comprehensive analysis
        combined_analysis = "\n\n".join(all_content)

        analysis_prompt = f"""
作为高级研究分析师，请对以下研究主题进行综合分析：

研究主题：{topic}
语言：{language}

各步骤分析结果：
{combined_analysis}

请提供综合性分析，包括：
1. 跨步骤的关键模式和趋势
2. 信息的一致性和矛盾点
3. 研究的完整性评估
4. 重要的知识空白
5. 信息来源的多样性和可靠性

要求：
- 分析要全面、深入
- 识别重要的连接和关系
- 评估研究质量
- 使用{language}语言
"""

        try:
            comprehensive_analysis_response = await self.ai_provider.text_completion(
                prompt=analysis_prompt,
                max_tokens=min(ai_config.max_tokens, 1500),
                temperature=0.3
            )
            # Extract text content from AIResponse object
            comprehensive_analysis = comprehensive_analysis_response.content if hasattr(comprehensive_analysis_response, 'content') else str(comprehensive_analysis_response)

            return {
                'comprehensive_analysis': comprehensive_analysis,
                'content_stats': content_stats,
                'total_sources': total_sources,
                'analysis_quality': 'high' if total_sources >= 20 else 'medium' if total_sources >= 10 else 'basic'
            }
        except Exception as e:
            logger.warning(f"Failed to generate comprehensive analysis: {e}")
            return {
                'comprehensive_analysis': f"收集了来自{total_sources}个来源的信息，等待进一步分析。",
                'content_stats': content_stats,
                'total_sources': total_sources,
                'analysis_quality': 'basic'
            }

    async def _generate_enhanced_report(self, topic: str, language: str,
                                      steps: List[EnhancedResearchStep],
                                      content_analysis: Dict[str, Any],
                                      duration: float,
                                      provider_stats: Dict[str, Any]) -> EnhancedResearchReport:
        """Generate comprehensive enhanced research report"""

        # Collect all findings
        all_findings = []
        all_sources = []

        for step in steps:
            all_findings.append(step.analysis)

            # Collect sources
            if step.tavily_results:
                for result in step.tavily_results:
                    url = result.get('url')
                    if url and url not in all_sources:
                        all_sources.append(url)

            if step.searxng_results:
                for result in step.searxng_results.results:
                    if result.url and result.url not in all_sources:
                        all_sources.append(result.url)

        # Generate executive summary
        summary_prompt = f"""
基于以下研究发现，为主题"{topic}"生成执行摘要：

研究发现：
{chr(10).join(all_findings)}

综合分析：
{content_analysis.get('comprehensive_analysis', '')}

请生成一个简洁而全面的执行摘要，包括：
1. 研究主题的核心要点
2. 主要发现和结论
3. 关键趋势和模式
4. 研究的整体质量评估

要求：
- 摘要要专业、准确
- 突出最重要的信息
- 语言使用{language}
- 长度控制在300-500字
"""

        try:
            executive_summary_response = await self.ai_provider.text_completion(
                prompt=summary_prompt,
                max_tokens=min(ai_config.max_tokens, 800),
                temperature=0.3
            )
            # Extract text content from AIResponse object
            executive_summary = executive_summary_response.content if hasattr(executive_summary_response, 'content') else str(executive_summary_response)
        except Exception as e:
            logger.warning(f"Failed to generate executive summary: {e}")
            executive_summary = f"针对主题'{topic}'的综合研究报告，包含{len(steps)}个研究步骤的深入分析。"

        # Extract key findings
        key_findings = await self._extract_key_findings(topic, language, all_findings)

        # Generate recommendations
        recommendations = await self._generate_recommendations(topic, language, all_findings, content_analysis)

        return EnhancedResearchReport(
            topic=topic,
            language=language,
            steps=steps,
            executive_summary=executive_summary,
            key_findings=key_findings,
            recommendations=recommendations,
            sources=all_sources,
            content_analysis=content_analysis,
            created_at=datetime.now(),
            total_duration=duration,
            provider_stats=provider_stats
        )

    async def _extract_key_findings(self, topic: str, language: str,
                                  all_findings: List[str]) -> List[str]:
        """Extract key findings from research"""

        findings_text = "\n\n".join(all_findings)

        prompt = f"""
从以下研究发现中提取5-8个最重要的关键发现：

主题：{topic}
研究发现：
{findings_text}

请提取最重要、最有价值的关键发现，每个发现应该：
1. 简洁明了（1-2句话）
2. 具有实际价值
3. 基于可靠证据
4. 与主题高度相关

请以列表形式返回，每行一个发现，使用{language}语言。
"""

        try:
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=min(ai_config.max_tokens, 600),
                temperature=0.3
            )
            # Extract text content from AIResponse object
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse findings from response
            findings = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line[0].isdigit()):
                    # Clean up the line
                    clean_line = line.lstrip('-•*0123456789. ').strip()
                    if clean_line:
                        findings.append(clean_line)

            return findings[:8] if findings else [f"关于{topic}的重要研究发现需要进一步分析。"]

        except Exception as e:
            logger.warning(f"Failed to extract key findings: {e}")
            return [f"关于{topic}的重要研究发现需要进一步分析。"]

    async def _generate_recommendations(self, topic: str, language: str,
                                      all_findings: List[str],
                                      content_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""

        findings_text = "\n\n".join(all_findings)
        analysis_text = content_analysis.get('comprehensive_analysis', '')

        prompt = f"""
基于以下研究发现和分析，为主题"{topic}"生成5-7个可行的建议和推荐：

研究发现：
{findings_text}

综合分析：
{analysis_text}

请生成实用的建议，每个建议应该：
1. 具有可操作性
2. 基于研究证据
3. 针对实际需求
4. 具有前瞻性

请以列表形式返回，每行一个建议，使用{language}语言。
"""

        try:
            response = await self.ai_provider.text_completion(
                prompt=prompt,
                max_tokens=min(ai_config.max_tokens, 600),
                temperature=0.4
            )
            # Extract text content from AIResponse object
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse recommendations from response
            recommendations = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line[0].isdigit()):
                    # Clean up the line
                    clean_line = line.lstrip('-•*0123456789. ').strip()
                    if clean_line:
                        recommendations.append(clean_line)

            return recommendations[:7] if recommendations else [f"基于{topic}的研究，建议进行更深入的分析。"]

        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            return [f"基于{topic}的研究，建议进行更深入的分析。"]

    def get_status(self) -> Dict[str, Any]:
        """Get enhanced research service status"""
        return {
            'service': 'enhanced_research',
            'available': self.is_available(),
            'providers': {
                'tavily': self.deep_research_service.is_available(),
                'searxng': self.searxng_provider.is_available(),
                'content_extraction': ai_config.research_enable_content_extraction
            },
            'configuration': {
                'research_provider': ai_config.research_provider,
                'enable_content_extraction': ai_config.research_enable_content_extraction,
                'max_content_length': ai_config.research_max_content_length,
                'extraction_timeout': ai_config.research_extraction_timeout
            },
            'ai_provider': ai_config.default_ai_provider
        }
