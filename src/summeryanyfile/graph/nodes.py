"""
图节点实现 - 定义LangGraph工作流中的各个节点
"""

import json
from typing import Dict, Any, Literal
import logging
from langchain_core.runnables import RunnableConfig

from ..core.models import PPTState
from ..core.json_parser import JSONParser
from ..generators.chains import ChainManager, ChainExecutor
from ..utils.logger import LoggerMixin

logger = logging.getLogger(__name__)


class GraphNodes(LoggerMixin):
    """图节点集合，包含所有工作流节点的实现"""

    def __init__(self, chain_manager: ChainManager, config=None):
        self.chain_manager = chain_manager
        self.chain_executor = ChainExecutor(chain_manager)
        self.json_parser = JSONParser()
        self.config = config  # 添加配置参数
    
    async def analyze_structure(self, state: PPTState, config: RunnableConfig) -> Dict[str, Any]:
        """
        分析文档结构节点
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            更新的状态字段
        """
        self.logger.info("开始分析文档结构...")
        
        try:
            # 获取第一个文档块
            first_chunk = state["document_chunks"][0] if state["document_chunks"] else ""
            
            if not first_chunk.strip():
                self.logger.warning("第一个文档块为空，使用默认结构")
                structure = {
                    "title": "文档分析",
                    "type": "通用文档",
                    "sections": [],
                    "key_concepts": [],
                    "language": "中文",
                    "complexity": "中等"
                }
            else:
                # 调用结构分析链
                structure_response = await self.chain_executor.execute_with_retry(
                    "structure_analysis",
                    {"content": first_chunk},
                    config
                )
                
                # 解析JSON响应
                structure = self.json_parser.extract_json_from_response(structure_response)
                
                # 验证结构
                if not isinstance(structure, dict):
                    raise ValueError("结构分析返回的不是有效的字典")
            
            self.logger.info(f"文档结构分析完成: {structure.get('title', '未知标题')}")
            
            return {
                "document_structure": structure,
                "accumulated_context": first_chunk[:500]  # 保留前500字作为上下文
            }
            
        except Exception as e:
            self.logger.error(f"文档结构分析失败: {e}")
            # 返回默认结构
            return {
                "document_structure": {
                    "title": "文档分析",
                    "type": "通用文档",
                    "sections": [],
                    "key_concepts": [],
                    "language": "中文",
                    "complexity": "中等"
                },
                "accumulated_context": first_chunk[:500] if state["document_chunks"] else ""
            }
    
    async def generate_initial_outline(self, state: PPTState, config: RunnableConfig) -> Dict[str, Any]:
        """
        生成初始PPT框架节点
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            更新的状态字段
        """
        self.logger.info("开始生成初始PPT框架...")
        
        try:
            # 准备输入
            structure_json = json.dumps(state["document_structure"], ensure_ascii=False)
            first_chunk = state["document_chunks"][0] if state["document_chunks"] else ""
            
            # 准备输入参数，包含页数范围和目标语言
            chain_inputs = {
                "structure": structure_json,
                "content": first_chunk
            }

            # 添加页数范围信息
            if self.config:
                chain_inputs["slides_range"] = self.config.slides_range
                chain_inputs["target_language"] = self.config.target_language
            else:
                chain_inputs["slides_range"] = "10-25页"  # 默认范围
                chain_inputs["target_language"] = "zh"  # 默认中文

            # 调用初始大纲生成链
            outline_response = await self.chain_executor.execute_with_retry(
                "initial_outline",
                chain_inputs,
                config
            )
            
            # 解析JSON响应
            outline = self.json_parser.extract_json_from_response(outline_response)
            
            # 验证和修复大纲结构
            outline = self.json_parser.validate_ppt_structure(outline)
            
            self.logger.info(f"初始PPT框架生成完成: {outline.get('title', '未知标题')}")
            
            return {
                "ppt_title": outline.get("title", "学术演示"),
                "total_pages": outline.get("total_pages", 15),
                "page_count_mode": outline.get("page_count_mode", "estimated"),
                "slides": outline.get("slides", []),
                "current_index": 1
            }
            
        except Exception as e:
            self.logger.error(f"初始PPT框架生成失败: {e}")
            # 返回默认框架
            return {
                "ppt_title": "学术演示",
                "total_pages": 15,
                "page_count_mode": "estimated",
                "slides": [
                    {
                        "page_number": 1,
                        "title": "标题页",
                        "content_points": ["演示标题", "演示者", "日期"],
                        "slide_type": "title",
                        "description": "PPT开场标题页"
                    }
                ],
                "current_index": 1
            }
    
    async def refine_outline(self, state: PPTState, config: RunnableConfig) -> Dict[str, Any]:
        """
        细化PPT大纲节点
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            更新的状态字段
        """
        current_index = state["current_index"]
        total_chunks = len(state["document_chunks"])
        
        self.logger.info(f"正在细化PPT大纲 ({current_index + 1}/{total_chunks})...")
        
        # 检查是否还有内容需要处理
        if current_index >= total_chunks:
            self.logger.info("所有文档块已处理完成")
            return state
        
        try:
            # 获取当前文档块
            current_content = state["document_chunks"][current_index]
            
            # 准备现有大纲
            existing_outline = {
                "title": state["ppt_title"],
                "total_pages": state["total_pages"],
                "slides": state["slides"]
            }
            existing_outline_json = json.dumps(existing_outline, ensure_ascii=False)
            
            # 准备输入参数，包含页数范围和目标语言
            chain_inputs = {
                "existing_outline": existing_outline_json,
                "new_content": current_content,
                "context": state["accumulated_context"]
            }

            # 添加页数范围信息和目标语言
            if self.config:
                chain_inputs["slides_range"] = self.config.slides_range
                chain_inputs["target_language"] = self.config.target_language
            else:
                chain_inputs["slides_range"] = "10-25页"  # 默认范围
                chain_inputs["target_language"] = "zh"  # 默认中文

            # 调用细化链
            refined_response = await self.chain_executor.execute_with_retry(
                "refine_outline",
                chain_inputs,
                config
            )
            
            # 解析JSON响应
            refined_outline = self.json_parser.extract_json_from_response(refined_response)
            
            # 验证和修复结构
            refined_outline = self.json_parser.validate_ppt_structure(refined_outline)
            
            # 更新累积上下文
            new_context = state["accumulated_context"] + "\n" + current_content[:300]
            if len(new_context) > 2000:  # 限制上下文长度
                new_context = new_context[-2000:]
            
            self.logger.info(f"PPT大纲细化完成，当前页数: {len(refined_outline.get('slides', []))}")
            
            return {
                "ppt_title": refined_outline.get("title", state["ppt_title"]),
                "total_pages": refined_outline.get("total_pages", state["total_pages"]),
                "slides": refined_outline.get("slides", state["slides"]),
                "current_index": current_index + 1,
                "accumulated_context": new_context
            }
            
        except Exception as e:
            self.logger.error(f"PPT大纲细化失败: {e}")
            # 继续处理下一个块
            return {
                **state,
                "current_index": current_index + 1
            }
    
    async def finalize_outline(self, state: PPTState, config: RunnableConfig) -> Dict[str, Any]:
        """
        最终优化PPT大纲节点
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            更新的状态字段
        """
        self.logger.info("开始最终优化PPT大纲...")
        
        try:
            # 准备当前大纲
            current_outline = {
                "title": state["ppt_title"],
                "total_pages": state["total_pages"],
                "page_count_mode": state["page_count_mode"],
                "slides": state["slides"]
            }
            outline_json = json.dumps(current_outline, ensure_ascii=False)
            
            # 准备输入参数，包含页数范围和目标语言
            chain_inputs = {"outline": outline_json}

            # 添加页数范围信息和目标语言
            if self.config:
                chain_inputs["slides_range"] = self.config.slides_range
                chain_inputs["target_language"] = self.config.target_language
            else:
                chain_inputs["slides_range"] = "10-25页"  # 默认范围
                chain_inputs["target_language"] = "zh"  # 默认中文

            # 调用最终优化链
            final_response = await self.chain_executor.execute_with_retry(
                "finalize_outline",
                chain_inputs,
                config
            )
            
            # 解析JSON响应
            final_outline = self.json_parser.extract_json_from_response(final_response)
            
            # 验证和修复结构
            final_outline = self.json_parser.validate_ppt_structure(final_outline)
            
            # 确保幻灯片编号正确
            slides = final_outline.get("slides", [])
            for i, slide in enumerate(slides):
                slide["page_number"] = i + 1

            # 验证页数是否在范围内
            total_pages = len(slides)
            if self.config:
                if total_pages < self.config.min_slides:
                    self.logger.warning(f"生成的页数({total_pages})少于最小要求({self.config.min_slides})")
                elif total_pages > self.config.max_slides:
                    self.logger.warning(f"生成的页数({total_pages})超过最大限制({self.config.max_slides})")
                    # 如果超过最大页数，截取到最大页数
                    slides = slides[:self.config.max_slides]
                    total_pages = len(slides)
                    # 重新编号
                    for i, slide in enumerate(slides):
                        slide["page_number"] = i + 1

            self.logger.info(f"PPT大纲最终优化完成，总页数: {total_pages}")

            return {
                "ppt_title": final_outline.get("title", state["ppt_title"]),
                "total_pages": total_pages,
                "page_count_mode": "final",
                "slides": slides
            }
            
        except Exception as e:
            self.logger.error(f"PPT大纲最终优化失败: {e}")
            # 返回当前状态，但标记为最终状态
            slides = state["slides"]
            for i, slide in enumerate(slides):
                slide["page_number"] = i + 1
            
            return {
                "ppt_title": state["ppt_title"],
                "total_pages": len(slides),
                "page_count_mode": "final",
                "slides": slides
            }
    
    def should_continue_refining(self, state: PPTState) -> Literal["refine_outline", "finalize_outline"]:
        """
        判断是否继续细化的条件函数
        
        Args:
            state: 当前状态
            
        Returns:
            下一个节点名称
        """
        current_index = state["current_index"]
        total_chunks = len(state["document_chunks"])
        
        if current_index >= total_chunks:
            self.logger.info("所有文档块已处理，进入最终优化阶段")
            return "finalize_outline"
        else:
            self.logger.debug(f"继续处理文档块 {current_index + 1}/{total_chunks}")
            return "refine_outline"
