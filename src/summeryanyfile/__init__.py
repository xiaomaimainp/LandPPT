"""
通用文本转PPT大纲生成器

基于LLM的智能文档分析和演示大纲生成工具
"""

__version__ = "0.1.0"
__author__ = "SummeryAnyFile Team"
__description__ = "通用文本转PPT大纲生成器 - 基于LLM的智能文档分析和演示大纲生成工具"

from .core.models import SlideInfo, PPTState
from .generators.ppt_generator import PPTOutlineGenerator
from .core.markitdown_converter import MarkItDownConverter
from .core.document_processor import DocumentProcessor

__all__ = [
    "SlideInfo",
    "PPTState",
    "PPTOutlineGenerator",
    "MarkItDownConverter",
    "DocumentProcessor",
    "__version__",
    "__author__",
    "__description__",
]
