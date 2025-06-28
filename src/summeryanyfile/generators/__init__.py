"""
生成器模块 - 包含PPT生成器和处理链
"""

from .ppt_generator import PPTOutlineGenerator
from .chains import ChainManager

__all__ = [
    "PPTOutlineGenerator",
    "ChainManager",
]
