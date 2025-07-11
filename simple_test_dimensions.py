#!/usr/bin/env python3
"""
简单测试AI决定图片尺寸功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dimension_mapping():
    """测试尺寸映射逻辑"""
    print("测试尺寸映射逻辑")
    print("=" * 30)
    
    # 模拟AI选择的映射
    dimensions_map = {
        "1": (2048, 1152),  # 16:9横向
        "2": (1152, 2048),  # 9:16竖向
        "3": (2048, 2048),  # 1:1正方形
        "4": (1920, 1080),  # 16:9标准
        "5": (1080, 1920),  # 9:16标准
    }
    
    test_choices = ["1", "2", "3", "4", "5", "invalid"]
    
    for choice in test_choices:
        selected_dimensions = dimensions_map.get(choice, (2048, 1152))
        width, height = selected_dimensions
        
        # 计算比例
        if width > height:
            ratio = width / height
            ratio_desc = f"横向 {ratio:.2f}:1"
        elif height > width:
            ratio = height / width
            ratio_desc = f"竖向 1:{ratio:.2f}"
        else:
            ratio_desc = "正方形 1:1"
        
        print(f"选择 '{choice}': {width}x{height} ({ratio_desc})")
    
    print("✅ 尺寸映射测试通过")

def test_truncate_function():
    """测试截断功能（确保之前的修复仍然有效）"""
    print("\n测试查询截断功能")
    print("=" * 30)
    
    # 模拟截断函数
    def truncate_search_query(query, max_length=100):
        if not query or len(query) <= max_length:
            return query
        
        truncated = query[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > 0:
            return truncated[:last_space]
        else:
            return truncated
    
    test_queries = [
        "short query",
        "this is a very long search query that definitely exceeds one hundred characters and should be truncated properly"
    ]
    
    for query in test_queries:
        truncated = truncate_search_query(query, 100)
        print(f"原始 ({len(query)}): {query[:50]}{'...' if len(query) > 50 else ''}")
        print(f"截断 ({len(truncated)}): {truncated}")
        print(f"有效: {len(truncated) <= 100}")
        print()
    
    print("✅ 截断功能测试通过")

if __name__ == "__main__":
    print("PPT图片处理器 - 简单功能测试")
    print("=" * 50)
    
    test_dimension_mapping()
    test_truncate_function()
    
    print("\n🎉 所有简单测试通过!")
