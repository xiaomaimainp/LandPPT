#!/usr/bin/env python3
"""
测试搜索查询截断功能的简单脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from landppt.services.ppt_image_processor import PPTImageProcessor

def test_truncate_search_query():
    """测试搜索查询截断功能"""
    processor = PPTImageProcessor()
    
    # 测试用例
    test_cases = [
        # (输入查询, 期望长度是否 <= 100)
        ("short query", True),
        ("business meeting presentation chart data analysis", True),
        ("this is a very long search query that definitely exceeds one hundred characters and should be truncated properly", False),
        ("artificial intelligence machine learning deep learning neural networks computer vision natural language processing", False),
        ("", True),  # 空字符串
        ("a" * 150, False),  # 150个字符，无空格
        ("word " * 30, False),  # 150个字符，有空格
    ]
    
    print("测试搜索查询截断功能:")
    print("=" * 60)
    
    for i, (query, should_be_short) in enumerate(test_cases, 1):
        truncated = processor._truncate_search_query(query, 100)
        is_valid = len(truncated) <= 100
        
        print(f"测试 {i}:")
        print(f"  原始查询 ({len(query)} 字符): {query[:50]}{'...' if len(query) > 50 else ''}")
        print(f"  截断查询 ({len(truncated)} 字符): {truncated}")
        print(f"  长度有效: {is_valid}")
        print(f"  测试通过: {is_valid}")
        print()
        
        if not is_valid:
            print(f"❌ 测试 {i} 失败: 截断后的查询仍然超过100字符")
            return False
    
    print("✅ 所有测试通过!")
    return True

def test_word_boundary_preservation():
    """测试单词边界保持功能"""
    processor = PPTImageProcessor()
    
    # 测试单词边界保持
    query = "artificial intelligence machine learning deep learning neural networks computer vision natural language"
    truncated = processor._truncate_search_query(query, 80)
    
    print("测试单词边界保持:")
    print("=" * 60)
    print(f"原始查询: {query}")
    print(f"截断查询: {truncated}")
    print(f"是否以完整单词结尾: {not truncated.endswith(' ') and ' ' in truncated}")
    
    # 检查是否没有在单词中间截断
    if truncated and not truncated.endswith(' '):
        last_word_in_truncated = truncated.split()[-1]
        words_in_original = query.split()
        
        # 检查最后一个单词是否完整
        word_is_complete = last_word_in_truncated in words_in_original
        print(f"最后一个单词完整: {word_is_complete}")
        
        if word_is_complete:
            print("✅ 单词边界保持测试通过!")
            return True
        else:
            print("❌ 单词边界保持测试失败!")
            return False
    
    print("✅ 单词边界保持测试通过!")
    return True

if __name__ == "__main__":
    print("PPT图片处理器 - 搜索查询截断功能测试")
    print("=" * 60)
    
    success1 = test_truncate_search_query()
    print()
    success2 = test_word_boundary_preservation()
    
    if success1 and success2:
        print("\n🎉 所有测试都通过了!")
        sys.exit(0)
    else:
        print("\n❌ 有测试失败!")
        sys.exit(1)
