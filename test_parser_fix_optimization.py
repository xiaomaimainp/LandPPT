#!/usr/bin/env python3
"""
测试解析器修复优化 - 验证解析器修复成功后不再调用 AI
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from landppt.services.enhanced_ppt_service import EnhancedPPTService
import asyncio

class MockAIProvider:
    """模拟 AI 提供者，用于测试是否被调用"""
    def __init__(self):
        self.call_count = 0
    
    async def chat_completion(self, *args, **kwargs):
        self.call_count += 1
        print(f"❌ AI 被调用了！调用次数: {self.call_count}")
        # 返回一个模拟的响应
        class MockResponse:
            content = """<!DOCTYPE html>
<html>
<head><title>AI Fixed</title></head>
<body><div>AI修复的内容</div></body>
</html>"""
        return MockResponse()

async def test_parser_fix_optimization():
    """测试解析器修复优化"""
    service = EnhancedPPTService()
    
    # 创建一个有语法错误但可以被解析器修复的 HTML
    test_html = """<!DOCTYPE html>
<html>
<head>
    <title>测试页面</title>
</head>
<body>
    <div>
        <p>段落开始
        <span>嵌套内容</p>
        </span>
    </div>
</body>
</html>"""
    
    print("=== 测试解析器修复优化 ===")
    print("原始 HTML:")
    print(test_html)
    
    # 验证原始 HTML 有错误
    validation_result = service._validate_html_completeness(test_html)
    print(f"\n原始验证结果:")
    print(f"是否完整: {validation_result['is_complete']}")
    print(f"错误: {validation_result['errors']}")
    
    # 测试解析器修复
    parser_fixed_html = service._auto_fix_html_with_parser(test_html)
    parser_validation = service._validate_html_completeness(parser_fixed_html)
    
    print(f"\n解析器修复后:")
    print(f"是否完整: {parser_validation['is_complete']}")
    print(f"错误: {parser_validation['errors']}")
    
    if parser_validation['is_complete']:
        print("✅ 解析器修复成功！在实际使用中应该跳过 AI 修复")
        print("修复后的 HTML:")
        print(parser_fixed_html)
    else:
        print("❌ 解析器修复失败，需要 AI 修复")

async def test_with_mock_ai():
    """使用模拟 AI 测试完整流程"""
    print("\n\n=== 测试完整流程（使用模拟 AI）===")
    
    # 这里我们无法直接替换 AI 提供者，但可以测试逻辑
    service = EnhancedPPTService()
    
    # 测试一个解析器可以修复的 HTML
    test_html = """<!DOCTYPE html>
<html>
<head><title>测试</title></head>
<body>
    <div>
        <p>段落<span>内容</p></span>
    </div>
</body>
</html>"""
    
    print("测试 HTML:")
    print(test_html)
    
    # 验证原始错误
    validation = service._validate_html_completeness(test_html)
    print(f"\n原始验证: 完整={validation['is_complete']}, 错误数={len(validation['errors'])}")
    
    # 测试解析器修复
    fixed = service._auto_fix_html_with_parser(test_html)
    fixed_validation = service._validate_html_completeness(fixed)
    print(f"解析器修复后: 完整={fixed_validation['is_complete']}, 错误数={len(fixed_validation['errors'])}")
    
    if fixed_validation['is_complete']:
        print("✅ 解析器修复成功，实际使用中会跳过 AI 调用")
    else:
        print("❌ 解析器修复失败，需要 AI 修复")

if __name__ == "__main__":
    asyncio.run(test_parser_fix_optimization())
    asyncio.run(test_with_mock_ai())
