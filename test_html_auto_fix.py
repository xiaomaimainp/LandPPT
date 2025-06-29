#!/usr/bin/env python3
"""
测试 HTML 自动修复功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from landppt.services.enhanced_ppt_service import EnhancedPPTService

def test_auto_fix_html():
    """测试 HTML 自动修复功能"""
    service = EnhancedPPTService()
    
    # 测试用例 1: 未闭合的标签
    test_html_1 = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>测试页面</title>
</head>
<body>
    <div class="container">
        <h1>标题</h1>
        <p>这是一个段落
        <div>未闭合的div
    </div>
</body>
</html>"""
    
    print("=== 测试用例 1: 未闭合的标签 ===")
    print("原始 HTML:")
    print(test_html_1)
    print("\n验证结果:")
    validation_result = service._validate_html_completeness(test_html_1)
    print(f"是否完整: {validation_result['is_complete']}")
    print(f"错误: {validation_result['errors']}")
    print(f"警告: {validation_result['warnings']}")
    
    print("\n自动修复后:")
    fixed_html = service._auto_fix_html_with_parser(test_html_1)
    print(fixed_html)
    
    print("\n修复后验证:")
    fixed_validation = service._validate_html_completeness(fixed_html)
    print(f"是否完整: {fixed_validation['is_complete']}")
    print(f"错误: {fixed_validation['errors']}")
    print(f"警告: {fixed_validation['warnings']}")
    
    # 测试用例 2: 嵌套错误的标签
    test_html_2 = """<!DOCTYPE html>
<html>
<head>
    <title>测试</title>
</head>
<body>
    <div>
        <p>段落开始
        <span>嵌套内容</p>
        </span>
    </div>
</body>
</html>"""
    
    print("\n\n=== 测试用例 2: 嵌套错误的标签 ===")
    print("原始 HTML:")
    print(test_html_2)
    print("\n验证结果:")
    validation_result_2 = service._validate_html_completeness(test_html_2)
    print(f"是否完整: {validation_result_2['is_complete']}")
    print(f"错误: {validation_result_2['errors']}")
    
    print("\n自动修复后:")
    fixed_html_2 = service._auto_fix_html_with_parser(test_html_2)
    print(fixed_html_2)
    
    print("\n修复后验证:")
    fixed_validation_2 = service._validate_html_completeness(fixed_html_2)
    print(f"是否完整: {fixed_validation_2['is_complete']}")
    print(f"错误: {fixed_validation_2['errors']}")
    
    # 测试用例 3: 正确的 HTML（应该不变）
    test_html_3 = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>正确的页面</title>
</head>
<body>
    <div class="container">
        <h1>标题</h1>
        <p>这是一个正确的段落</p>
        <div>正确闭合的div</div>
    </div>
</body>
</html>"""
    
    print("\n\n=== 测试用例 3: 正确的 HTML ===")
    print("原始 HTML:")
    print(test_html_3)
    print("\n验证结果:")
    validation_result_3 = service._validate_html_completeness(test_html_3)
    print(f"是否完整: {validation_result_3['is_complete']}")
    print(f"错误: {validation_result_3['errors']}")
    
    print("\n自动修复后:")
    fixed_html_3 = service._auto_fix_html_with_parser(test_html_3)
    print("HTML 是否改变:", fixed_html_3 != test_html_3)
    if fixed_html_3 != test_html_3:
        print("修复后的 HTML:")
        print(fixed_html_3)

def test_complex_errors():
    """测试更复杂的 HTML 错误"""
    service = EnhancedPPTService()

    # 测试用例 4: 多个错误
    test_html_4 = """<!DOCTYPE html>
<html>
<head>
    <title>复杂错误测试</title>
</head>
<body>
    <div class="main">
        <h1>标题
        <p>段落1</p>
        <div>
            <span>内容1
            <p>段落2
            <strong>粗体文本
        </div>
        <ul>
            <li>项目1
            <li>项目2</li>
        </ul>
    </div>
</body>
</html>"""

    print("\n\n=== 测试用例 4: 多个错误 ===")
    print("原始 HTML:")
    print(test_html_4)
    print("\n验证结果:")
    validation_result = service._validate_html_completeness(test_html_4)
    print(f"是否完整: {validation_result['is_complete']}")
    print(f"错误: {validation_result['errors']}")

    print("\n自动修复后:")
    fixed_html = service._auto_fix_html_with_parser(test_html_4)
    print(fixed_html)

    print("\n修复后验证:")
    fixed_validation = service._validate_html_completeness(fixed_html)
    print(f"是否完整: {fixed_validation['is_complete']}")
    print(f"错误: {fixed_validation['errors']}")

if __name__ == "__main__":
    test_auto_fix_html()
    test_complex_errors()
