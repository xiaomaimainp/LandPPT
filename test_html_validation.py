#!/usr/bin/env python3
"""
测试HTML验证逻辑的修改
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from landppt.services.enhanced_ppt_service import EnhancedPPTService

def test_html_validation():
    """测试HTML验证逻辑"""
    service = EnhancedPPTService()
    
    # 测试用例1: 只有missing_elements的HTML（应该被认为是完整的）
    html_with_missing_elements = """
    <div class="slide">
        <h1>Test Title</h1>
        <p>Test content</p>
    </div>
    """
    
    print("测试用例1: 只有missing_elements的HTML")
    result1 = service._validate_html_completeness(html_with_missing_elements)
    print(f"is_complete: {result1['is_complete']}")
    print(f"errors: {result1['errors']}")
    print(f"missing_elements: {result1['missing_elements']}")
    print(f"warnings: {result1['warnings']}")
    print()
    
    # 测试用例2: 有真正错误的HTML（应该被认为是不完整的）
    html_with_errors = """
    <div class="slide">
        <h1>Test Title
        <p>Test content</p>
    </div>
    """
    
    print("测试用例2: 有真正错误的HTML")
    result2 = service._validate_html_completeness(html_with_errors)
    print(f"is_complete: {result2['is_complete']}")
    print(f"errors: {result2['errors']}")
    print(f"missing_elements: {result2['missing_elements']}")
    print(f"warnings: {result2['warnings']}")
    print()
    
    # 测试用例3: 完整的HTML
    html_complete = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test</title>
    </head>
    <body>
        <div class="slide">
            <h1>Test Title</h1>
            <p>Test content</p>
        </div>
    </body>
    </html>
    """
    
    print("测试用例3: 完整的HTML")
    result3 = service._validate_html_completeness(html_complete)
    print(f"is_complete: {result3['is_complete']}")
    print(f"errors: {result3['errors']}")
    print(f"missing_elements: {result3['missing_elements']}")
    print(f"warnings: {result3['warnings']}")
    print()
    
    # 测试lxml解析器自动修复
    print("测试lxml解析器自动修复:")
    if result2['errors']:  # 只对有错误的HTML进行修复测试
        print("尝试修复有错误的HTML...")
        fixed_html = service._auto_fix_html_with_parser(html_with_errors)
        if fixed_html != html_with_errors:
            fixed_result = service._validate_html_completeness(fixed_html)
            print(f"修复后is_complete: {fixed_result['is_complete']}")
            print(f"修复后errors: {fixed_result['errors']}")
        else:
            print("lxml解析器未能修复HTML")

    print("\n✅ 测试完成 - AI修复功能已移除，只使用lxml解析器自动修复")

if __name__ == "__main__":
    test_html_validation()
