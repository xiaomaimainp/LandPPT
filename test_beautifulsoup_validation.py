#!/usr/bin/env python3
"""
测试新的HTML验证功能
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from landppt.services.enhanced_ppt_service import EnhancedPPTService

def test_html_validation():
    """测试HTML验证功能"""
    service = EnhancedPPTService()

    print("🧪 测试新的HTML验证功能")
    print("=" * 60)

    # 测试用例1：完整正确的HTML
    valid_html = """<!DOCTYPE html>
<html>
<head>
    <title>测试页面</title>
    <meta charset="UTF-8">
</head>
<body>
    <div class="container">
        <h1>标题</h1>
        <p>这是一个段落。</p>
    </div>
</body>
</html>"""

    print("\n1. 测试完整正确的HTML:")
    result1 = service._validate_html_completeness(valid_html)
    print(f"   是否完整: {result1['is_complete']}")
    print(f"   错误: {result1['errors']}")
    print(f"   警告: {result1['warnings']}")

    # 测试用例2：缺少DOCTYPE的HTML（应该只有警告）
    no_doctype_html = """<html>
<head>
    <title>测试页面</title>
</head>
<body>
    <div>内容</div>
</body>
</html>"""

    print("\n2. 测试缺少DOCTYPE的HTML（应该只有警告）:")
    result2 = service._validate_html_completeness(no_doctype_html)
    print(f"   是否完整: {result2['is_complete']}")
    print(f"   错误: {result2['errors']}")
    print(f"   警告: {result2['warnings']}")

    # 测试用例3：有未闭合关键标签的HTML（应该有错误）
    unclosed_tags_html = """<!DOCTYPE html>
<html>
<head>
    <title>测试页面</title>
</head>
<body>
    <div class="container">
        <h1>标题
        <p>这是一个段落。
    </div>
</body>
</html>"""

    print("\n3. 测试有未闭合关键标签的HTML（应该有错误）:")
    result3 = service._validate_html_completeness(unclosed_tags_html)
    print(f"   是否完整: {result3['is_complete']}")
    print(f"   错误: {result3['errors']}")
    print(f"   警告: {result3['warnings']}")

    # 测试用例4：结构顺序错误的HTML（应该只有警告）
    wrong_order_html = """<!DOCTYPE html>
<html>
<body>
    <div>内容在head之前</div>
</body>
<head>
    <title>测试页面</title>
</head>
</html>"""

    print("\n4. 测试结构顺序错误的HTML（应该只有警告）:")
    result4 = service._validate_html_completeness(wrong_order_html)
    print(f"   是否完整: {result4['is_complete']}")
    print(f"   错误: {result4['errors']}")
    print(f"   警告: {result4['warnings']}")

    # 测试用例5：包含自定义标签的HTML（应该被忽略，只检查HTML标签）
    custom_tags_html = """<!DOCTYPE html>
<html>
<head>
    <title>测试页面</title>
</head>
<body>
    <custom-component>
        <my-widget>自定义内容</my-widget>
    </custom-component>
    <div>正常内容</div>
</body>
</html>"""

    print("\n5. 测试包含自定义标签的HTML（应该被忽略）:")
    result5 = service._validate_html_completeness(custom_tags_html)
    print(f"   是否完整: {result5['is_complete']}")
    print(f"   错误: {result5['errors']}")
    print(f"   警告: {result5['warnings']}")

    # 测试用例6：空HTML（应该有错误）
    empty_html = ""

    print("\n6. 测试空HTML（应该有错误）:")
    result6 = service._validate_html_completeness(empty_html)
    print(f"   是否完整: {result6['is_complete']}")
    print(f"   错误: {result6['errors']}")
    print(f"   警告: {result6['warnings']}")

    # 测试用例7：格式错误的标签（应该有错误）
    malformed_html = """<!DOCTYPE html>
<html>
<head><title>测试</title></head>
<body>
    <div <p>>格式错误的标签</p></div>
</body>
</html>"""

    print("\n7. 测试格式错误的标签（应该有错误）:")
    result7 = service._validate_html_completeness(malformed_html)
    print(f"   是否完整: {result7['is_complete']}")
    print(f"   错误: {result7['errors']}")
    print(f"   警告: {result7['warnings']}")

    print("\n" + "=" * 60)
    print("✅ HTML验证测试完成")
    print("\n总结:")
    print("- 完整正确的HTML应该通过验证（无错误）")
    print("- 缺少DOCTYPE等结构问题应该只产生警告")
    print("- 未闭合的关键标签应该产生错误")
    print("- 格式错误的标签应该产生错误")
    print("- 自定义标签应该被忽略")

if __name__ == "__main__":
    test_html_validation()
