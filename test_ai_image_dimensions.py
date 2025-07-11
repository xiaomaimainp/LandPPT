#!/usr/bin/env python3
"""
测试AI决定图片尺寸功能的脚本
"""

import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from landppt.services.ppt_image_processor import PPTImageProcessor
from landppt.models.image_models import ImageRequirement, ImagePurpose

class MockAIProvider:
    """模拟AI提供者，用于测试"""
    
    def __init__(self, mock_responses=None):
        self.mock_responses = mock_responses or {}
        self.call_count = 0
    
    async def text_completion(self, prompt, temperature=0.5):
        """模拟文本完成"""
        self.call_count += 1
        
        # 根据提示词内容返回不同的响应
        if "选择最佳的尺寸规格" in prompt:
            # 根据内容特点返回不同的选择
            if "人物" in prompt or "肖像" in prompt:
                return MockResponse("2")  # 竖向
            elif "风景" in prompt or "背景" in prompt:
                return MockResponse("1")  # 横向
            elif "图标" in prompt or "logo" in prompt:
                return MockResponse("3")  # 正方形
            else:
                return MockResponse("1")  # 默认横向
        
        return MockResponse("默认响应")

class MockResponse:
    def __init__(self, content):
        self.content = content

async def test_ai_image_dimensions():
    """测试AI决定图片尺寸功能"""
    print("测试AI决定图片尺寸功能")
    print("=" * 50)
    
    # 创建处理器并设置模拟AI提供者
    processor = PPTImageProcessor()
    processor.ai_provider = MockAIProvider()
    
    # 测试用例
    test_cases = [
        {
            "name": "风景背景图",
            "slide_title": "美丽的自然风光",
            "slide_content": "展示壮丽的山川河流风景",
            "project_topic": "旅游推广",
            "project_scenario": "旅游宣传",
            "expected_ratio": "16:9横向"
        },
        {
            "name": "人物肖像",
            "slide_title": "团队介绍",
            "slide_content": "展示团队成员的专业形象",
            "project_topic": "公司介绍",
            "project_scenario": "商务演示",
            "expected_ratio": "9:16竖向"
        },
        {
            "name": "产品图标",
            "slide_title": "产品特色",
            "slide_content": "展示产品logo和核心功能图标",
            "project_topic": "产品发布",
            "project_scenario": "产品展示",
            "expected_ratio": "1:1正方形"
        },
        {
            "name": "数据图表",
            "slide_title": "销售数据分析",
            "slide_content": "展示季度销售增长趋势",
            "project_topic": "业务报告",
            "project_scenario": "商务汇报",
            "expected_ratio": "16:9横向"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {case['name']}")
        print("-" * 30)
        
        try:
            # 调用AI决定尺寸的方法
            width, height = await processor._ai_decide_image_dimensions(
                case["slide_title"],
                case["slide_content"],
                case["project_topic"],
                case["project_scenario"]
            )
            
            # 判断比例类型
            if width > height:
                if width / height > 1.5:
                    ratio_type = "横向"
                else:
                    ratio_type = "略横向"
            elif height > width:
                if height / width > 1.5:
                    ratio_type = "竖向"
                else:
                    ratio_type = "略竖向"
            else:
                ratio_type = "正方形"
            
            print(f"  内容: {case['slide_title']}")
            print(f"  AI选择尺寸: {width}x{height}")
            print(f"  比例类型: {ratio_type}")
            print(f"  期望类型: {case['expected_ratio']}")
            
            # 验证尺寸是否合理
            valid_dimensions = [
                (2048, 1152), (1152, 2048), (2048, 2048),
                (1920, 1080), (1080, 1920)
            ]
            
            is_valid = (width, height) in valid_dimensions
            print(f"  尺寸有效: {is_valid}")
            
            if is_valid:
                print("  ✅ 测试通过")
            else:
                print("  ❌ 测试失败: 尺寸不在预期范围内")
                
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")

async def test_image_requirement_integration():
    """测试图片需求集成"""
    print("\n\n测试图片需求集成")
    print("=" * 50)
    
    processor = PPTImageProcessor()
    processor.ai_provider = MockAIProvider()
    
    # 创建不同用途的图片需求
    requirements = [
        ImageRequirement(
            purpose=ImagePurpose.BACKGROUND,
            description="用作幻灯片背景的风景图",
            count=1,
            priority=1
        ),
        ImageRequirement(
            purpose=ImagePurpose.ILLUSTRATION,
            description="说明概念的示意图",
            count=1,
            priority=2
        ),
        ImageRequirement(
            purpose=ImagePurpose.DECORATION,
            description="装饰性图标",
            count=1,
            priority=3
        )
    ]
    
    for i, req in enumerate(requirements, 1):
        print(f"\n需求 {i}: {req.purpose.value}")
        print("-" * 20)
        
        try:
            width, height = await processor._ai_decide_image_dimensions(
                "测试标题",
                "测试内容",
                "测试项目",
                "测试场景",
                req
            )
            
            print(f"  用途: {req.description}")
            print(f"  AI选择尺寸: {width}x{height}")
            print(f"  优先级: {req.priority}")
            print("  ✅ 集成测试通过")
            
        except Exception as e:
            print(f"  ❌ 集成测试失败: {e}")

if __name__ == "__main__":
    print("PPT图片处理器 - AI决定图片尺寸功能测试")
    print("=" * 60)
    
    try:
        # 运行异步测试
        asyncio.run(test_ai_image_dimensions())
        asyncio.run(test_image_requirement_integration())
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试运行失败: {e}")
        sys.exit(1)
