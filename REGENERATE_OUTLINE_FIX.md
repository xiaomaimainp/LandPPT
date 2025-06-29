# 重新生成大纲功能修复报告

## 问题描述

用户报告重新生成大纲功能（POST `/projects/{project_id}/regenerate-outline`）存在以下问题：
1. 生成大纲时遗漏了用户输入的具体要求（scenarios.html中的"具体要求"字段）
2. 遗漏了选择的目标受众信息
3. 遗漏了PPT风格和自定义风格提示等配置

## 根本原因分析

通过代码分析发现，问题出现在以下几个方面：

1. **PPTGenerationRequest模型不完整**：原始的 `PPTGenerationRequest` 模型缺少目标受众、PPT风格等关键字段
2. **参数传递不完整**：在 `regenerate_outline` 路由中创建 `PPTGenerationRequest` 时，没有传递目标受众等信息
3. **提示词生成逻辑缺陷**：`_create_outline_prompt` 方法没有正确处理自定义风格提示
4. **具体要求字段丢失**：在需求确认时，原始的 `requirements`（具体要求）字段没有被保存到 `confirmed_requirements` 中

## 修复内容

### 1. 扩展PPTGenerationRequest模型

**文件**: `src/landppt/api/models.py`

添加了以下字段：
```python
# 目标受众和风格相关参数
target_audience: Optional[str] = Field(None, description="Target audience for the PPT")
ppt_style: str = Field("general", description="PPT style: 'general', 'conference', 'custom'")
custom_style_prompt: Optional[str] = Field(None, description="Custom style prompt")
description: Optional[str] = Field(None, description="Additional description or requirements")
```

### 2. 修复regenerate_outline路由

**文件**: `src/landppt/web/routes.py`

修复了以下路由中的 `PPTGenerationRequest` 创建：

#### regenerate_outline路由 (行 860-870)
```python
project_request = PPTGenerationRequest(
    scenario=confirmed_requirements.get('scenario', 'general'),
    topic=confirmed_requirements.get('topic', project.topic),
    requirements=confirmed_requirements.get('requirements', project.requirements),
    language="zh",
    network_mode=confirmed_requirements.get('network_mode', False),
    target_audience=confirmed_requirements.get('target_audience', '普通大众'),
    ppt_style=confirmed_requirements.get('ppt_style', 'general'),
    custom_style_prompt=confirmed_requirements.get('custom_style_prompt'),
    description=confirmed_requirements.get('description')
)
```

#### 文件项目的FileOutlineGenerationRequest (行 875-887)
```python
file_request = FileOutlineGenerationRequest(
    # ... 其他参数 ...
    target_audience=confirmed_requirements.get('target_audience', '普通大众'),
    ppt_style=confirmed_requirements.get('ppt_style', 'general'),
    custom_style_prompt=confirmed_requirements.get('custom_style_prompt'),
    content_analysis_depth=confirmed_requirements.get('content_analysis_depth', 'standard')
)
```

#### 其他相关路由
- `start_workflow` 路由 (行 686-698)
- `generate_outline` 路由 (行 795-805)

### 3. 修复具体要求字段保存

**文件**: `src/landppt/web/routes.py`

#### 修复confirm_project_requirements路由 (行 1190-1239)

在需求确认时保留原始的具体要求字段：
```python
# Get project to access original requirements
project = await ppt_service.project_manager.get_project(project_id)
if not project:
    raise HTTPException(status_code=404, detail="Project not found")

# Update project with confirmed requirements
confirmed_requirements = {
    "topic": topic,
    "requirements": project.requirements,  # 保留原始的具体要求
    "target_audience": target_audience,
    # ... 其他字段
}
```

### 4. 改进提示词生成逻辑

**文件**: `src/landppt/services/enhanced_ppt_service.py`

#### 修复_create_outline_prompt方法 (行 549-576)

1. **正确获取参数**：
```python
target_audience = getattr(request, 'target_audience', None) or '普通大众'
ppt_style = getattr(request, 'ppt_style', None) or 'general'
custom_style_prompt = getattr(request, 'custom_style_prompt', None)
description = getattr(request, 'description', None)
```

2. **改进风格描述逻辑**：
```python
# 基础风格描述
style_descriptions = {
    "general": "通用商务风格，简洁专业",
    "conference": "学术会议风格，严谨正式",
    "custom": custom_style_prompt or "自定义风格"
}
style_desc = style_descriptions.get(ppt_style, "通用商务风格")

# 如果有自定义风格提示，追加到风格描述中
if custom_style_prompt and ppt_style != "custom":
    style_desc += f"，{custom_style_prompt}"
```

3. **完善提示词内容**：
```python
**项目信息：**
主题：{request.topic}
场景：{scenario_desc}
目标受众：{target_audience}
PPT风格：{style_desc}
特殊要求：{request.requirements or '无'}
补充说明：{description or '无'}{research_section}
```

## 测试验证

创建了完整的测试脚本验证修复效果：

1. ✅ 具体要求信息正确传递（包含用户在scenarios.html中输入的详细要求）
2. ✅ 目标受众信息正确传递
3. ✅ PPT风格信息正确传递
4. ✅ 自定义风格提示正确传递
5. ✅ 补充说明正确传递
6. ✅ 文件项目的参数传递正确

### 测试示例

用户输入的具体要求：
```
请重点关注以下内容：
1. 技术发展历程和里程碑事件
2. 当前主流技术栈和应用场景
3. 未来发展趋势和挑战
4. 实际案例分析和最佳实践
5. 对企业数字化转型的影响

风格要求：
- 内容要有深度，避免泛泛而谈
- 包含具体的数据和统计信息
- 适合技术管理层受众
```

在生成大纲的提示词中正确包含：
```
特殊要求：
    请重点关注以下内容：
    1. 技术发展历程和里程碑事件
    2. 当前主流技术栈和应用场景
    3. 未来发展趋势和挑战
    4. 实际案例分析和最佳实践
    5. 对企业数字化转型的影响

    风格要求：
    - 内容要有深度，避免泛泛而谈
    - 包含具体的数据和统计信息
    - 适合技术管理层受众
```

## 影响范围

修复涉及以下功能：
- 重新生成大纲功能
- 标准大纲生成功能
- 文件基础的大纲生成功能
- 项目工作流程中的大纲生成

## 向后兼容性

所有新增字段都是可选的，具有默认值，因此不会影响现有功能的正常运行。

## 总结

通过这次修复，重新生成大纲功能现在能够：
1. **完整保留用户的具体要求**：包含用户在项目创建时输入的详细要求和风格偏好
2. **正确使用目标受众信息**：确保内容适合指定的受众群体
3. **应用PPT风格设置**：使用用户选择的演示风格
4. **包含自定义风格提示**：体现用户的个性化风格要求
5. **考虑补充说明**：包含用户在需求确认时添加的额外信息

这确保了重新生成的大纲与用户的原始需求完全一致，提供更准确、更个性化的内容，真正体现用户的具体要求和期望。
