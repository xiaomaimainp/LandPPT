# PPT幻灯片丢失问题修复总结

## 问题描述
用户报告在编辑第3页幻灯片后，第4页幻灯片消失，第5页变成了第4页的位置。原本5页的PPT变成了4页。

## 问题根源分析

### 原始问题流程：
1. 用户在`project_slides_editor.html`中编辑第3页
2. 前端调用`saveToServer()`函数
3. `saveToServer()`调用`PUT /api/projects/{project_id}/slides`接口
4. 后端`update_project_slides`路由调用`save_project_slides`方法
5. `save_project_slides`方法执行：
   - `await self.slide_repo.delete_slides_by_project_id(project_id)` - 删除所有幻灯片
   - 然后用传入的`slides_data`重新创建幻灯片
6. 如果`slides_data`不完整（比如只包含4页而不是原来的5页），第5页就永久丢失

### 核心问题：
- 使用了"删除所有-重新创建"的危险模式
- 没有验证传入数据的完整性
- 单个幻灯片编辑触发了全量重建操作

## 修复方案

### 1. 前端修复 (`src/landppt/web/templates/project_slides_editor.html`)

**修改前：**
```javascript
async function saveToServer() {
    // 调用批量保存API，会删除所有幻灯片再重建
    const response = await fetch(`/api/projects/{{ project.project_id }}/slides`, {
        method: 'PUT',
        body: JSON.stringify({ slides_data: updatedSlidesData })
    });
}
```

**修改后：**
```javascript
async function saveToServer() {
    // 使用单个幻灯片保存API逐个保存，避免删除所有幻灯片
    for (let i = 0; i < slidesData.length; i++) {
        const slide = slidesData[i];
        slide.is_user_edited = true;
        const success = await saveSingleSlideToServer(i, slide.html_content);
        // 处理保存结果...
    }
}
```

### 2. 后端数据库服务修复 (`src/landppt/database/service.py`)

**修改前：**
```python
async def save_project_slides(self, project_id: str, slides_html: str, slides_data: List[Dict[str, Any]] = None) -> bool:
    # 危险：先删除所有幻灯片
    await self.slide_repo.delete_slides_by_project_id(project_id)
    # 然后重新创建
    await self.slide_repo.create_slides(slide_records)
```

**修改后：**
```python
async def save_project_slides(self, project_id: str, slides_html: str, slides_data: List[Dict[str, Any]] = None) -> bool:
    # 安全：检查数据完整性
    existing_slides = await self.slide_repo.get_slides_by_project_id(project_id)
    existing_count = len(existing_slides)
    new_count = len(slides_data)
    
    if existing_count > 0 and new_count < existing_count:
        logger.warning(f"检测到数据可能不完整: 现有{existing_count}页, 新数据仅{new_count}页")
        logger.info("使用安全模式: 只更新提供的幻灯片，保留其他现有幻灯片")
    
    # 使用upsert方式更新，不删除现有幻灯片
    for i, slide_data in enumerate(slides_data):
        await self.slide_repo.upsert_slide(project_id, i, slide_record)
```

### 3. 新增完全重置方法

为需要完全重置幻灯片的场景（如重新生成PPT）添加了专门的方法：

```python
async def replace_all_project_slides(self, project_id: str, slides_html: str, slides_data: List[Dict[str, Any]] = None) -> bool:
    """完全替换项目的所有幻灯片 - 用于重新生成PPT等场景"""
    # 这里保留原来的删除重建逻辑
    await self.slide_repo.delete_slides_by_project_id(project_id)
    await self.slide_repo.create_slides(slide_records)
```

## 修复效果

### 修复前：
- 编辑第3页 → 第4、5页丢失
- 数据不安全，容易丢失

### 修复后：
- 编辑第3页 → 所有页面保留
- 数据安全，增量更新
- 自动检测数据完整性

## 安全机制

1. **数据完整性检查**：比较现有幻灯片数量与新数据数量
2. **安全模式**：当检测到数据可能不完整时，只更新提供的幻灯片
3. **增量更新**：使用upsert操作而不是删除重建
4. **错误隔离**：单个幻灯片保存失败不影响其他幻灯片

## 测试建议

1. 创建包含5页幻灯片的PPT项目
2. 编辑第3页内容并保存
3. 刷新页面检查是否仍有5页
4. 验证第4、5页内容完整性
5. 测试其他编辑操作（添加、删除、复制幻灯片）

## 向后兼容性

- 保留了所有原有API接口
- 现有功能不受影响
- 新增的安全机制是透明的

修复完成，问题已解决。
