# LandPPT uv 使用指南

## 什么是 uv？

uv 是一个极快的 Python 包管理器和项目管理工具，由 Astral 开发。它是用 Rust 编写的，比传统的 pip 快 10-100 倍，并提供了更好的依赖解析和项目管理功能。

## 为什么选择 uv？

### 🚀 性能优势
- **极快的安装速度**：比 pip 快 10-100 倍
- **并行下载**：同时下载多个包
- **智能缓存**：避免重复下载

### 🔒 可靠性
- **确定性构建**：uv.lock 文件确保所有环境一致
- **更好的依赖解析**：避免依赖冲突
- **版本锁定**：精确控制包版本

### 🛠️ 开发体验
- **自动虚拟环境管理**：无需手动创建和激活
- **项目感知**：自动检测项目配置
- **兼容性**：与 pip 和 PyPI 完全兼容

## LandPPT 中的 uv 配置

### 项目结构
```
LandPPT/
├── pyproject.toml      # 项目配置和依赖定义
├── uv.toml            # uv 特定配置
├── uv.lock            # 锁定的依赖版本
└── .venv/             # 虚拟环境（自动创建）
```

### 配置文件说明

#### pyproject.toml
- 定义项目元数据和依赖
- 包含开发依赖配置
- 设置项目脚本和入口点

#### uv.toml
- uv 特定配置
- 缓存目录设置
- 额外的包索引配置

#### uv.lock
- 锁定所有依赖的精确版本
- 确保跨环境的一致性
- 自动生成，不应手动编辑

## 常用命令

### 基本操作
```bash
# 同步项目依赖（推荐）
uv sync

# 添加新依赖
uv add fastapi

# 添加开发依赖
uv add --dev pytest

# 移除依赖
uv remove fastapi

# 更新依赖
uv sync --upgrade
```

### 运行命令
```bash
# 在项目环境中运行Python
uv run python script.py

# 运行项目脚本
uv run landppt

# 运行测试
uv run pytest

# 激活虚拟环境（可选）
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 环境管理
```bash
# 创建虚拟环境
uv venv

# 指定Python版本
uv venv --python 3.11

# 清理环境
rm -rf .venv
uv sync  # 重新创建
```

## 迁移指南

### 从 pip 迁移到 uv

1. **安装 uv**
   ```bash
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **同步现有项目**
   ```bash
   # 如果有 requirements.txt
   uv pip install -r requirements.txt
   
   # 如果有 pyproject.toml
   uv sync
   ```

3. **更新工作流**
   ```bash
   # 旧方式
   pip install package
   python script.py
   
   # 新方式
   uv add package
   uv run python script.py
   ```

## 最佳实践

### 1. 使用 uv sync 而不是 uv install
```bash
# 推荐：同步整个项目环境
uv sync

# 不推荐：只安装单个包
uv install package
```

### 2. 提交 uv.lock 文件
- 确保团队成员使用相同的依赖版本
- 提供可重现的构建

### 3. 使用 uv run 运行命令
```bash
# 推荐：自动使用项目环境
uv run python script.py

# 可选：手动激活环境
source .venv/bin/activate
python script.py
```

### 4. 分离生产和开发依赖
```toml
# pyproject.toml
[project]
dependencies = [
    "fastapi>=0.104.0",
    # ... 生产依赖
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    # ... 开发依赖
]
```

## 故障排除

### 常见问题

1. **uv 命令未找到**
   - 确保 uv 已正确安装
   - 重新启动终端或重新加载 PATH

2. **依赖冲突**
   ```bash
   # 清理并重新同步
   rm -rf .venv uv.lock
   uv sync
   ```

3. **缓存问题**
   ```bash
   # 清理缓存
   uv cache clean
   ```

4. **网络问题**
   ```bash
   # 使用镜像源
   uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

### 获取帮助
```bash
# 查看帮助
uv --help
uv sync --help

# 查看版本
uv --version

# 诊断信息
uv pip list
```

## 与其他工具的比较

| 特性 | uv | pip | poetry | pipenv |
|------|----|----|--------|--------|
| 安装速度 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 依赖解析 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 锁文件 | ✅ | ❌ | ✅ | ✅ |
| 虚拟环境 | ✅ | ❌ | ✅ | ✅ |
| 项目管理 | ✅ | ❌ | ✅ | ✅ |
| 学习曲线 | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## 总结

uv 为 LandPPT 项目提供了：
- 更快的依赖安装和管理
- 更可靠的环境一致性
- 更简单的开发工作流
- 更好的项目管理体验

通过使用 uv，开发者可以专注于代码开发，而不是环境管理问题。
