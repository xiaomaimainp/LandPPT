# LandPPT - AI驱动的PPT生成平台

### 🎯 项目简介

LandPPT 是一个基于人工智能的演示文稿生成平台，能够自动将文档内容转换为专业的PPT演示文稿。该平台支持多种AI模型，提供丰富的模板和样式选择，让用户能够快速创建高质量的演示文稿。

### ✨ 主要特性

#### 🤖 AI驱动的内容生成
- **多AI提供商支持**：OpenAI、Anthropic Claude、Google Gemini、Ollama
- **智能内容分析**：自动提取文档关键信息和结构
- **场景化生成**：支持商务、教育、技术等多种演示场景
- **自动大纲生成**：基于内容智能生成PPT结构

#### 📄 强大的文件处理能力
- **多格式支持**：PDF、Word、Markdown、TXT等
- **智能内容解析**：使用MarkItDown和MinerU进行高质量文档解析
- **批量处理**：支持多文件同时上传和处理
- **内容优化**：自动优化文档内容适配PPT格式

#### 🎨 丰富的模板系统
- **全局主模板**：统一的设计风格和布局
- **场景化模板**：针对不同使用场景的专业模板
- **自定义样式**：支持颜色、字体、布局的个性化定制
- **响应式设计**：适配不同屏幕尺寸和设备

#### 📊 项目管理功能
- **项目工作流**：完整的PPT创建和管理流程
- **版本控制**：支持多版本管理和历史记录
- **协作功能**：团队协作和权限管理
- **进度跟踪**：实时跟踪项目状态和完成度

#### 📤 多格式导出
- **PDF导出**：使用Pyppeteer生成高质量PDF文件
- **HTML导出**：支持独立HTML文件导出
- **PPTX导出**：基于Apryse SDK的PowerPoint格式导出
- **批量导出**：支持单页或整个项目的批量导出

#### 🔧 开发者友好
- **RESTful API**：完整的API接口文档
- **OpenAI兼容**：支持OpenAI API格式
- **Web界面**：直观的用户操作界面
- **数据库管理**：完整的数据持久化方案
- **跨平台支持**：Windows、Linux、macOS全平台兼容

### 🏗️ 技术架构

#### 后端技术栈
- **Web框架**：FastAPI - 高性能异步Web框架
- **数据库**：SQLAlchemy + SQLite - 轻量级数据库解决方案
- **AI集成**：多提供商统一接口设计
- **文件处理**：MarkItDown、MinerU、PyPDF等
- **认证系统**：JWT + Session管理

#### 前端技术栈
- **模板引擎**：Jinja2
- **样式框架**：CSS3 + 响应式设计
- **交互逻辑**：原生JavaScript
- **PDF生成**：Puppeteer + PDF-lib

#### 核心模块
```
src/landppt/
├── ai/              # AI提供商集成
├── api/             # API接口定义
├── auth/            # 认证和权限管理
├── core/            # 核心配置
├── database/        # 数据库模型和操作
├── services/        # 业务逻辑服务
└── web/             # Web界面
```

### 🚀 快速开始

#### 环境要求
- Python 3.11+
- Node.js 16.0+
- SQLite 3

#### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd LandPPT
```

2. **安装Python依赖**
```bash
pip install -e .
```

3. **安装Node.js依赖**
```bash
npm install
```

4. **初始化数据库**
```bash
python setup_database.py
```

5. **启动服务**
```bash
python run.py
```

#### 访问应用
- **Web界面**：http://localhost:8000/web
- **API文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/api/health

### ⚙️ 配置说明

#### AI提供商配置
在使用前需要配置相应的AI提供商API密钥：

```python
# 通过Web界面配置或直接设置环境变量
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

#### 数据库配置
默认使用SQLite，数据库文件位于项目根目录的`landppt.db`。

#### PPTX导出配置（可选）
要启用PPTX格式导出功能，需要安装和配置Apryse SDK：

```bash
# 安装Apryse SDK
pip install apryse-sdk

# 在.env文件中配置许可证密钥
# https://docs.apryse.com/core/guides/get-started/trial-key
APRYSE_LICENSE_KEY=your_apryse_license_key_here
```

### 📖 使用指南

#### 1. 创建PPT项目
- 访问Web界面
- 选择生成场景（商务、教育、技术等）
- 输入主题和要求
- 上传相关文档（可选）

#### 2. 生成和编辑
- 系统自动生成PPT大纲
- 确认或修改大纲结构
- 生成完整的PPT内容
- 预览和调整样式

#### 3. 导出和分享
- 导出为PDF格式（使用Pyppeteer）
- 导出为PPTX格式（需要Apryse SDK）
- 导出为HTML格式
- 在线预览和分享
- 下载源文件

### 🔌 API使用

#### 基础API调用
```python
import requests

# 创建PPT项目
response = requests.post('http://localhost:8000/api/projects', json={
    "topic": "人工智能发展趋势",
    "scenario": "business",
    "requirements": "包含市场分析、技术趋势、未来展望"
})

project = response.json()
```

#### OpenAI兼容API
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "生成关于AI的PPT大纲"}]
)
```

### 🛠️ 开发指南

#### 项目结构
```
LandPPT/
├── src/
│   ├── landppt/           # 主应用
│   │   ├── ai/            # AI提供商集成
│   │   ├── api/           # API路由和模型
│   │   ├── auth/          # 认证系统
│   │   ├── core/          # 核心配置
│   │   ├── database/      # 数据库相关
│   │   ├── services/      # 业务逻辑
│   │   └── web/           # Web界面
│   └── summeryanyfile/    # 文档处理模块
├── static/                # 静态资源
├── research_reports/      # 研究报告存储
├── package.json          # Node.js依赖
├── pyproject.toml        # Python项目配置
├── run.py               # 应用启动脚本
└── setup_database.py    # 数据库初始化
```

#### 添加新的AI提供商
1. 在`src/landppt/ai/providers.py`中创建新的提供商类
2. 继承`AIProvider`基类并实现必要方法
3. 在`AIProviderFactory`中注册新提供商
4. 更新配置文件支持新提供商

#### 自定义模板
1. 在数据库中创建新的全局主模板
2. 定义模板的HTML结构和CSS样式
3. 配置模板的元数据和参数
4. 通过API或Web界面应用模板

### 🧪 测试

#### 运行测试
```bash
# 安装测试依赖
pip install pytest pytest-asyncio

# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_api.py
```

#### 测试覆盖率
```bash
pip install pytest-cov
pytest --cov=src/landppt tests/
```

### 📦 部署

#### Docker部署
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .
RUN npm install

EXPOSE 8000
CMD ["python", "run.py"]
```

#### 生产环境配置
- 使用PostgreSQL替代SQLite
- 配置反向代理（Nginx）
- 设置环境变量和密钥管理
- 配置日志和监控

### 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

### 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的Python Web框架
- [SQLAlchemy](https://www.sqlalchemy.org/) - Python SQL工具包
- [Puppeteer](https://pptr.dev/) - 无头Chrome控制库
- [MarkItDown](https://github.com/microsoft/markitdown) - 文档转换工具

### 📞 联系方式

- 项目主页：[GitHub Repository]
- 问题反馈：[GitHub Issues]
- 邮箱：[your-email@example.com]

---

**Made with ❤️ by the LandPPT Team**
