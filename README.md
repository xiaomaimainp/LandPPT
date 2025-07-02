# LandPPT - AI驱动的PPT生成平台

[![GitHub stars](https://img.shields.io/github/stars/sligter/LandPPT?style=flat-square)](https://github.com/sligter/LandPPT/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/sligter/LandPPT?style=flat-square)](https://github.com/sligter/LandPPT/network)
[![GitHub issues](https://img.shields.io/github/issues/sligter/LandPPT?style=flat-square)](https://github.com/sligter/LandPPT/issues)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg?style=flat-square)](https://fastapi.tiangolo.com)

[English](README_EN.md) | **中文**

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心功能](#-核心功能)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [配置说明](#-配置说明)
- [API文档](#-api文档)
- [技术栈](#️-技术栈)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

## 🎯 项目简介

LandPPT 是一个基于人工智能的演示文稿生成平台，能够自动将文档内容转换为专业的PPT演示文稿。平台支持多种AI模型，提供丰富的模板和样式选择，让用户能够创建高质量的演示文稿。


![image](https://img.pub/p/17ebc2a837fc02827d4a.png)

![image](https://img.pub/p/3a7dff4a4b9aebedc260.png)

![image](https://img.pub/p/56e2a5801875736f9fc5.png)

![image](https://img.pub/p/b6efaedcbff7c4f96d39.png)

![image](https://img.pub/p/61800c015f600210e8c4.png)


## ✨ 核心功能

### 🤖 多AI提供商支持
- **OpenAI GPT系列**：GPT-4.5/GPT-5 等主流模型
- **Anthropic Claude**：Claude-3 系列模型
- **Google Gemini**：Gemini-2.5 系列模型
- **Azure OpenAI**：企业级AI服务
- **Ollama**：本地部署的开源模型

### 📄 强大的文件处理能力
- **多格式支持**：PDF、Word、Markdown、TXT等
- **智能解析**：使用minueru和markitdown进行内容提取
- **深度研究**：集成Tavily API的DEEP研究功能

### 🎨 丰富的模板系统
- **全局主模板**：统一的HTML模板系统
- **多样化布局**：AI生成多种创意页面布局
- **自定义模板**：支持导入和创建个性化模板

### 📊 完整的项目管理
- **三阶段工作流**：需求确认 → 大纲生成 → PPT生成
- **可视化编辑**：大纲编辑器
- **实时预览**：16:9响应式页面预览

### 🌐 现代化Web界面
- **直观操作**：用户友好的Web界面
- **AI聊天编辑**：侧边栏AI编辑功能
- **多格式导出**：PDF/HTML/PPTX导出支持

## 🚀 快速开始

### 系统要求
- Python 3.11+
- SQLite 3
- Docker (可选)

### 本地安装

#### 方一：uv（推荐）

```bash
# 克隆项目
git clone https://github.com/sligter/LandPPT.git
cd LandPPT

# 安装uv（如果尚未安装）
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使用uv同步环境
uv sync

uv pip install apryse-sdk --extra-index-url=https://pypi.apryse.com
# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置你的AI API密钥

# 启动服务
uv run python run.py
```

### Docker部署

#### 使用预构建镜像（推荐）

```bash
# 拉取最新镜像
docker pull bradleylzh/landppt:latest

# 运行容器
docker run -d \
  --name landppt \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env \
  -v landppt_data:/app/data \
  -v landppt_reports:/app/research_reports \
  -v landppt_cache:/app/temp \
  bradleylzh/landppt:latest

# 查看日志
docker logs -f landppt
```

> **注意**: 确保在运行前创建并配置好 `.env` 文件，包含必要的API密钥。


## 📖 使用指南

### 1. 访问Web界面
启动服务后，访问以下地址：
- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

### 2. 配置AI提供商
在设置页面配置你的AI API密钥：
- OpenAI API Key
- Anthropic API Key
- Google API Key
- 或配置本地Ollama服务

### 3. 创建PPT项目
1. **需求确认**：输入主题、选择受众、设置页数范围
2. **大纲生成**：AI生成大纲
3. **PPT生成**：基于大纲生成完整的HTML演示文稿

### 4. 编辑和导出
- 使用AI聊天功能编辑HTML内容
- 导出为PDF、HTML或PPTX格式
- 保存项目版本和历史记录

## 🔧 配置说明

### 环境变量配置

主要配置项（详见 `.env.example`）：

```bash
# AI提供商配置
DEFAULT_AI_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# 服务器配置
HOST=0.0.0.0
PORT=8000
SECRET_KEY=your-secure-secret-key

# 功能配置
TAVILY_API_KEY=your_tavily_api_key_here  # 联网功能
APRYSE_LICENSE_KEY=your_apryse_key_here  # PPTX导出

# 生成参数
MAX_TOKENS=8192
TEMPERATURE=0.7
```

## 📚 API文档

启动服务后访问：
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🛠️ 技术栈

### 后端技术
- **FastAPI**: 现代化的Python Web框架
- **SQLAlchemy**: ORM数据库操作
- **Pydantic**: 数据验证和序列化
- **Uvicorn**: ASGI服务器

### AI集成
- **OpenAI**: GPT系列模型
- **Anthropic**: Claude系列模型
- **Google AI**: Gemini系列模型
- **LangChain**: AI应用开发框架

### 文件处理
- **mineru**: PDF智能解析
- **markitdown**: 多格式文档转换

### 导出功能
- **Pyppeteer**: HTML转PDF
- **Apryse SDK**: PPTX生成

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

详情请见 [贡献指南](CONTRIBUTING.md)。

### 报告问题
如果你发现了bug或有功能建议，请在 [Issues](https://github.com/sligter/LandPPT/issues) 页面创建新的issue。

## 📄 许可证

本项目采用 Apache License 2.0 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sligter/LandPPT&type=Date)](https://star-history.com/#sligter/LandPPT&Date)

## 📞 联系我们

- **项目主页**: https://github.com/sligter/LandPPT
- **问题反馈**: https://github.com/sligter/LandPPT/issues
- **讨论区**: https://github.com/sligter/LandPPT/discussions

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️ Star！**

Made with ❤️ by the LandPPT Team

</div>