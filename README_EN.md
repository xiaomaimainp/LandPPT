# LandPPT - AI-Powered PPT Generation Platform

[![GitHub stars](https://img.shields.io/github/stars/sligter/LandPPT?style=flat-square)](https://github.com/sligter/LandPPT/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/sligter/LandPPT?style=flat-square)](https://github.com/sligter/LandPPT/network)
[![GitHub issues](https://img.shields.io/github/issues/sligter/LandPPT?style=flat-square)](https://github.com/sligter/LandPPT/issues)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg?style=flat-square)](https://fastapi.tiangolo.com)

**English** | [中文](README.md)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Tech Stack](#️-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Project Overview

LandPPT is an AI-powered presentation generation platform that automatically converts document content into professional PPT presentations. The platform supports multiple AI models and provides rich templates and style options, enabling users to create high-quality presentations.


![image](https://img.pub/p/17ebc2a837fc02827d4a.png)

![image](https://img.pub/p/3a7dff4a4b9aebedc260.png)

![image](https://img.pub/p/56e2a5801875736f9fc5.png)

## ✨ Key Features

### 🤖 Multi-AI Provider Support
- **OpenAI GPT Series**: GPT-4.5/GPT-5 and other mainstream models
- **Anthropic Claude**: Claude-3 series models
- **Google Gemini**: Gemini-2.5 series models
- **Azure OpenAI**: Enterprise-grade AI services
- **Ollama**: Locally deployed open-source models

### 📄 Powerful File Processing
- **Multi-format Support**: PDF, Word, Markdown, TXT, etc.
- **Intelligent Parsing**: Content extraction using minueru and markitdown
- **Deep Research**: DEEP research functionality with Tavily API integration

### 🎨 Rich Template System
- **Global Master Templates**: Unified HTML template system
- **Diverse Layouts**: AI-generated creative page layouts
- **Custom Templates**: Support for importing and creating personalized templates

### 📊 Complete Project Management
- **Three-stage Workflow**: Requirements confirmation → Outline generation → PPT generation
- **Visual Editing**: Outline editor
- **Real-time Preview**: 16:9 responsive page preview

### 🌐 Modern Web Interface
- **Intuitive Operation**: User-friendly web interface
- **AI Chat Editing**: Sidebar AI editing functionality
- **Multi-format Export**: PDF/HTML/PPTX export support

## 🚀 Quick Start

### System Requirements
- Python 3.11+
- SQLite 3
- Docker (optional)

### Local Installation

#### Method 1: Manual Setup with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/sligter/LandPPT.git
cd LandPPT

# Install uv (if not already installed)
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync environment with uv
uv sync

uv pip install apryse-sdk --extra-index-url=https://pypi.apryse.com
# Configure environment variables
cp .env.example .env
# Edit .env file and configure your AI API keys

# Start the service
uv run python run.py
```

### Docker Deployment

#### Using Pre-built Image (Recommended)

```bash
# Pull the latest image
docker pull bradleylzh/landppt:latest

# Run container
docker run -d \
  --name landppt \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env \
  -v landppt_data:/app/data \
  -v landppt_reports:/app/research_reports \
  -v landppt_cache:/app/temp \
  bradleylzh/landppt:latest

# View logs
docker logs -f landppt
```

> **Note**: Make sure to create and configure the `.env` file with necessary API keys before running.



## 📖 Usage Guide

### 1. Access Web Interface
After starting the service, visit:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. Configure AI Providers
Configure your AI API keys in the settings page:
- OpenAI API Key
- Anthropic API Key
- Google API Key
- Or configure local Ollama service

### 3. Create PPT Projects
1. **Requirements Confirmation**: Input topic, select audience, set page range
2. **Outline Generation**: AI generates editable mind map outline
3. **PPT Generation**: Generate complete HTML presentation based on outline

### 4. Edit and Export
- Use AI chat functionality to edit content
- Export as PDF, HTML, or PPTX format
- Save project versions and history

## 🔧 Configuration

### Environment Variables

Main configuration items (see `.env.example` for details):

```bash
# AI Provider Configuration
DEFAULT_AI_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
SECRET_KEY=your-secure-secret-key

# Feature Configuration
TAVILY_API_KEY=your_tavily_api_key_here  # Research functionality
APRYSE_LICENSE_KEY=your_apryse_key_here  # PPTX export

# Generation Parameters
MAX_TOKENS=8192
TEMPERATURE=0.7
```

## 📚 API Documentation

After starting the service, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🛠️ Tech Stack

### Backend Technologies
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: ORM database operations
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server

### AI Integration
- **OpenAI**: GPT series models
- **Anthropic**: Claude series models
- **Google AI**: Gemini series models
- **LangChain**: AI application development framework

### File Processing
- **mineru**: Intelligent PDF parsing
- **markitdown**: Multi-format document conversion

### Export Functionality
- **Pyppeteer**: HTML to PDF conversion
- **Apryse SDK**: PPTX generation

## 🤝 Contributing

We welcome all forms of contributions!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For details, please see [Contributing Guide](CONTRIBUTING.md).

### Reporting Issues
If you find bugs or have feature suggestions, please create a new issue on the [Issues](https://github.com/sligter/LandPPT/issues) page.

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sligter/LandPPT&type=Date)](https://star-history.com/#sligter/LandPPT&Date)

## 📞 Contact Us

- **Project Homepage**: https://github.com/sligter/LandPPT
- **Issue Reporting**: https://github.com/sligter/LandPPT/issues
- **Discussions**: https://github.com/sligter/LandPPT/discussions

---

<div align="center">

**If this project helps you, please give us a ⭐️ Star!**

Made with ❤️ by the LandPPT Team

</div>
