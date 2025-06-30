# LandPPT - AI-Powered PPT Generation Platform

[![GitHub stars](https://img.shields.io/github/stars/your-username/LandPPT?style=flat-square)](https://github.com/your-username/LandPPT/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/your-username/LandPPT?style=flat-square)](https://github.com/your-username/LandPPT/network)
[![GitHub issues](https://img.shields.io/github/issues/your-username/LandPPT?style=flat-square)](https://github.com/your-username/LandPPT/issues)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg?style=flat-square)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?style=flat-square)](https://hub.docker.com)

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

LandPPT is an AI-powered presentation generation platform that automatically converts document content into professional PPT presentations. The platform supports multiple AI models and provides rich templates and style options, enabling users to quickly create high-quality presentations.

## ✨ Key Features

### 🤖 Multi-AI Provider Support
- **OpenAI GPT Series**: GPT-3.5/GPT-4 and other mainstream models
- **Anthropic Claude**: Claude-3 series models
- **Google Gemini**: Gemini-1.5 series models
- **Azure OpenAI**: Enterprise-grade AI services
- **Ollama**: Locally deployed open-source models

### 📄 Powerful File Processing
- **Multi-format Support**: PDF, Word, Markdown, TXT, etc.
- **Intelligent Parsing**: Content extraction using magic-pdf and markitdown
- **Local Caching**: File caching system with MD5 hash verification
- **Deep Research**: DEEP research functionality with Tavily API integration

### 🎨 Rich Template System
- **Global Master Templates**: Unified HTML template system
- **Diverse Layouts**: AI-generated creative page layouts
- **Scenario-based Design**: Professional templates for business, education, tech, etc.
- **Custom Templates**: Support for importing and creating personalized templates

### 📊 Complete Project Management
- **Three-stage Workflow**: Requirements confirmation → Outline generation → PPT generation
- **Visual Editing**: Mind map-style outline editor
- **Version Control**: Project version management and history tracking
- **Real-time Preview**: 16:9 responsive page preview

### 🔧 Developer Friendly
- **RESTful API**: Complete API interface
- **OpenAI Compatible**: Compatible with OpenAI API format
- **Real-time Configuration**: Configuration updates without restart
- **Health Monitoring**: Comprehensive service monitoring

### 🌐 Modern Web Interface
- **Intuitive Operation**: User-friendly web interface
- **AI Chat Editing**: Sidebar AI editing functionality
- **Multi-format Export**: PDF/HTML/PPTX export support
- **Responsive Design**: Adaptive to various devices

## 🚀 Quick Start

### System Requirements
- Python 3.11+
- SQLite 3
- Docker (optional)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LandPPT.git
cd LandPPT

# Install dependencies using uv (recommended)
pip install uv
uv pip install -e .

# Or use pip
pip install -e .

# Configure environment variables
cp .env.example .env
# Edit .env file and configure your AI API keys

# Initialize database
python setup_database.py

# Start the service
python run.py
```

### Docker Deployment

```bash
# Build image
docker build -t landppt .

# Run container
docker run -d \
  --name landppt \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env \
  -v landppt_data:/app/data \
  -v landppt_uploads:/app/uploads \
  landppt

# View logs
docker logs -f landppt
```

### Using Docker Compose (Recommended)

Create `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  landppt:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./.env:/app/.env
      - landppt_data:/app/data
      - landppt_uploads:/app/uploads
      - landppt_reports:/app/research_reports
    environment:
      - PYTHONPATH=/app/src
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "./docker-healthcheck.sh"]
      interval: 30s
      timeout: 30s
      retries: 3
      start_period: 40s

volumes:
  landppt_data:
  landppt_uploads:
  landppt_reports:
```

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

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

### Main API Endpoints

```bash
# Health Check
GET /health

# Project Management
POST /api/projects          # Create project
GET /api/projects           # Get project list
GET /api/projects/{id}      # Get project details

# File Processing
POST /api/upload            # Upload file
POST /api/files/upload-and-generate-outline  # Upload and generate outline

# OpenAI Compatible Interface
POST /v1/chat/completions   # Chat completion
POST /v1/completions        # Text completion
```

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
- **magic-pdf**: Intelligent PDF parsing
- **markitdown**: Multi-format document conversion
- **python-docx**: Word document processing
- **BeautifulSoup**: HTML parsing

### Export Functionality
- **Pyppeteer**: HTML to PDF conversion
- **Apryse SDK**: PPTX generation
- **Jinja2**: Template rendering

## 🤝 Contributing

We welcome all forms of contributions!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Environment Setup
```bash
# Clone your fork
git clone https://github.com/your-username/LandPPT.git
cd LandPPT

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src/
isort src/
```

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
