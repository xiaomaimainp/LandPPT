# LandPPT - AI-Powered PPT Generation Platform

[English](README_EN.md) | [中文](README_CN.md)

---

## 🎯 Project Overview

LandPPT is an AI-powered presentation generation platform that automatically converts document content into professional PPT presentations. The platform supports multiple AI models and provides rich templates and style options, enabling users to quickly create high-quality presentations.

## ✨ Key Features

- 🤖 **Multi-AI Provider Support**: OpenAI, Anthropic Claude, Google Gemini, Ollama
- 📄 **Powerful File Processing**: PDF, Word, Markdown, TXT support with intelligent parsing
- 🎨 **Rich Template System**: Global master templates and scenario-specific designs
- 📊 **Project Management**: Complete workflow with version control and collaboration
- 🔧 **Developer Friendly**: RESTful API with OpenAI compatibility
- 🌐 **Web Interface**: Intuitive user operation interface

## 🚀 Quick Start

### Requirements
- Python 3.11+
- Node.js 16.0+
- SQLite 3

### Installation

```bash
# Clone the project
git clone <repository-url>
cd LandPPT

# Install Python dependencies
pip install -e .

# Install Node.js dependencies
npm install

# Initialize database
python setup_database.py

# Start the service
python run.py
```

### Access
- **Web Interface**: http://localhost:8000/web
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 🏗️ Technical Architecture

### Backend Stack
- **Web Framework**: FastAPI
- **Database**: SQLAlchemy + SQLite
- **AI Integration**: Multi-provider unified interface
- **File Processing**: MarkItDown, MinerU, PyPDF
- **Authentication**: JWT + Session management

### Frontend Stack
- **Template Engine**: Jinja2
- **Styling**: CSS3 + Responsive design
- **Interactive Logic**: Native JavaScript
- **PDF Generation**: Puppeteer + PDF-lib

## 📖 Documentation

For detailed documentation, please refer to:
- [中文文档](README_CN.md) - Complete Chinese documentation
- [English Documentation](README_EN.md) - Complete English documentation

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by the LandPPT Team**