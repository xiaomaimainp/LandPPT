# LandPPT - AI-Powered PPT Generation Platform

### 🎯 Project Overview

LandPPT is an AI-powered presentation generation platform that automatically converts document content into professional PPT presentations. The platform supports multiple AI models and provides rich templates and style options, enabling users to quickly create high-quality presentations.

### ✨ Key Features

#### 🤖 AI-Driven Content Generation
- **Multi-AI Provider Support**: OpenAI, Anthropic Claude, Google Gemini, Ollama
- **Intelligent Content Analysis**: Automatically extract key information and structure from documents
- **Scenario-based Generation**: Support for business, education, technical, and other presentation scenarios
- **Automatic Outline Generation**: Intelligently generate PPT structure based on content

#### 📄 Powerful File Processing
- **Multi-format Support**: PDF, Word, Markdown, TXT, etc.
- **Intelligent Content Parsing**: High-quality document parsing using MarkItDown and MinerU
- **Batch Processing**: Support for multiple file uploads and processing
- **Content Optimization**: Automatically optimize document content for PPT format

#### 🎨 Rich Template System
- **Global Master Templates**: Unified design style and layout
- **Scenario-specific Templates**: Professional templates for different use cases
- **Custom Styling**: Personalized customization of colors, fonts, and layouts
- **Responsive Design**: Adapt to different screen sizes and devices

#### 📊 Project Management
- **Project Workflow**: Complete PPT creation and management process
- **Version Control**: Support for multi-version management and history
- **Collaboration Features**: Team collaboration and permission management
- **Progress Tracking**: Real-time tracking of project status and completion

#### 🔧 Developer Friendly
- **RESTful API**: Complete API interface documentation
- **OpenAI Compatible**: Support for OpenAI API format
- **Web Interface**: Intuitive user operation interface
- **Database Management**: Complete data persistence solution

### 🏗️ Technical Architecture

#### Backend Stack
- **Web Framework**: FastAPI - High-performance async web framework
- **Database**: SQLAlchemy + SQLite - Lightweight database solution
- **AI Integration**: Unified interface design for multiple providers
- **File Processing**: MarkItDown, MinerU, PyPDF, etc.
- **Authentication**: JWT + Session management

#### Frontend Stack
- **Template Engine**: Jinja2
- **Styling Framework**: CSS3 + Responsive design
- **Interactive Logic**: Native JavaScript
- **PDF Generation**: Puppeteer + PDF-lib

#### Core Modules
```
src/landppt/
├── ai/              # AI provider integration
├── api/             # API interface definitions
├── auth/            # Authentication and permission management
├── core/            # Core configuration
├── database/        # Database models and operations
├── services/        # Business logic services
└── web/             # Web interface
```

### 🚀 Quick Start

#### Requirements
- Python 3.11+
- Node.js 16.0+
- SQLite 3

#### Installation Steps

1. **Clone the project**
```bash
git clone <repository-url>
cd LandPPT
```

2. **Install Python dependencies**
```bash
pip install -e .
```

3. **Install Node.js dependencies**
```bash
npm install
```

4. **Initialize database**
```bash
python setup_database.py
```

5. **Start the service**
```bash
python run.py
```

#### Access the Application
- **Web Interface**: http://localhost:8000/web
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

### ⚙️ Configuration

#### AI Provider Configuration
Configure the corresponding AI provider API keys before use:

```python
# Configure through web interface or set environment variables directly
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

#### Database Configuration
Uses SQLite by default, with the database file located at `landppt.db` in the project root directory.

### 📖 Usage Guide

#### 1. Create PPT Project
- Access the web interface
- Select generation scenario (business, education, technical, etc.)
- Enter topic and requirements
- Upload related documents (optional)

#### 2. Generate and Edit
- System automatically generates PPT outline
- Confirm or modify outline structure
- Generate complete PPT content
- Preview and adjust styles

#### 3. Export and Share
- Export as PDF format
- Online preview and sharing
- Download source files

### 🔌 API Usage

#### Basic API Calls
```python
import requests

# Create PPT project
response = requests.post('http://localhost:8000/api/projects', json={
    "topic": "AI Development Trends",
    "scenario": "business",
    "requirements": "Include market analysis, technology trends, future outlook"
})

project = response.json()
```

#### OpenAI Compatible API
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Generate PPT outline about AI"}]
)
```

### 🛠️ Development Guide

#### Project Structure
```
LandPPT/
├── src/
│   ├── landppt/           # Main application
│   │   ├── ai/            # AI provider integration
│   │   ├── api/           # API routes and models
│   │   ├── auth/          # Authentication system
│   │   ├── core/          # Core configuration
│   │   ├── database/      # Database related
│   │   ├── services/      # Business logic
│   │   └── web/           # Web interface
│   └── summeryanyfile/    # Document processing module
├── static/                # Static resources
├── research_reports/      # Research report storage
├── package.json          # Node.js dependencies
├── pyproject.toml        # Python project configuration
├── run.py               # Application startup script
└── setup_database.py    # Database initialization
```

#### Adding New AI Providers
1. Create a new provider class in `src/landppt/ai/providers.py`
2. Inherit from `AIProvider` base class and implement required methods
3. Register the new provider in `AIProviderFactory`
4. Update configuration files to support the new provider

#### Custom Templates
1. Create new global master templates in the database
2. Define template HTML structure and CSS styles
3. Configure template metadata and parameters
4. Apply templates through API or web interface

### 🧪 Testing

#### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run specific tests
pytest tests/test_api.py
```

#### Test Coverage
```bash
pip install pytest-cov
pytest --cov=src/landppt tests/
```

### 📦 Deployment

#### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .
RUN npm install

EXPOSE 8000
CMD ["python", "run.py"]
```

#### Production Configuration
- Use PostgreSQL instead of SQLite
- Configure reverse proxy (Nginx)
- Set environment variables and key management
- Configure logging and monitoring

### 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - Python SQL toolkit
- [Puppeteer](https://pptr.dev/) - Headless Chrome control library
- [MarkItDown](https://github.com/microsoft/markitdown) - Document conversion tool

### 📞 Contact

- Project Homepage: [GitHub Repository]
- Issue Reports: [GitHub Issues]
- Email: [your-email@example.com]

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/LandPPT&type=Date)](https://star-history.com/#your-username/LandPPT&Date)

---

**Made with ❤️ by the LandPPT Team**
