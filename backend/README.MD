# DataInsight API - Production-Grade Backend

AI-powered data analysis and insights platform built with FastAPI, SQLAlchemy, and modern Python technologies.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Features

### Core Features (MVP)
- **User Authentication** - JWT-based secure authentication with refresh tokens
- **Dataset Management** - Upload, manage, and process CSV, Excel, JSON, and Parquet files
- **Data Cleaning** - Automated data preprocessing with multiple strategies
- **Exploratory Data Analysis** - Automated EDA report generation with ydata-profiling
- **AI-Powered Insights** - OpenAI GPT-4 and Anthropic Claude integration for intelligent insights
- **Advanced Visualizations** - Interactive Plotly charts with automated recommendations

### Advanced Features (Production)
- **Background Processing** - Celery task queue for async operations
- **Caching** - Redis integration for performance optimization
- **Cloud Storage** - AWS S3 support for scalable file storage
- **Email Notifications** - SMTP integration for user communications
- **Payment Integration** - Razorpay for subscription management
- **Monitoring** - Sentry integration for error tracking

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

┌─────────────┐ ┌──────────────┐ ┌─────────────┐
│ Client │────▶│ FastAPI │────▶│ Database │
│ (React) │ │ Backend │ │ PostgreSQL │
└─────────────┘ └──────────────┘ └─────────────┘
│
├──────────┐
▼ ▼
┌─────────┐ ┌─────────┐
│ Redis │ │ Celery │
│ Cache │ │ Workers │
└─────────┘ └─────────┘
│
▼
┌─────────────┐
│ AI Services│
│ OpenAI/Claude│
└─────────────┘

text

## ✅ Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+** (or SQLite for development)
- **Redis 7+** (optional for caching and Celery)
- **Docker & Docker Compose** (optional for containerized deployment)

## 📦 Installation

### Local Development

1. **Clone the repository**
git clone https://github.com/yourusername/datainsight-api.git
cd datainsight-api

text

2. **Create virtual environment**
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

text

3. **Install dependencies**
pip install -r requirements.txt

text

4. **Set up environment variables**
cp .env.example .env

Edit .env with your configuration
text

5. **Initialize database**
python -c "from app.database import init_db; init_db()"

text

6. **Run the application**
python app/main.py

Or with uvicorn:
uvicorn app.main:app --reload --port 8000

text

### Docker Development

Build and run with Docker Compose
docker-compose up --build

Run in detached mode
docker-compose up -d

View logs
docker-compose logs -f web

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

Security
SECRET_KEY=your-secret-key-here # Generate: python -c "import secrets; print(secrets.token_urlsafe(32))"

Database
DATABASE_URL=postgresql://user:password@localhost:5432/datainsight

AI Services
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

Features
FEATURE_AI_INSIGHTS=True
FEATURE_AUTOML=False

See `.env.example` for all available options.

## 🚀 Running the Application

### Development Mode
With auto-reload
uvicorn app.main:app --reload --port 8000

Or use the main.py entry point
python app/main.py

### Production Mode
With Gunicorn and Uvicorn workers
gunicorn app.main:app
--workers 4
--worker-class uvicorn.workers.UvicornWorker
--bind 0.0.0.0:8000
--timeout 120

### Docker Production
docker-compose -f docker-compose.prod.yml up -d

## 📚 API Documentation

Once the application is running, access:

- **Interactive API Docs (Swagger UI)**: http://localhost:8000/api/v1/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/api/v1/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

### Quick API Examples

**Register User:**
curl -X POST "http://localhost:8000/api/v1/auth/register"
-H "Content-Type: application/json"
-d '{
"email": "user@example.com",
"username": "johndoe",
"full_name": "John Doe",
"password": "SecurePass123!"
}'

**Upload Dataset:**
curl -X POST "http://localhost:8000/api/v1/datasets/upload"
-H "Authorization: Bearer YOUR_TOKEN"
-F "file=@data.csv"
-F "name=Sales Data"

**Generate Insights:**
curl -X POST "http://localhost:8000/api/v1/insights/generate/1"
-H "Authorization: Bearer YOUR_TOKEN"

## 🧪 Testing

Install dev dependencies
pip install -r requirements-dev.txt

Run tests
pytest

Run with coverage
pytest --cov=app --cov-report=html

Run specific test file
pytest tests/test_auth.py -v

## 🚢 Deployment

### Docker Production Deployment

1. **Build production image**
docker build -f Dockerfile.prod -t datainsight-api:latest .

2. **Run with docker-compose**
docker-compose -f docker-compose.prod.yml up -d

### Manual Deployment

1. **Set up production environment**
export ENVIRONMENT=production
export DEBUG=False

Set all required environment variables

2. **Install dependencies**
pip install -r requirements.txt

3. **Run database migrations**
alembic upgrade head

4. **Start with Gunicorn**
gunicorn app.main:app
--workers 4
--worker-class uvicorn.workers.UvicornWorker
--bind 0.0.0.0:8000
--access-logfile -
--error-logfile -

### Cloud Deployment

- **AWS**: Use ECS/EKS with RDS for database
- **Azure**: Azure App Service with Azure Database for PostgreSQL
- **Google Cloud**: Cloud Run with Cloud SQL
- **DigitalOcean**: App Platform or Kubernetes

## 📁 Project Structure

backend/
├── app/
│ ├── api/ # API routes
│ │ └── v1/
│ │ ├── endpoints/ # API endpoints
│ │ └── api.py # Router aggregator
│ ├── core/ # Core configuration
│ │ ├── config.py # Settings
│ │ └── security.py # Security utilities
│ ├── models/ # SQLAlchemy models
│ │ ├── base.py
│ │ ├── user.py
│ │ └── dataset.py
│ ├── schemas/ # Pydantic schemas
│ │ ├── user.py
│ │ ├── dataset.py
│ │ └── token.py
│ ├── services/ # Business logic
│ │ ├── auth_service.py
│ │ ├── dataset_service.py
│ │ ├── cleaning_service.py
│ │ ├── eda_service.py
│ │ ├── ai_service.py
│ │ └── visualization_service.py
│ ├── utils/ # Utilities
│ │ ├── helpers.py
│ │ └── file_utils.py
│ ├── database.py # Database setup
│ └── main.py # Application entry
├── tests/ # Test suite
├── uploads/ # File uploads (gitignored)
├── .env # Environment variables
├── .gitignore
├── requirements.txt # Python dependencies
├── Dockerfile # Docker configuration
├── docker-compose.yml # Docker Compose
└── README.md # This file

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write tests for new features
- Update documentation
- Use type hints
- Add docstrings to functions

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [Full Docs](https://docs.example.com)
- **API Reference**: [API Docs](http://localhost:8000/api/v1/docs)
- **Issues**: [GitHub Issues](https://github.com/yourusername/datainsight-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/datainsight-api/discussions)

## 👥 Authors

- **Your Name** - *Initial work* - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- FastAPI framework by Sebastián Ramírez
- SQLAlchemy ORM
- Pydantic for data validation
- ydata-profiling for automated EDA
- Plotly for interactive visualizations
- OpenAI and Anthropic for AI capabilities

## 📊 Status

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Code Coverage](https://img.shields.io/badge/coverage-85%25-green)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

---

**Made with ❤️ using FastAPI**
