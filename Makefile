# LandPPT Docker Management Makefile

.PHONY: help build up down logs restart clean dev prod test backup restore

# Default target
help:
	@echo "LandPPT Docker Management Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development environment"
	@echo "  make dev-logs     - View development logs"
	@echo "  make dev-down     - Stop development environment"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start production environment"
	@echo "  make prod-logs    - View production logs"
	@echo "  make prod-down    - Stop production environment"
	@echo ""
	@echo "Building:"
	@echo "  make build           - Build Docker images"
	@echo "  make build-try-all   - Try all build strategies (recommended)"
	@echo "  make build-python-only - Build with minimal Python (fastest)"
	@echo "  make build-alpine    - Build with Alpine Linux"
	@echo "  make build-ubuntu    - Build with Ubuntu"
	@echo "  make build-requirements - Build with requirements.txt"
	@echo "  make build-simple    - Build with simple Dockerfile"
	@echo "  make build-retry     - Build with retry mechanism"
	@echo ""
	@echo "General:"
	@echo "  make up           - Start services (production)"
	@echo "  make down         - Stop all services"
	@echo "  make logs         - View logs"
	@echo "  make restart      - Restart services"
	@echo "  make clean        - Clean up containers and images"
	@echo "  make test         - Run tests in container"
	@echo ""
	@echo "Data Management:"
	@echo "  make backup       - Backup application data"
	@echo "  make restore      - Restore application data"
	@echo ""
	@echo "Ollama:"
	@echo "  make ollama       - Start with Ollama support"
	@echo "  make ollama-down  - Stop Ollama services"

# Build Docker images
build:
	@echo "🔨 Building Docker images..."
	docker-compose build

# Build with different base images (for network issues)
build-alpine:
	@echo "🔨 Building with Alpine Linux base..."
	docker build -f Dockerfile.alpine -t landppt:alpine .

build-ubuntu:
	@echo "🔨 Building with Ubuntu base..."
	docker build -f Dockerfile.ubuntu -t landppt:ubuntu .

build-python-only:
	@echo "🔨 Building with Python-only (no system packages)..."
	docker build -f Dockerfile.python-only -t landppt:python-only .

build-requirements:
	@echo "🔨 Building with requirements.txt..."
	docker build -f Dockerfile.requirements -t landppt:requirements .

build-simple:
	@echo "🔨 Building Docker images with simple Dockerfile..."
	docker build -f Dockerfile.simple -t landppt:simple .

# Try different build strategies
build-try-all:
	@echo "🔨 Trying different build strategies..."
	@echo "1. Trying Python-only build (fastest)..."
	@if docker build -f Dockerfile.python-only -t landppt:latest .; then \
		echo "✅ Python-only build successful!"; \
	elif docker build -f Dockerfile.alpine -t landppt:latest .; then \
		echo "✅ Alpine build successful!"; \
	elif docker build -f Dockerfile.ubuntu -t landppt:latest .; then \
		echo "✅ Ubuntu build successful!"; \
	elif docker build -f Dockerfile.requirements -t landppt:latest .; then \
		echo "✅ Requirements build successful!"; \
	else \
		echo "❌ All build strategies failed"; \
		exit 1; \
	fi

# Build with retry mechanism
build-retry:
	@echo "🔨 Building Docker images with retry..."
	@for i in 1 2 3; do \
		echo "Attempt $$i/3..."; \
		if docker-compose build; then \
			echo "✅ Build successful on attempt $$i"; \
			break; \
		else \
			echo "❌ Build failed on attempt $$i"; \
			if [ $$i -eq 3 ]; then \
				echo "💡 Try 'make build-try-all' for network issues"; \
				exit 1; \
			fi; \
			sleep 5; \
		fi; \
	done

# Development environment
dev:
	@echo "🚀 Starting development environment..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file from template..."; \
		cp .env.docker .env; \
		echo "⚠️  Please edit .env file with your API keys before continuing"; \
		exit 1; \
	fi
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development environment started!"
	@echo "🌐 Access the application at: http://localhost:8000"

dev-logs:
	@echo "📋 Viewing development logs..."
	docker-compose -f docker-compose.dev.yml logs -f

dev-down:
	@echo "🛑 Stopping development environment..."
	docker-compose -f docker-compose.dev.yml down

# Production environment
prod:
	@echo "🚀 Starting production environment..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file from template..."; \
		cp .env.docker .env; \
		echo "⚠️  Please edit .env file with your API keys before continuing"; \
		exit 1; \
	fi
	docker-compose up -d
	@echo "✅ Production environment started!"
	@echo "🌐 Access the application at: http://localhost:8000"

prod-logs:
	@echo "📋 Viewing production logs..."
	docker-compose logs -f

prod-down:
	@echo "🛑 Stopping production environment..."
	docker-compose down

# General commands
up: prod

down:
	@echo "🛑 Stopping all services..."
	docker-compose down
	docker-compose -f docker-compose.dev.yml down

logs:
	@echo "📋 Viewing logs..."
	docker-compose logs -f

restart:
	@echo "🔄 Restarting services..."
	docker-compose restart

# Ollama support
ollama:
	@echo "🤖 Starting with Ollama support..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file from template..."; \
		cp .env.docker .env; \
		echo "⚠️  Please edit .env file with your API keys before continuing"; \
		exit 1; \
	fi
	docker-compose --profile ollama up -d
	@echo "✅ Services with Ollama started!"
	@echo "🌐 LandPPT: http://localhost:8000"
	@echo "🤖 Ollama: http://localhost:11434"

ollama-down:
	@echo "🛑 Stopping Ollama services..."
	docker-compose --profile ollama down

# Testing
test:
	@echo "🧪 Running tests..."
	docker-compose exec landppt python -m pytest tests/ -v

# Cleanup
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v
	docker-compose -f docker-compose.dev.yml down -v
	docker system prune -f
	@echo "✅ Cleanup completed!"

# Data management
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	docker run --rm \
		-v landppt_data:/data \
		-v $(PWD)/backups:/backup \
		alpine tar czf /backup/landppt-data-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "✅ Backup created in backups/ directory"

restore:
	@echo "📥 Restoring from backup..."
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "❌ Please specify BACKUP_FILE=path/to/backup.tar.gz"; \
		exit 1; \
	fi
	docker run --rm \
		-v landppt_data:/data \
		-v $(PWD)/backups:/backup \
		alpine tar xzf /backup/$(BACKUP_FILE) -C /data
	@echo "✅ Backup restored successfully"

# Health check
health:
	@echo "🏥 Checking service health..."
	@curl -f http://localhost:8000/health || echo "❌ Service is not healthy"

# Show status
status:
	@echo "📊 Service status:"
	docker-compose ps

# Update images
update:
	@echo "🔄 Updating Docker images..."
	docker-compose pull
	docker-compose up -d
	@echo "✅ Images updated and services restarted"

# Initialize environment
init:
	@echo "🎯 Initializing LandPPT environment..."
	@if [ ! -f .env ]; then \
		cp .env.docker .env; \
		echo "📝 Created .env file from template"; \
		echo "⚠️  Please edit .env file with your API keys"; \
	else \
		echo "✅ .env file already exists"; \
	fi
	@echo "🔨 Building images..."
	make build
	@echo "✅ Environment initialized! Run 'make dev' or 'make prod' to start"
