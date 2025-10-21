# AI Call Debit Spread Trader - Makefile

.PHONY: help install dev test lint format clean docker-build docker-run setup-db

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Set up development environment"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean up files"
	@echo "  docker-build- Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"
	@echo "  setup-db    - Initialize database"
	@echo "  run-api     - Start FastAPI backend"
	@echo "  run-dash    - Start Streamlit dashboard"
	@echo "  scan        - Run market scanner"
	@echo "  backtest    - Run strategy backtest"
	@echo "  train       - Train ML model"

# Development setup
install:
	pip install -r requirements.txt
	cd external/optlib && pip install -e .
	cd external/pyalgostrategypool && pip install -e .

dev: install
	cp .env.example .env
	@echo "Remember to update .env with your API keys!"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

# Database
setup-db:
	python -c "from src.core.database import create_tables; create_tables()"

# Application runners
run-api:
	python main.py api

run-dash:
	python main.py dashboard

scan:
	python main.py scan

backtest:
	python main.py backtest

train:
	python main.py train

# Docker
docker-build:
	docker build -t ai-trader:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Production deployment
deploy-staging:
	@echo "Deploying to staging..."
	# Add staging deployment commands

deploy-prod:
	@echo "Deploying to production..."
	# Add production deployment commands

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

# Development utilities
requirements:
	pip-compile requirements.in

requirements-dev:
	pip-compile requirements-dev.in

update-deps:
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in

# Monitoring
logs:
	tail -f logs/*.log

monitor:
	python -m http.server 8080 --directory logs/

# Data management
backup-db:
	pg_dump $(DATABASE_URL) > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql

restore-db:
	psql $(DATABASE_URL) < $(BACKUP_FILE)

# Performance testing
load-test:
	locust -f tests/load_test.py --host=http://localhost:8000

# Security
security-scan:
	bandit -r src/
	safety check

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/

docs-serve:
	cd docs/_build && python -m http.server 8080