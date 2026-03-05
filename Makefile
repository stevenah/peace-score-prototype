.PHONY: dev dev-web dev-ml build build-web build-ml test test-web test-ml \
       lint typecheck check install install-web install-ml \
       db-push db-generate db-studio \
       docker-up docker-down docker-build \
       clean

# -------------------------------------------------------------------
# Development
# -------------------------------------------------------------------

dev: ## Run both frontend and ML backend (requires two terminals — use docker-up for single command)
	@echo "Use 'make dev-web' and 'make dev-ml' in separate terminals, or 'make docker-up'"

dev-web: ## Start Next.js dev server
	npm run dev

dev-ml: ## Start FastAPI dev server
	cd ml-backend && uv run uvicorn app.main:app --reload --port 8000

# -------------------------------------------------------------------
# Build
# -------------------------------------------------------------------

build: build-web ## Build all

build-web: ## Build Next.js for production
	npm run build

build-ml: ## Install ML backend in production mode
	cd ml-backend && uv sync --no-dev

# -------------------------------------------------------------------
# Test
# -------------------------------------------------------------------

test: test-web test-ml ## Run all tests

test-web: ## Run frontend tests (vitest)
	npm run test

test-web-watch: ## Run frontend tests in watch mode
	npm run test:watch

test-ml: ## Run ML backend tests (pytest)
	cd ml-backend && uv run pytest

# -------------------------------------------------------------------
# Lint & Typecheck
# -------------------------------------------------------------------

lint: ## Run ESLint
	npm run lint

typecheck: ## Run TypeScript type checking
	npx tsc --noEmit

check: lint typecheck test ## Run lint, typecheck, and tests

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------

install: install-web install-ml ## Install all dependencies

install-web: ## Install frontend dependencies
	npm install

install-ml: ## Install ML backend dependencies (with dev extras)
	cd ml-backend && uv sync --extra dev

# -------------------------------------------------------------------
# Database (Prisma)
# -------------------------------------------------------------------

db-push: ## Push Prisma schema to database
	npx prisma db push

db-generate: ## Generate Prisma client
	npx prisma generate

db-studio: ## Open Prisma Studio
	npx prisma studio

# -------------------------------------------------------------------
# Docker
# -------------------------------------------------------------------

docker-up: ## Start all services with docker compose
	docker compose up

docker-up-d: ## Start all services in background
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-build: ## Rebuild docker images
	docker compose build

# -------------------------------------------------------------------
# Clean
# -------------------------------------------------------------------

clean: ## Remove build artifacts
	rm -rf .next node_modules/.cache
	cd ml-backend && rm -rf __pycache__ .pytest_cache

# -------------------------------------------------------------------
# Help
# -------------------------------------------------------------------

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
