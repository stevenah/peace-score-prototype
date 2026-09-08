.PHONY: dev dev-web dev-ml build build-web build-ml test test-web test-ml \
       lint typecheck check install install-web install-ml \
       db-push db-generate db-studio \
       docker-up docker-down docker-build \
       db-up minio-up infra-up infra-down bootstrap \
       clean

# -------------------------------------------------------------------
# Development
# -------------------------------------------------------------------

dev: ## Run both frontend and ML backend (requires two terminals — use docker-up for single command)
	@echo "Use 'make dev-web' (:3001) and 'make dev-ml' (:8001) in separate terminals, or 'make docker-up'"

dev-web: ## Start Next.js dev server (http://localhost:3001)
	pnpm dev

dev-ml: ## Start FastAPI dev server (http://localhost:8001)
	cd ml-backend && uv run uvicorn app.main:app --reload --port 8001

# -------------------------------------------------------------------
# Build
# -------------------------------------------------------------------

build: build-web ## Build all

build-web: ## Build Next.js for production
	pnpm build

build-ml: ## Install ML backend in production mode
	cd ml-backend && uv sync --no-dev

# -------------------------------------------------------------------
# Test
# -------------------------------------------------------------------

test: test-web test-ml ## Run all tests

test-web: ## Run frontend tests (vitest)
	pnpm test

test-web-watch: ## Run frontend tests in watch mode
	pnpm test:watch

test-ml: ## Run ML backend tests (pytest)
	cd ml-backend && uv run pytest

# -------------------------------------------------------------------
# Lint & Typecheck
# -------------------------------------------------------------------

lint: ## Run ESLint
	pnpm lint

typecheck: ## Run TypeScript type checking
	pnpm exec tsc --noEmit

check: lint typecheck test ## Run lint, typecheck, and tests

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------

install: install-web install-ml ## Install all dependencies

install-web: ## Install frontend dependencies
	pnpm install

install-ml: ## Install ML backend dependencies (with dev extras)
	cd ml-backend && uv sync --extra dev

# -------------------------------------------------------------------
# Database (Prisma)
# -------------------------------------------------------------------

db-push: ## Push Prisma schema to database
	pnpm exec prisma db push

db-generate: ## Generate Prisma client
	pnpm exec prisma generate

db-studio: ## Open Prisma Studio
	pnpm exec prisma studio

# -------------------------------------------------------------------
# Local infrastructure (Postgres + MinIO only — app runs on the host)
# -------------------------------------------------------------------

db-up: ## Start only Postgres (localhost:5432)
	docker compose up -d postgres

minio-up: ## Start only MinIO S3 (API localhost:9000, console localhost:9001)
	docker compose up -d minio minio-init

infra-up: db-up minio-up ## Start Postgres + MinIO

infra-down: ## Stop local infrastructure
	docker compose stop postgres minio

# -------------------------------------------------------------------
# Bootstrap
# -------------------------------------------------------------------

bootstrap: ## Fresh clone -> runnable: deps, env, prisma client, migrations, seed
	@test -f .env || cp .env.example .env
	pnpm install
	cd ml-backend && uv sync --extra dev
	pnpm exec prisma generate
	pnpm exec prisma migrate deploy
	pnpm run db:seed
	@echo ""
	@echo "Ready. Run 'make dev-web' (:3001) and 'make dev-ml' (:8001) in separate terminals."

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
