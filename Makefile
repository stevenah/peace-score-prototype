.PHONY: dev dev-web dev-ml build build-web build-ml test test-web test-ml \
       lint typecheck check install install-web install-ml \
       db-push db-generate db-studio \
       docker-up docker-down docker-build \
       minio-up infra-up infra-down bootstrap \
       train-landmarks train-landmarks-results \
       deploy prod-status \
       clean

# -------------------------------------------------------------------
# Development
# -------------------------------------------------------------------

dev: ## Run the frontend and ML backend together in one terminal (mprocs; MinIO is a stopped tab)
	mprocs

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
# Landmark model training (GPU box; targets live in ml-backend/Makefile)
# -------------------------------------------------------------------

train-landmarks: ## Fine-tune all landmark candidates on all dev folds (resumable)
	$(MAKE) -C ml-backend train

train-landmarks-results: ## Held-out clip macro-F1 per candidate and fold
	$(MAKE) -C ml-backend train-results

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
# Local infrastructure (MinIO only — the database is a SQLite file)
# -------------------------------------------------------------------

minio-up: ## Start MinIO S3 (API localhost:9000, console localhost:9001)
	docker compose up -d minio minio-init

infra-up: minio-up ## Start local infrastructure

infra-down: ## Stop local infrastructure
	docker compose stop minio

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
	@echo "Ready. Run 'make dev' (frontend :3001 and ML backend :8001 in one terminal)."

# -------------------------------------------------------------------
# Docker
# -------------------------------------------------------------------

docker-up: ## Start MinIO, the ML backend and the frontend with docker compose
	docker compose up

docker-up-d: ## Start all services in background
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-build: ## Rebuild docker images
	docker compose build

# -------------------------------------------------------------------
# Deployment (Fly.io through GitHub Actions; see README)
# -------------------------------------------------------------------

deploy: ## Redeploy both apps from main (a push to main already does this for what changed)
	gh workflow run deploy-web.yml --ref main
	gh workflow run deploy-ml.yml --ref main

prod-status: ## Machine state of both Fly apps
	fly status -a peace-frontend
	fly status -a peace-ml

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
