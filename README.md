# PEACE Web Prototype

Next.js frontend + FastAPI ML backend for endoscopy video analysis.

## Local setup

Ports: **frontend 3001**, **ML backend 8001**, Postgres 5432, MinIO 9000/9001.
(3000 and 8000 are deliberately avoided — they collide with other projects.)

### Prerequisites

- Node 20+ and [pnpm](https://pnpm.io) (`corepack enable`)
- [uv](https://docs.astral.sh/uv/) for the Python backend
- Postgres reachable at `localhost:5432` — either a native install, or `make db-up`
  to run one in Docker on 5433 (adjust `DATABASE_URL` accordingly)

### One-time bootstrap

```bash
make bootstrap
```

That copies `.env.example` to `.env` if needed, installs both dependency trees,
generates the Prisma client, applies migrations, and seeds the database.

### Run it

Two terminals:

```bash
make dev-web   # http://localhost:3001
make dev-ml    # http://localhost:8001  (docs at /docs)
```

### Video storage

Uploads go straight from the browser to S3 via a presigned URL, so uploads and
playback need object storage. Two options:

- **Fully local (no AWS):** `make minio-up`, then uncomment the MinIO block at
  the bottom of `.env`. Both S3 clients honour `S3_ENDPOINT` /
  `PEACE_S3_ENDPOINT` and switch to path-style addressing automatically.
- **Real AWS:** leave `S3_ENDPOINT` unset and fill in the AWS keys.

Everything else — auth, analysis history, live frame analysis — works without
object storage configured.

### Everything in containers

```bash
make docker-up
```

Brings up Postgres, MinIO, the ML backend and the frontend, already wired to
each other. Frontend on 3001, ML backend on 8001.

## Common tasks

```bash
make check       # lint + typecheck + tests
make test        # frontend (vitest) + backend (pytest)
make db-studio   # Prisma Studio
make help        # all targets
```

## Deployment

`fly.toml` and `ml-backend/fly.toml` describe the Fly.io deployment
(`peace-frontend` and `peace-ml`). Both apps are currently scaled to zero. The
local setup above does not depend on them.

## Notes

- The model weights (`ml-backend/app/models/best_model.pt`, ~73 MB) are committed,
  so a fresh clone can run analysis with no extra download.
- The ML backend requires Python 3.11 or 3.12 — PyTorch has no 3.13+ wheels yet.
  `uv sync` picks a compatible interpreter automatically.
