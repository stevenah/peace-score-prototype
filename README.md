# PEACE Web Prototype

Next.js frontend + FastAPI ML backend for endoscopy video analysis.

## Local setup

Ports: **frontend 3001**, **ML backend 8001**, Postgres 5432, MinIO 9000/9001.
(3000 and 8000 are deliberately avoided — they collide with other projects.)

### Prerequisites

- Node 22.13+ (pnpm 11 will not run on older versions) and [pnpm](https://pnpm.io) (`corepack enable`)
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

Two Fly.io apps in `ams`, both stopped while idle: `peace-frontend` (`fly.toml`)
and `peace-ml` (`ml-backend/fly.toml`). Postgres is on Neon and videos are in
S3. The local setup above does not depend on any of it.

### How a deploy happens

Push to `main`. Two workflows each decide whether they need to run:

| Workflow | Runs when | Does |
|---|---|---|
| `deploy-web.yml` | anything outside `ml-backend/` changes, docs excepted | frontend tests, then deploys `peace-frontend` |
| `deploy-ml.yml` | `ml-backend/` changes, except `training/` and docs | backend tests, then deploys `peace-ml` |

`make deploy` runs both by hand and `make prod-status` shows the machines.

Database migrations and the admin seed run once per frontend deploy, as the
`release_command` in `fly.toml`, before the new version takes traffic.

Do not run `fly deploy` from a laptop. It uploads the local directory as the
build context, and a working copy can hold patient-derived files
(`ml-backend/runs/`, `.env`) that a CI checkout never has.

### Secrets

Set with `fly secrets set NAME=value -a <app>`.

| App | Secret | Purpose |
|---|---|---|
| `peace-frontend` | `DATABASE_URL` | Neon connection string (the direct one, not the pooled one) |
| | `NEXTAUTH_SECRET` | Signs sessions: `openssl rand -base64 32` |
| | `ML_API_KEY` | Shared key for the ML backend |
| | `SEED_ADMIN_PASSWORD` | Password for the admin account the seed creates; `SEED_ADMIN_EMAIL` changes its address |
| | `S3_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Video storage |
| `peace-ml` | `PEACE_API_KEY` | Same value as `ML_API_KEY` |
| | `PEACE_S3_BUCKET`, `PEACE_AWS_ACCESS_KEY_ID`, `PEACE_AWS_SECRET_ACCESS_KEY` | Video storage |

GitHub Actions needs one secret, `FLY_API_TOKEN`.

With `PEACE_API_KEY` set, the ML backend refuses analyze requests that lack it
and live sockets that lack a token from the web app. Locally both keys are
unset and nothing is checked.

### Giving someone access

Accounts are invite-only. Sign in as an admin, open `/admin`, choose **Add
user**, generate a password and tick **Require password change on first
login**. Send them the link and that password.

### Cost

While idle the bill is the `peace-ml` volume (10 GB, $1.50/month) plus storage
for the two stopped machines, well under $1. While awake the frontend costs
about $0.005/hour and the ML backend about $0.037/hour, and each stops itself
after a few idle minutes. The first request after a stop waits for a cold
start. Fly prices as of October 2026.

## Notes

- The model weights (`ml-backend/app/models/best_model.pt`, ~73 MB) are committed,
  so a fresh clone can run analysis with no extra download.
- The ML backend requires Python 3.11 or 3.12 — PyTorch has no 3.13+ wheels yet.
  `uv sync` picks a compatible interpreter automatically.
