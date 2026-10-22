# PEACE Web Prototype

Next.js frontend + FastAPI ML backend for endoscopy video analysis.

## Local setup

Ports: **frontend 3001**, **ML backend 8001**, local S3 9000/9001.
(3000 and 8000 are deliberately avoided — they collide with other projects.)

The web app's database is a SQLite file (`dev.db` locally, a Fly volume in
production), so there is no database server to run.

### Prerequisites

- Node 22.13+ (pnpm 11 will not run on older versions) and [pnpm](https://pnpm.io) (`corepack enable`)
- [uv](https://docs.astral.sh/uv/) for the Python backend

### One-time bootstrap

```bash
make bootstrap
```

That copies `.env.example` to `.env` if needed, installs both dependency trees,
generates the Prisma client, creates `dev.db` with the migrations, and seeds
the admin account (`admin@peace.dev` / `admin123`, local only).

### Run it

```bash
make dev
```

That runs [mprocs](https://github.com/pvolok/mprocs) (`brew install mprocs`)
with the frontend on <http://localhost:3001> and the ML backend on
<http://localhost:8001> (API docs at `/docs`) as two tabs, plus a stopped
`s3` tab for when uploads are needed. Without mprocs, `make dev-web` and
`make dev-ml` are the same two commands in two terminals.

### Video storage

Uploads go straight from the browser to S3 via a presigned URL, so uploads and
playback need object storage. Two options:

- **Fully local (no AWS):** `make s3-up` (or start the `s3` tab in `make dev`)
  runs [RustFS](https://github.com/rustfs/rustfs) in Docker with a
  `peace-videos` bucket, then uncomment the local block at the bottom of
  `.env` and set the same four values with the `PEACE_` prefix in
  `ml-backend/.env`. Both S3 clients honour `S3_ENDPOINT` /
  `PEACE_S3_ENDPOINT` and switch to path-style addressing automatically.
  RustFS replaced MinIO here in October 2026, after MinIO withdrew its
  images from Docker Hub and quay.io; it speaks the same API on the same
  ports.
- **Real AWS:** leave `S3_ENDPOINT` unset and fill in the AWS keys.

Everything else — auth, analysis history, live frame analysis — works without
object storage configured.

### Everything in containers

```bash
make docker-up
```

Brings up the local S3, the ML backend and the frontend, already wired to
each other. Frontend on 3001, ML backend on 8001. The frontend's SQLite file lives
in `.docker-data/web/`.

## Common tasks

```bash
make check       # lint + typecheck + tests
make test        # frontend (vitest) + backend (pytest)
make db-studio   # Prisma Studio
make help        # all targets
```

## Landmark models and evals

`/admin` → **Models** lists every landmark bundle the ML backend can find
(`ml-backend/runs/**/bundle/`, `ml-backend/app/models/landmarks/` and the S3
bundle cache), says which one is served, and lets an admin:

- switch station recognition between **Off**, **Shadow** (computed and logged,
  nothing shown) and **On screen**;
- make another bundle active. It is verified and warmed up first, and a bundle
  that fails keeps the current one serving. Live sessions already open keep the
  model they started with;
- run evals, one at a time, each as its own process:

| Eval | Runs | Needs |
|---|---|---|
| Self-test | file hashes, reference logits, synthetic frames through the live path | nothing |
| Latency | per-frame latency on the CPU, with and without the bundle (`bench.py`) | nothing |
| Held-out accuracy | clip and frame metrics on the dev fold a fold model never saw, or the shift set (`bundle_eval.py`) | the private frame cache and the `train` extra |
| Out-of-fold metrics, gates report, procedure replay | `evaluate.py` and `stream_eval.py` on a training run | the run's saved predictions and `calibration.json` |

The locked test set cannot be scored from the page; that stays
`evaluate.py --test`, once.

The choice is saved in `ml-backend/app/models/landmarks/ACTIVE.json` and eval
results in `ml-backend/runs/ui_evals/`, both gitignored. Production has this
switched off (`PEACE_MODEL_ADMIN=false` in `ml-backend/fly.toml`): the tab
shows what is served there and changes nothing.

## Deployment

Two Fly.io apps in `ams`, both stopped while idle: `peace-frontend` (`fly.toml`)
and `peace-ml` (`ml-backend/fly.toml`). Each keeps its state in a SQLite file
on its own Fly volume (`peace_web_data`, `peace_ml_data`); videos are in S3.
There is no database server anywhere. The local setup above does not depend
on any of it.

### How a deploy happens

Push to `main`. Two workflows each decide whether they need to run:

| Workflow | Runs when | Does |
|---|---|---|
| `deploy-web.yml` | anything outside `ml-backend/` changes, docs excepted | frontend tests, then deploys `peace-frontend` |
| `deploy-ml.yml` | `ml-backend/` changes, except `training/` and docs | backend tests, then deploys `peace-ml` |

`make deploy` runs both by hand and `make prod-status` shows the machines.

Database migrations and the admin seed run when the frontend machine boots
(the `CMD` in `Dockerfile.frontend`). A Fly release command would run in a
temporary machine without the volume, so it cannot reach the file; at boot
both steps take about a second and are no-ops when nothing changed.

Fly snapshots the volume daily and keeps five days. Restoring a snapshot is
`fly volumes snapshots list` then `fly volumes create --snapshot-id`.

Do not run `fly deploy` from a laptop. It uploads the local directory as the
build context, and a working copy can hold patient-derived files
(`ml-backend/runs/`, `.env`) that a CI checkout never has.

### Secrets

Set with `fly secrets set NAME=value -a <app>`.

| App | Secret | Purpose |
|---|---|---|
| `peace-frontend` | `NEXTAUTH_SECRET` | Signs sessions: `openssl rand -base64 32` |
| | `ML_API_KEY` | Shared key for the ML backend |
| | `SEED_ADMIN_PASSWORD` | Password for the admin account the seed creates; `SEED_ADMIN_EMAIL` changes its address |
| | `S3_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Video storage |
| `peace-ml` | `PEACE_API_KEY` | Same value as `ML_API_KEY` |
| | `PEACE_S3_BUCKET`, `PEACE_AWS_ACCESS_KEY_ID`, `PEACE_AWS_SECRET_ACCESS_KEY` | Video storage |

GitHub Actions needs one secret, `FLY_API_TOKEN`.

With `PEACE_API_KEY` set, the ML backend refuses analyze requests that lack it
and live sockets that lack a token from the web app. Locally both keys are
unset and nothing is checked.

### Using the GPU box for compute

The Fly ML machine is a shared two-CPU box. When the GPU box at home is up,
the web app sends analysis there instead and only falls back to the Fly
machine when the box does not answer.

The box is a WireGuard peer on the Fly organisation's private network
(`oslo-home-wg`, `fdaa:6:1210:a7b:8cfe:0:a:602`, listed by
`fly wireguard list personal`), so `peace-frontend` reaches it directly and
nothing on the box is exposed to the internet. The live WebSocket still goes
to the Fly machine, because the browser cannot reach the private network.

On the box, once:

```bash
sudo wg-quick up /path/to/oslo-home-wg.conf   # the peer config from `fly wireguard create`
cd ml-backend && cp gpu-box.env.example gpu-box.env   # then fill it in
```

Then whenever it should take the work, `make serve` in `ml-backend/`. That is
the whole switch: the web app's `fly.toml` already names the box as
`ML_BACKEND_GPU_URL`, and a request waits at most 2.5 seconds for it before
using the Fly machine, after which the box is retried once a minute. Stopping
`make serve` hands the work back. To drop the preference entirely, remove the
line from `fly.toml`.

An analysis is stored by the backend that ran it. The web app asks the other
backend when one says it has never heard of an id, so results from both are
readable as long as both answer; results of analyses run on the box are not
readable while the box is off. The admin page's Models tab likewise shows
whichever backend answers first.

The peer was created with the `iad` gateway; recreating it with `ams`
(`fly wireguard remove personal oslo-home-wg`, then
`fly wireguard create personal ams oslo-home-wg`, and a new `.conf` on the
box) shortens the path from Amsterdam.

### Giving someone access

Accounts are invite-only. Sign in as an admin, open `/admin`, choose **Add
user**, generate a password and tick **Require password change on first
login**. Send them the link and that password.

### Cost

While idle the bill is the two volumes (10 GB for `peace-ml`, 1 GB for
`peace-frontend`, $1.65/month together) plus storage for the two stopped
machines, well under $1. While awake the frontend costs
about $0.005/hour and the ML backend about $0.037/hour, and each stops itself
after a few idle minutes. The first request after a stop waits for a cold
start. Fly prices as of October 2026.

## Notes

- The model weights (`ml-backend/app/models/best_model.pt`, ~73 MB) are committed,
  so a fresh clone can run analysis with no extra download.
- The ML backend requires Python 3.11 or 3.12 — PyTorch has no 3.13+ wheels yet.
  `uv sync` picks a compatible interpreter automatically.
