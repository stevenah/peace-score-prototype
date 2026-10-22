.PHONY: train train-results serve help

SHELL := /bin/bash

PY         ?= .venv/bin/python
CANDIDATES ?= cnxt_t_in1k dinov2_s14 effb0_in1k regy32_endopos
FOLDS      ?= 0 1 2 3 4
SEED       ?= 0

# -------------------------------------------------------------------
# Landmark training (GPU box) — see training/landmarks/README.md
# -------------------------------------------------------------------

train: ## Fine-tune every candidate on every dev fold (resumes: skips finished folds; FORCE=1 retrains)
	@mkdir -p runs
	@set -o pipefail; \
	for c in $(CANDIDATES); do for k in $(FOLDS); do \
	  if [ -z "$(FORCE)" ] && [ -f runs/stageB_$${c}_s$(SEED)/fold$$k/metrics.json ]; then \
	    echo "skip $$c fold $$k (done)"; continue; \
	  fi; \
	  $(PY) -m training.landmarks.train --config stageB_$$c --fold $$k --seed $(SEED) 2>&1 \
	    | tee -a runs/train_$$c.log || exit 1; \
	done; done
	@$(MAKE) --no-print-directory train-results

# -------------------------------------------------------------------
# Serving the production web app from this machine — README, "Using the GPU
# box for compute". Binds IPv6 because peace-frontend arrives over the Fly
# private network (WireGuard); port 8000 matches the Fly machine.
# -------------------------------------------------------------------

serve: ## Serve the API on [::]:8000 for peace-frontend, with settings from gpu-box.env
	@test -f gpu-box.env || { echo "Copy gpu-box.env.example to gpu-box.env and fill it in."; exit 1; }
	@set -a; . ./gpu-box.env; set +a; $(PY) -m uvicorn app.main:app --host :: --port 8000

train-results: ## Held-out clip macro-F1 per candidate and fold
	@$(PY) -c "import glob, json; \
	rows = sorted(glob.glob('runs/stageB_*/fold*/metrics.json')); \
	[print(f'{p.split(chr(47))[1]:28s} {p.split(chr(47))[2]:6s} {json.load(open(p))[\"oof\"][\"clip_macro_f1\"]:.3f}') for p in rows]"

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
