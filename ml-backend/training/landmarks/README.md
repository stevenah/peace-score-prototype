# Landmark data pipeline (M1)

Turns the ESGE-10 station clips in `data/landmarks/` into a patient-grouped
manifest, a masked frame cache and locked splits for training the landmark
classifier (`app/ml/landmarks/`, schema `esge10.v1`). Runs on the laptop; only
the masked frame cache and the manifest go to the GPU box.

## Privacy rules (the repo is public)

- `data/` and `ml-backend/training/landmarks/cache/` are gitignored. Never
  force-add anything from them.
- Everything dataset-derived is written to `data/landmarks_private/`
  (manifest, overrides, OCR, adjudication sheets, reports) or to the cache.
- The only dataset-derived file in the tracked tree is
  `manifest_summary_v1.json`: counts only, no IDs, dates or paths
  (`tests/test_landmark_manifest.py` checks it).
- The manifest holds no absolute dates or scope serials: processor ("room"),
  scope and recording day are salted hashes (salt in
  `data/landmarks_private/.salt`, created on first run); times are relative to
  the patient's first clip (`rel_t_s`) or time of day (`tod_s`).
- The five full-length recordings (`data/landmarks/Full lenght videos/`) are
  never listed, read or referenced: `common.iter_clips` skips the folder,
  every ingested audit table is filtered with `drop_full_length`, and the
  manifest writer, cache selector and `folds --verify` refuse such rows.

## Run it end to end

From `ml-backend/`, with the `train` extra installed (`uv pip install -e ".[train]"`)
plus `ffmpeg` and `tesseract` on the PATH:

```bash
PY=.venv/bin/python
# 0. once: copy the read-only audit tables in (REC_* rows are dropped)
$PY -m training.landmarks.build_manifest import-audit <audit scratchpad dir>
# 1. manifest, phase A: probe + md5, IDs, OSD clock, processor OCR, duplicates,
#    groups, overlaps, exclusion policy (~2 min first run, cached after)
$PY -m training.landmarks.build_manifest
# 2. legacy layout geometry (only when re-deriving; paste into layouts.py and
#    bump LAYOUTS_VERSION)
$PY -m training.landmarks.derive_layouts legacy
# 3. frame cache (~10 min, incremental)
$PY -m training.landmarks.extract_cache
# 4. manifest, phase B: NBI fraction, frozen, cap, near-duplicates from the cache
$PY -m training.landmarks.build_manifest
# 5. static-overlay scan: exits 1 if anything is drawn inside the aperture
$PY -m training.landmarks.derive_layouts static
# 6. splits (locked test + 5 dev folds), then integrity check
$PY -m training.landmarks.folds
$PY -m training.landmarks.folds --verify
# 7. leak audit: embeddings + gain-invariant dHash rule, planted positive controls (exits 1 if missed)
$PY -m training.landmarks.leak_audit
# 8. clinician adjudication material (M1a)
$PY -m training.landmarks.contact_sheets
```

Every rebuild of the manifest resets `split`/`fold`: always re-run `folds`
afterwards (it is deterministic). After clinicians fill the adjudication
CSVs: `contact_sheets collect` → `build_manifest` → `folds` → `folds --verify`
(the manifest becomes v1.1; the cache does not need re-extraction).

## Ship to the GPU box

Only masked 320 px canvases and the manifest leave the laptop. Never copy
`data/landmarks/` (raw video with burned-in dates and serials) or
`data/landmarks_private/audit/` (OSD OCR with dates and scope serials).

```bash
cd /path/to/peace-web-prototype
GPU=user@gpu-box:/data/peace/landmarks
rsync -a --partial --info=progress2 --exclude '_parts/' \
  ml-backend/training/landmarks/cache/v1/ "$GPU/cache/v1/"
rsync -a data/landmarks_private/manifest_v1.csv "$GPU/"
# on the GPU box: check the transfer
python -c "import json;print(json.load(open('/data/peace/landmarks/cache/v1/CACHE_INFO.json'))['counts'])"
sha256sum /data/peace/landmarks/cache/v1/frames.csv.gz   # == CACHE_INFO frames_csv_sha256
```

## Modules

| Module | Role |
|---|---|
| `common.py` | paths, `iter_clips` (full-length never listed), ffmpeg decode with the explicit BT.709 chain, CSV/JSON IO, salted hashes |
| `ids.py` | pure filename parser: patient ID (dirty forms), station suffix, take, copy markers; override table loader |
| `build_manifest.py` | the manifest and `manifest_summary_v1.json`; `import-audit` |
| `derive_layouts.py` | legacy geometry, static-overlay scan, cap-ring detector |
| `extract_cache.py` | frame cache, per-frame quality, dHash, frame duplicates, near-duplicate search |
| `folds.py` | locked test + dev folds; `--verify` |
| `leak_audit.py` | cross-split audit: ResNet-50 embedding nearest neighbours + the gain-invariant dHash near-duplicate rule, planted controls |
| `contact_sheets.py` | adjudication sheets/templates; `collect` into `label_overrides.csv` |

## Formats

### `data/landmarks_private/manifest_v1.csv` (one row per clip file)

Byte-identical copies filed under two station folders collapse into one row
(`alt_rel_path` lists the other copy). Exact duplicates within a folder keep a
row each (`exclude_reason=dup_exact`, `clip_uid` suffixed `-dupN`).

| Column | Meaning |
|---|---|
| `clip_uid` | `md5[:12]` (unique; `-dupN` for excluded exact copies); cache folder name |
| `rel_path`, `alt_rel_path` | path under `data/landmarks/`; other identical copies |
| `folder_class`, `suffix`, `suffix_class` | station of the folder; parsed suffix tokens (`lc+inc`); station named by a single known suffix |
| `label`, `label_source` | station key used for stratification/training; `folder` or `adjudicated` |
| `soft_label` | e.g. `lesser_curvature_retroflex:0.5\|incisura:0.5`; when set it replaces `label` as the training target |
| `exclude_reason` | empty = usable; else `dup_exact`, `too_short` (≤2 frames), `dup_near`, `id_uncertain`, `manual`, `adjudicated_exclude`, `combo`, `quarantine_suffix_mismatch` |
| `eval_exclude` | not part of primary evaluation (every excluded row, and soft-labelled rows) |
| `flags` | `contested`, `combo`, `suffix_ambiguous`, `frozen`, `dup_near`, `cap_by_patient`, `time_overlap_other_label`, `adjudicated`, `manual:<reason>`, `unknown_layout` |
| `dup_group`, `near_dup_group` | md5 group of exact copies; clips sharing near-identical canonical frames |
| `patient_id`, `patient_id_raw`, `id_resolution` | `HT###`; ID token as typed; `regex`, `regex_letter_o`, `regex_padded`, `override`, `uncertain`, `ambiguous`, `unparsed` |
| `group_id` | the patient, merged with another patient only if a clip's firm OSD time lies inside that patient's (day, processor) window ±2 min |
| `room_hash`, `scope_hash`, `day_key`, `tod_s` | salted hashes of the processor fingerprint, scope model+serial and OSD date; OSD time of day in s |
| `cohort`, `layout` | `new_1920` / `legacy_1350`; layout name in `layouts.py` |
| `width`, `height`, `n_frames`, `fps`, `dur_s`, `md5` | ffprobe + file md5 |
| `nbi_frac`, `mode_major` | share of cached frames whose badge reads NBI/RDI; `wl` (≤0.2), `nbi` (≥0.8), `mixed` |
| `cap`, `cap_score` | transparent distal cap: clip score ≥ `derive_layouts.CAP_THRESHOLD` (0.35), or the patient has ≥ 50% such clips (a cap stays on for the procedure) |
| `frozen`, `n_cached`, `n_unique` | < 4 unique frames or < 25% unique among cached frames |
| `rel_t_s`, `order_in_patient` | OSD start relative to the patient's first clip; chronological rank |
| `overlap_head_s`, `overlap_tail_s`, `overlap_with` | seconds at the start/end of the clip that overlap (OSD clock, 1 s resolution) a clip of the same patient with a different label; drop those frames in training |
| `split`, `fold` | `dev` (fold 0..4), `test`, `shift` (legacy), `excluded` (no safe patient group: dup_exact, too_short, dup_near, id_uncertain); `-1` when not dev |

Combo and quarantined clips keep their patient's split (for the replay
benchmark) but always carry an `exclude_reason`: never train or score on a
row with a non-empty `exclude_reason`.

### `cache/v1/`

- `frames/{clip_uid}/{src_idx:05d}.jpg`: 320×320 canonical canvas
  (`app.ml.landmarks.preprocess.canonicalize`), JPEG q95 4:4:4, every 3rd
  source frame (10 fps). Pixels outside the aperture are 0 before JPEG; the
  model input is re-masked by `to_model_input`.
- `frames.csv.gz`: `clip_uid, src_idx, u, t_s, sharpness, dark_frac,
  sat_frac, redout, quality, sharp_ratio_to_clip_median, dhash, dup_of,
  mode_frame, layout_ok`. Quality metrics come from `app.ml.landmarks.quality`
  on the `QUALITY_SIZE` (224 px) model input with default `QualityParams`,
  whatever the model's input size (serving measures quality at the same size); `u` = position in the
  clip (0..1); `dup_of` = src_idx of an earlier near-identical frame of the
  same clip; `mode_frame` = `detect_mode` on the full frame; `layout_ok` =
  `detect_layout` verified the frame.
- `thumbs.npy`: uint8 (n, 32, 32) grey centre crops, row-aligned with
  `frames.csv.gz`.
- `CACHE_INFO.json`: preprocess/layout versions, `clip_set_sha256` (sorted
  `clip_uid:md5` of the extracted clips; unchanged by later manifest edits),
  manifest sha256 at extraction time, ffmpeg version and filter, git sha,
  counts, `frames_csv_sha256`.

### Reviewable overrides (`data/landmarks_private/overrides/`)

| File | Columns | Effect |
|---|---|---|
| `patient_id_overrides.csv` | `match_type` (`file`\|`raw_id`), `match`, `patient_id` (`HT###`\|`uncertain`), `reason`, `reviewer` | file beats raw-ID beats the generic rule |
| `label_overrides.csv` | `clip_uid`, `label` (station key \| `k:w\|k:w` \| `keep` \| `exclude`), `reason`, `reviewer` | `label_source=adjudicated`; lifts combo/quarantine |
| `exclusions.csv` | `key` (clip_uid or rel_path), `exclude_reason`, `reason`, `reviewer` | `exclude_reason=manual` |

## Numbers (run of 2026-09-28)

**Manifest.** 1,723 clip files in the 10 station folders → 1,697 rows (the 26
Incisura/Lesser-curvature byte-identical pairs are one row each, soft label
0.5/0.5, `eval_exclude`). 98 patients (91 new cohort, 7 legacy). Usable
1,620; excluded: 56 `dup_exact` (D2 copies), 3 `dup_near` (re-encodes found on
the cache; the same 3 clips the audit flagged),
3 `too_short`, 3 `id_uncertain`, 8 `combo`, 4 `quarantine_suffix_mismatch`.
IDs: 1,654 regex, 16 letter-O, 19 padded two-digit, 5 override, 3 uncertain;
0 md5 mismatches with the audit; 0 OSD-date outliers after overrides.

**Rooms.** The OSD panel OCR separates two processors by configuration
(scope button 2, confirmed by the empty comment placeholder), not by scope
(scopes move between rooms):
1,020 new-cohort rows on processor A, 505 on B, 0 unresolved. No (date,
processor) window merges were needed in the new cohort. The legacy recorder's
OCR is unreliable, so only *firm* OSD times (first and last frame agree with
the duration) can merge groups; an earlier draft merged two legacy patients
on a misread clock, which this rule prevents. 52 rows overlap a different-label
clip of the same patient in OSD time (mostly 0.1–1 s, i.e. clock resolution).

**Cache.** 1,638 clips, 94,573 frames (10 fps), 3.2 GB JPEG; re-extracted
with lay-3 (the border-eroded mask) on 2026-09-28, same clip set
(`clip_set_sha256` b75517f4…). Freeze holds are common: 15.5% of frames
duplicate an earlier frame of their clip and 73% of clips contain a hold (the
photo moment); 7 clips are `frozen` (< 25% unique frames). Median sharpness
of centre frames at 224 px: 1,133 over all cached clips, 1,118 over usable
dev clips (`calibrate.sharp_ref_from_cache`, the shipped reference; the
`QualityParams.sharp_ref` placeholder is 1,100). Removing the rim changes a
frame's sharpness by a median factor of 0.998 but 0.88 at P10, which is why
the metrics were recomputed rather than kept from lay-2. NBI: usable clips 841 WL,
729 NBI, 50 mixed; WL oesophagus is scarce (proximal 9, distal 15 clips).

**Cap.** Score = share of a circle's in-aperture arc with a thin, static
edge (temporal median over unique frames). 16 patients are clearly cap
patients (73–100% of their clips ≥ 0.35; every other patient ≤ 33%), so the
flag is propagated per patient (28 clips): 310 clips flagged, including
107 of 237 Z-line clips. 5 isolated positives (3 visually confirmed caps in
one patient's Z-line clips, 2 borderline single-frame clips). Circular duodenal
folds scored high only in freeze holds; scoring unique frames removed them.

**Legacy layout (lay-2).** Measured over 2,134 frames of 171 legacy clips:
aperture (416,88)–(1138,88)–(1295,300)–(1295,781)–(1140,991)–(414,991)–(260,782)–(260,298),
ROI (260,88,1296,992); 4 OSD boxes inside the aperture (patient-ID/name
labels, birth-date/age/sex labels, comment field, NBI badge + icon).

**Static-overlay scan: OK** (lay-3). Global: 4,000 canvases, minimum
temporal std inside the mask 29.6 (fail < 2). Per clip: 1,458 clips with
>= 8 unique frames, no static pixel recurring in >= 2% of clips (max 20
clips; 24 at the left rim before the lay-3 mask fix).

**Splits.** Test: seed 0 / fold 0, 19 patients (8 with all 10 stations), NBI
share per family within 1.5 pp of dev. Dev: 72 patients, 5 folds, seed 45,
max per-fold per-class relative deviation 0.12. `folds --verify`: OK.

| station | dev | test | shift | dev folds 0–4 (eval clips) |
|---|---|---|---|---|
| esophagus_proximal | 116 | 30 | 12 | 23 22 26 23 22 |
| esophagus_distal | 82 | 24 | 13 | 15 17 17 16 17 |
| z_line | 185 | 43 | 9 | 36 41 33 38 37 |
| duodenal_bulb | 91 | 20 | 13 | 18 19 18 18 18 |
| duodenum_descending | 88 | 24 | 10 | 18 18 17 17 18 |
| antrum | 118 | 32 | 32 | 23 25 23 24 23 |
| cardia_fundus_retroflex | 114 | 30 | 21 | 23 23 25 20 23 |
| lesser_curvature_retroflex | 73 | 18 | 22 | 14 14 15 15 15 |
| incisura | 100 | 23 | 10 | 18 15 15 16 18 |
| corpus_greater_curvature | 193 | 48 | 26 | 39 39 39 38 38 |

(usable clips; test eval clips for incisura = 15 because 8 contested clips
are soft-labelled.)

**Leak audit: PASS** (lay-3 cache). Distinct takes of one patient reach at
most 0.904 cosine, so T_dup = 0.952; 0 cross-split and 0 cross-patient pairs
above it. The gain-invariant dHash rule finds 0 near-duplicate clip pairs
among dev/test/shift clips. Controls: planted re-exported copy 0.986
(embedding and dHash: caught), shifted stress copy (2 px shift, q75, +3%
gain) 0.896 by embedding (missed) but caught by the dHash rule (required:
either detector), the 3 real re-encodes 1.000 (caught); the re-cut (0.795)
is informative. Measured on 40 test clips, embeddings never catch the
shifted copy (0.82–0.91) and the plain dHash rule catches 70% (the +3% gain
pushes the thumbnail difference over 4.0); the gain-invariant rule catches
59/60. ImageNet features do not identify patients (56% false alarms at 90%
sensitivity), so "same person under two IDs" needs the custodian's
repeat-procedure mapping.

**Adjudication (M1a).** Sheets and templates: 26 contested, 4 suffix
mismatch, 8 combo, 100 blind label-audit clips (10 per station).

### Mask border (fixed in lay-3)

`app/ml/landmarks/layouts.aperture_mask` used to erode with OpenCV's default
border, which treats pixels outside the image as foreground. The OLYMPUS
octagon's vertical sides sit on the canvas border (ROI x0 = 550, x1 = 1900),
so canvas columns 0–3 and 316–319 (the aperture rim) stayed in every model
input. lay-3 erodes with `borderType=BORDER_CONSTANT, borderValue=0`. The
cache was re-extracted: `to_model_input` re-masks images, but the cached
quality metrics were measured on the old mask. The manifest was NOT rebuilt
(a rebuild resets splits): its `cap`/`cap_score` columns were computed on
lay-2 canvases (the only phase-B column computed from whole canvases rather
than the badge or the centre thumbnails).
Runs and bundles made before lay-3 (`runs/mps_short_effb0`,
`runs/smoke_dev`) carry lay-2 metrics; `lm-0.0.4-mpsfold0` is a lay-3
re-export of the MPS fold-0 model for the slow parity test.

## Training, evaluation and export (M2–M4)

Runs on the GPU box against the rsynced `cache/v1/` + `manifest_v1.csv`
(install the `train` extra and the CUDA torch 2.10 / torchvision 0.25 wheels).
Everything a run writes goes to the gitignored `ml-backend/runs/<exp>/`
(`<exp>` = config name + `_s<seed>`); only aggregate tables go to `results/`.
Commands are run from `ml-backend/` with `PY=.venv/bin/python`.

```bash
# 0. once, on the laptop: our endoscopy-position backbone -> clean state_dict
#    (cache/weights/, copy it to the GPU box with the cache); fetch+verify the rest
$PY -m training.landmarks.backbones prepare-endopos
$PY -m training.landmarks.backbones fetch
# Stage A: frozen probes (laptop MPS is enough) -> results/stageA_v1.csv
$PY -m training.landmarks.embed_probe --device mps
# Stage B: fine-tune, 5 outer folds per candidate (LR sweep: --fold 0 --set optim.lr_backbone=3e-4 ...)
for k in 0 1 2 3 4; do $PY -m training.landmarks.train --config stageB_regy32_in1k --fold $k; done
# sanity: G-PIPELINE, then (winner) ablations on fold 0 with configs/ablation_*.json (set "base")
$PY -m training.landmarks.sanity perm --config sanity_perm --exp-out runs/stageB_regy32_in1k_s0
# M3: calibration (cross-fit) -> replay benchmark (descriptive) -> nested tracker tuning -> shortcuts
$PY -m training.landmarks.calibrate --exp runs/stageB_regy32_in1k_s0
$PY -m training.landmarks.stream_eval --exp runs/stageB_regy32_in1k_s0 --transit model --gate-off
#    nested: per outer fold tuned tracker + thresholds + auto_enabled -> G-STREAM, G-STATION S4 FDR, G-TEST refs
$PY -m training.landmarks.tune_tracker --exp runs/stageB_regy32_in1k_s0 --transit model
#    G-STREAM failed? dev-only retune: force the worst station manual-only, one at a time, then --report again
# $PY -m training.landmarks.tune_tracker --exp runs/stageB_regy32_in1k_s0 --transit model --disable esophagus_proximal
$PY -m training.landmarks.sanity shortcuts --exp runs/stageB_regy32_in1k_s0
$PY -m training.landmarks.sanity unmasked-cache            # laptop (reads raw clips); rsync cache/v1_unmasked/
$PY -m training.landmarks.train --config sanity_unmasked --fold 0
$PY -m training.landmarks.sanity positive-control --run runs/sanity_unmasked_s0/fold0 --exp runs/stageB_regy32_in1k_s0
$PY -m training.landmarks.evaluate --exp runs/stageB_regy32_in1k_s0 --report   # gates_report.md, auto_enabled, outcome
$PY -m training.landmarks.evaluate --exp runs/stageB_regy32_in1k_s0 --ci-sim   # before freezing gates.json
# M4: final model (same schedule, stopped at the median best fold step), 3 seeds, deploy seed 0
$PY -m training.landmarks.train --config stageB_regy32_in1k --all-dev --steps-from runs/stageB_regy32_in1k_s0 --exp stageB_regy32_in1k_s0
$PY -m training.landmarks.evaluate --predict test --run runs/stageB_regy32_in1k_s0/all_dev
$PY -m training.landmarks.evaluate --predict shift --run runs/stageB_regy32_in1k_s0/all_dev
#    ONCE; transit classified by all_dev/final_<variant>.pt, shipped auto_enabled; refused (nothing scored)
#    unless tracker.json was tuned with --transit model and gates_result.json exists
$PY -m training.landmarks.evaluate --exp runs/stageB_regy32_in1k_s0 --test --transit model
$PY -m training.landmarks.evaluate --exp runs/stageB_regy32_in1k_s0 --report   # outcome now includes G-TEST
$PY -m training.landmarks.export_bundle --exp runs/stageB_regy32_in1k_s0 --version lm-1.0.0
# I1: online == offline against a local server started with PEACE_LANDMARK_BUNDLE_DIR=<bundle>
$PY -m training.landmarks.replay_ws --patient HT### --bundle runs/<exp>/all_dev/bundle/lm-1.0.0
# laptop smoke of the whole path (MPS/CPU): train -> export -> load_bundle
$PY -m training.landmarks.train --config smoke --smoke --device mps
$PY -m training.landmarks.export_bundle --run runs/smoke_<stamp>/smoke --version lm-0.0.1-smoke --smoke
```

### Modules

| Module | Role |
|---|---|
| `backbones.py` | allowlist of permissive weights (torchvision enums, HF repos pinned to a commit, our endopos checkpoint) with sha256; `load_pretrained(id)` refuses anything else |
| `dataset.py` | manifest/cache tables, training-frame filters, hierarchical class→patient→clip sampler with per-patient cap and NBI/WL balancing, torch datasets |
| `augment.py` | cv2 augmentation, mask applied last; `assert_batch_masked` |
| `train.py` | config-driven fine-tuning (`--fold`, `--all-dev`, `--holdout`, `--smoke`), EMA, early stopping on inner-val NLL, fp32 OOF |
| `embed_probe.py` | Stage A frozen probes (LR + kNN) and confound decodability |
| `calibrate.py` | T, per-station tau (closed-set precision), q_min, sharp_ref; cross-fit per outer fold, shipped on all OOF |
| `stream_sim.py`, `stream_eval.py`, `tune_tracker.py` | Procedure Replay Benchmark (S0/S2/S4/S5, 1 Hz) through the serving `StationTracker`; nested tuning |
| `evaluate.py` | metrics with patient/group bootstrap CIs, merged hierarchy, gates report, CI-width simulation, locked test (once), temporal split |
| `sanity.py` | G-PIPELINE permutation run, NBI / mode-switch / cap shortcut tests, unmasked positive control, blank-frame check |
| `lso.py` | leave-station-out MSP vs energy vs kNN (implemented, not yet run) |
| `export_bundle.py` | `lm-<semver>/` via `app.ml.landmarks.bundle.write_bundle`, CPU selftest, tarball + sha256, `aws s3 cp` and `BUNDLE.lock` text |
| `replay_ws.py` | online (`/api/v1/ws/live`) vs offline station sequence on one patient |
| `run_utils.py` | run dirs, provenance, metrics, bootstrap |
| `gates.json`, `configs/*.json` | pre-registered gates; Stage B, ablation, sanity and smoke configs (`"base"` inherits) |

### Run formats

- `runs/<exp>/fold{k}/`: `config.json` (resolved), `provenance.json` (git sha
  + dirty flag, manifest sha256, cache `clip_set_sha256`/`frames_csv_sha256`,
  versions, torch, device), `metrics.json` (eval history for raw AND EMA,
  `best = {variant, step, nll, ...}`, mode-balance report), `best_raw.pt`,
  `best_ema.pt` (fp32 state_dicts of `architectures.build(arch)`).
- `oof_fold{k}.npz` / `inner_fold{k}.npz` (outer / inner-val patients, EVERY
  cached frame of their usable + ambiguous clips) and `oof_fold{k}_alt.npz`
  (the other variant, logits only): `logits` f32 (n,10, no autocast), `emb`
  f16, `clip_uid`, `src_idx`, `u`, `t_s`, `sharpness`, `dark_frac`,
  `sat_frac`, `redout`, `sharp_ratio`, `dup`, `mode_frame`, `layout_ok`, and
  scalars `variant`, `fold`, `input_size`, `candidate`.
- Experiment level: `calibration.json` (`per_fold` cross-fit {T, tau, q_min},
  `temps_heldout` {k: {j: T_j}} = the temperatures to use while fold k is
  reported (model k-1's T refit on its own outer fold, since its inner-val IS
  fold k), `shipped`, `quality_params`, sensitivity),
  `stream_eval_<transit>.json` (descriptive; `--auto-from gates_result.json`
  applies a station set), `tracker.json` (per-fold selection, `auto_enabled`,
  `station_gate`, `temperatures` and held-out score; `heldout_pooled` S2/S4
  on the per-fold station sets, S4 also `fdr_per_station_if_enabled` = the
  G-STATION input; `transit_source`, `force_disabled`, `shipped` tracker,
  `shipped_calibration`), `tracker_heldout_{S2,S4}_stations.csv.gz` (the
  held-out station records; `--ci-sim` input), `sanity.json`, `eval_oof.json`,
  `gates_result.json` (`auto_enabled`, `auto_enabled_by_mode` (informational),
  `s4_fdr_source`, `outcome` GO/PARTIAL/NO-GO/PENDING), `gates_report.md`,
  `ci_sim.json` (test-sized patient sets drawn WITH replacement),
  `eval_test.json` (`transit_source` {test, dev_reference}, `auto_enabled`),
  optional hand-written `serve_result.json` {"status": "PASS"|"FAIL", ...}
  from I1 (G-SERVE in the outcome).
- Station records (`stream_*_stations.csv.gz`, `tracker_heldout_*`): one row
  per stream x station with `present`, `allowed`, `auto` (enabled in that
  replay), `observed` (live: never for a disabled station),
  `observed_if_enabled`, `on_transit` (the observing frame was a synthetic
  transit frame), `tto`, `within_clip`.
- Replay (stream_sim.py): a synthetic transit frame is made only from a clip
  that was already replayed in the stream, so S2 never previews a later
  station and S4 never makes transit from the removed station's clips.
- Numerics: every logit that feeds T, tau, q_min, a replay or a gate (OOF,
  inner, test, transit) is fp32 with TF32 off (`run_utils.fp32_inference`);
  `provenance.json` records `tf32_training` and `inference_precision`.

### Stage A (frozen probes, laptop MPS, 2026-09-28)

`results/stageA_v1.csv`: 1,142 primary dev clips, 72 patients, 5 frames per
clip, 5 patient-grouped folds, 95% patient-bootstrap CIs.

| candidate | LR macro-F1 [95% CI] | kNN macro-F1 | mode | cap | room |
|---|---|---|---|---|---|
| cnxt_t_in1k | 0.728 [0.696, 0.758] | 0.571 | 0.998 | 0.988 | 0.741 |
| dinov2_s14 | 0.721 [0.687, 0.749] | 0.595 | 0.997 | 0.982 | 0.671 |
| dinov2_b14 (probe-only) | 0.716 [0.690, 0.740] | 0.585 | 0.998 | 0.990 | 0.663 |
| effb0_in1k | 0.713 [0.682, 0.744] | 0.573 | 0.997 | 0.983 | 0.694 |
| r50_in1k | 0.697 [0.663, 0.727] | 0.543 | 0.997 | 0.984 | 0.708 |
| regy32_in1k | 0.694 [0.666, 0.719] | 0.591 | 0.997 | 0.978 | 0.729 |
| regy32_endopos | 0.675 [0.641, 0.705] | 0.533 | 0.993 | 0.934 | 0.645 |

mode / cap / room = balanced accuracy of a linear probe for the clip's
imaging mode, transparent cap and processor, averaged within stations
(0.5 = not decodable beyond the station). Weakest stations for every
backbone: distal oesophagus (LR F1 0.43–0.56), incisura (0.43–0.64) and
antrum (0.57–0.70).

All CIs overlap, so Stage A does not separate the candidates. The DINOv2-B
ceiling does not beat the CPU candidates, so the "≥ 5 points above the CPU
winner" GPU-host trigger is not met. Imaging mode and cap are almost
perfectly linearly decodable from every backbone (expected: NBI changes the
colours and the cap ring is visible), which is why G-SHORTCUT tests their
effect on the station prediction rather than their visibility.
