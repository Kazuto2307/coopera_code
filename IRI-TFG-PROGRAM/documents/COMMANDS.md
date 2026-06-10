# COMMANDS — IRI-TFG-PROGRAM

All commands assume you are in `IRI-TFG-PROGRAM/` unless stated otherwise.
Set `PYTHONPATH` once (or activate the environment that already has it):

```bash
# From IRI-TFG-PROGRAM/
export PYTHONPATH="$(pwd)/src"   # Linux / macOS / Git Bash
# PowerShell: $env:PYTHONPATH = "$(Get-Location)\src"
```

---

## A. Setup

```bash
# Install requirements (from coopera_code/ root, adjust env as needed)
pip install -r requirements.txt           # if requirements.txt exists

# Verify imports (smoke test — CPU, ~1s)
cd IRI-TFG-PROGRAM
python -c "
from iri_tfg_program import PROJECT_ROOT
from iri_tfg_program.taxonomy.preference_taxonomy import PREFERENCE_SIGNALS
from iri_tfg_program.situations.build_from_external import main
print('OK - PROJECT_ROOT:', PROJECT_ROOT)
"
```

---

## B. Generate synthetic user profiles

**Input:** COOPERA dataset root (parent of IRI-TFG-PROGRAM)
**Output:** `data/generated/profiles/*.json`
**Hardware:** CPU

```bash
# Standard run (25 profiles, gpt_response source)
python scripts/generate_profiles.py \
  --coopera-root .. \
  --num-profiles 25 \
  --output-dir data/generated/profiles

# Dry-run (no Qwen calls, checks everything else)
python scripts/generate_profiles.py --dry-run

# Specific profile indices
python scripts/generate_profiles.py \
  --profile-indices 0 1 2 3 4 \
  --output-dir data/generated/profiles
```

---

## C. Pipeline A — External Situations (Charades / EPIC)

### C1. Build raw situations (CPU)

**Input:** `data/external/raw/charades/` + `data/external/raw/epic/`
**Output:** `data/generated/situations_external/external_situations.jsonl`

```bash
# Charades only (default)
python scripts/build_situations_from_external.py \
  --sources charades \
  --output data/generated/situations_external/external_situations.jsonl

# Both Charades + EPIC-100
python scripts/build_situations_from_external.py \
  --sources charades epic \
  --output data/generated/situations_external/external_situations.jsonl

# Limit per source (smoke test)
python scripts/build_situations_from_external.py \
  --sources charades \
  --max-per-source 500 \
  --output data/generated/situations_external/external_situations_smoke.jsonl
```

### C2. Analyse situations (CPU)

**Input:** situations JSONL file
**Output:** console + study/ folder with plots

```bash
python scripts/study_situations.py \
  --input data/generated/situations_external/external_situations.jsonl \
  --output-dir data/generated/situations_external/study
```

### C3. Translate to robot-action situations (Qwen — GPU)

**Input:** `external_situations.jsonl`
**Output:** `external_situations_robot.jsonl`

```bash
# Full run
python scripts/translate_situations.py \
  --input data/generated/situations_external/external_situations.jsonl \
  --output data/generated/situations_external/external_situations_robot.jsonl \
  --batch-size 20

# Quick sample (smoke test, 50 situations)
python scripts/translate_situations.py \
  --input data/generated/situations_external/external_situations.jsonl \
  --output data/generated/situations_external/external_situations_robot_sample50.jsonl \
  --sample 50 --seed 42 --batch-size 20
```

### C4. Generate training data (Qwen — GPU)

**Input:** robot situations + profiles
**Output:** `data/generated/preference_training_<timestamp>.jsonl`

```bash
python scripts/generate_training_data.py \
  --profiles-dir data/generated/profiles \
  --situations data/generated/situations_external/external_situations_robot.jsonl \
  --target-samples 5000 \
  --seed 0 \
  --output data/generated/preference_training_5k.jsonl \
  --summary-output data/generated/preference_training_5k_summary.json
```

### C5. Full Pipeline A (orchestrated)

```bash
python scripts/run_pipeline.py \
  --sources charades \
  --num-profiles 25 \
  --target-samples 5000 \
  --seed 0

# Dry-run (validates arguments, no Qwen)
python scripts/run_pipeline.py --dry-run

# Skip steps that already ran
python scripts/run_pipeline.py \
  --skip-profiles \
  --skip-build \
  --skip-translate \
  --target-samples 5000
```

---

## D. Pipeline B — Hourly Situations

### D1. Build hourly situations (CPU)

```bash
python scripts/build_situations_hourly.py \
  --output data/generated/situations_hourly/situations_hourly.jsonl
```

### D2. Generate training data from hourly situations (Qwen — GPU)

```bash
python scripts/generate_training_data_hourly.py \
  --profiles-dir data/generated/profiles \
  --situations data/generated/situations_hourly/situations_hourly.jsonl \
  --target-samples 3000 \
  --output data/generated/preference_training_hourly.jsonl
```

---

## E. Demo / Analysis

```bash
# Interactive demo of the full pipeline on a small sample
python scripts/demo_situations_pipeline.py \
  --raw-dir data/external/raw \
  --n-situations 10

# Analyse an existing situations file
python scripts/study_situations.py \
  --input data/generated/situations_external/external_situations.jsonl
```

---

## F. Module-level invocation (alternative to scripts/)

If `PYTHONPATH=src` is set:

```bash
python -m iri_tfg_program.situations.build_from_external --help
python -m iri_tfg_program.training_data.generate_training_data --help
python -m iri_tfg_program.profiles.generate_profiles --help
```
