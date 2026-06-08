# IRI-TFG-PROGRAM

Generacion de datos sinteticos de entrenamiento para un modelo de personalizacion
de decisiones de un robot asistencial: dado un humano (perfil de personalidad) y
una situacion domestica con una accion propuesta del robot, predecir que querria
la persona que hiciera el robot.

- **Labels de decision**: `do_now`, `do_later`, `tell_the_user`, `no_action`.
- **Taxonomia de preferencias**: 11 senales jerarquicas (con `prefer`/`avoid` y
  `weight` 1-10) en `preference_taxonomy.py`.
- **LLM**: Qwen local (`qwen_labeler.py`, GPU; torch lazy). En local sin CUDA se
  prueba con `--dry-run` o stubs.

> Nota de estructura: los scripts estan en plano (mismo directorio) porque se
> importan entre si por nombre (`from human_sim_preference_data import ...`), que
> solo resuelve si comparten carpeta. Se ejecutan como
> `python IRI-TFG-PROGRAM/<script>.py` (ese dir queda en `sys.path[0]`).

---

## Pipelines

Hay tres pipelines. Todos comparten el **paso 1 (perfiles)** y la **traduccion a
accion de robot**.

### Paso 1 (comun): perfiles
`generate_profiles.py` — lee `mypersonality_final.csv` (+ traits opcional) y
genera por humano: `profile_summary`, `preference_profile` (taxonomia nueva:
`{signal_name, polarity, weight}`) y una `description`. Salida:
`generated_data/profiles/human_XXXXX.json`.

```bash
python IRI-TFG-PROGRAM/generate_profiles.py --num-profiles 25
```

### Pipeline A — dataset-grounded libre (principal)
Situaciones reales de Charades (EPIC opcional), traducidas a accion de robot, y
decision LIBRE (sin forzar labels, sin anchoring).

```
build_situations_from_external.py   (CPU)  -> situaciones (accion humana)
translate_situations.py             (GPU)  -> accion de robot
generate_training_data.py           (GPU)  -> JSONL de entrenamiento (decision libre)
```
Orquestador: `run_pipeline.py` (perfiles -> build -> translate -> training;
`--skip-*`, `--sample N` para pruebas).

```bash
python IRI-TFG-PROGRAM/run_pipeline.py --num-profiles 25 --target-samples 3000 \
  --output IRI-TFG-PROGRAM/generated_data/preference_training.jsonl
```

### Pipeline B — horario minimo (nuevo)
Como A pero cada situacion se expande a **24 copias (una por hora)** y la
estructura es **minima**: `structured_task_features = {kind, quiet_hours}` (se
omiten urgency/sensitivity/user_busy/conditions/context_flags). Sin invencion por
LLM en el build.

```
build_situations_hourly.py          (CPU)  -> situaciones minimas x24h
translate_situations.py             (GPU)  -> accion de robot
generate_training_data_hourly.py    (GPU)  -> JSONL (muestras minimas)
```

```bash
python IRI-TFG-PROGRAM/build_situations_hourly.py \
  --output IRI-TFG-PROGRAM/generated_data/situations_hourly/situations_hourly.jsonl
python IRI-TFG-PROGRAM/translate_situations.py \
  --input  IRI-TFG-PROGRAM/generated_data/situations_hourly/situations_hourly.jsonl \
  --output IRI-TFG-PROGRAM/generated_data/situations_hourly/situations_hourly_robot.jsonl
python IRI-TFG-PROGRAM/generate_training_data_hourly.py \
  --situations IRI-TFG-PROGRAM/generated_data/situations_hourly/situations_hourly_robot.jsonl \
  --target-samples 3000 --output IRI-TFG-PROGRAM/generated_data/preference_training_hourly.jsonl
```
Charades son ~64k base x24 = ~1.5M situaciones: para pruebas usa
`build_situations_hourly.py --max-per-source 200` o `--hours 8 14 22`.

### Pipeline V1 — anclado (CONGELADO)
`pipeline_v1_anchored/` — copia autocontenida del pipeline anclado original
(commit 748d6ae): taxonomia VIEJA (49 senales + `remind`), situaciones ancladas a
una senal y training con balanceo/forzado de labels. Genero los datos de ~5k que
se explican en la memoria del TFG. Trae sus propias copias de los modulos, asi
que corre aislado del codigo nuevo. Ver `pipeline_v1_anchored/ABOUT_THIS_FOLDER.md`.
No editar: es el registro historico V1.

---

## Mapa de ficheros

### Modulos compartidos (librerias, no se ejecutan solos)
| Fichero | Rol |
|---|---|
| `preference_taxonomy.py` | Taxonomia nueva: 11 senales, labels, SIGNAL_SEMANTICS, helpers. |
| `qwen_labeler.py` | `QwenDecisionLabeler` (Qwen local, JSON). |
| `coopera_profile_loader.py` | Carga mypersonality + traits. |
| `human_sim_preference_data.py` | **v1 base**: normalizers, `build_sample_from_stages`, `summarize`, `ProgressDisplay`, etc. |
| `human_sim_preference_data_v2_balanced.py` | **v2 base**: `routine_signature`, `register_routine`, etc. |

### Pipeline A (dataset-grounded libre)
`build_situations_from_external.py`, `translate_situations.py`,
`generate_training_data.py`, `run_pipeline.py`.

### Pipeline B (horario minimo)
`build_situations_hourly.py`, `generate_training_data_hourly.py`
(+ `translate_situations.py` compartido).

### Comun
`generate_profiles.py`.

### Herramientas
| Fichero | Rol |
|---|---|
| `study_situations.py` | EDA de un JSONL de situaciones (distribuciones, redundancia, diagnostico, plots). |
| `demo_situations_pipeline.py` | Muestra ejemplos RAW -> situacion -> accion de robot. |

### Legacy COOPERA (planes de Habitat; no usado por los pipelines actuales)
`build_training_data_from_coopera.py`, `coopera_plan_parser.py`,
`run_controlled_pipeline.py`, `example_pipeline_config.json`.

### Carpetas
| Carpeta | Contenido |
|---|---|
| `external_datasets/raw/` | Anotaciones crudas (Charades, EPIC) — solo texto, sin video. |
| `generated_data/` | Salidas (perfiles, situaciones, JSONL, study). **Gitignored.** |
| `pipeline_v1_anchored/` | Pipeline V1 anclado congelado (ver arriba). |

---

## Datasets fuente
- **Charades** (por defecto): Sigurdsson et al., ECCV 2016. Actividades
  domesticas en toda la casa. `arXiv:1604.01753`.
- **EPIC-Kitchens-100** (opcional, `--sources epic charades`): Damen et al.,
  IJCV 2022. Cocina egocentrica. `arXiv:2006.13256`.

## Pruebas sin GPU
Todos los scripts aceptan `--dry-run` (muestran el plan sin cargar Qwen). El
build y el study corren en CPU.
