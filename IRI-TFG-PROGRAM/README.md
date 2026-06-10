# IRI-TFG-PROGRAM

Generacion de datos sinteticos de entrenamiento para un modelo de personalizacion
de decisiones de un robot asistencial: dado un humano (perfil de personalidad) y
una situacion domestica con una accion propuesta del robot, predecir que querria
la persona que hiciera el robot.

- **Labels de decision**: `do_now`, `do_later`, `tell_the_user`, `no_action`.
- **Taxonomia de preferencias**: 11 senales jerarquicas (con `prefer`/`avoid` y
  `weight` 1-10) en `src/iri_tfg_program/taxonomy/preference_taxonomy.py`.
- **LLM**: Qwen local (`qwen_labeler.py`, GPU; torch lazy). En local sin CUDA se
  prueba con `--dry-run` o stubs.

**Todos los comandos ejecutables estan en [`documents/COMMANDS.md`](documents/COMMANDS.md).**

---

## Estructura

```
IRI-TFG-PROGRAM/
  src/iri_tfg_program/        # Paquete Python principal
    taxonomy/                   # preference_taxonomy.py
    profiles/                   # coopera_profile_loader, generate_profiles
    labeling/                   # qwen_labeler
    simulation/                 # human_sim_preference_data v1 + v2
    situations/                 # build_from_external, build_hourly, translate
    training_data/              # generate_training_data (A + B)
    analysis/                   # study_situations, demo_situations_pipeline

  scripts/                    # Wrappers ejecutables (apuntan al paquete)
    run_pipeline.py             # Orchestrador Pipeline A completo
    generate_profiles.py
    build_situations_from_external.py
    build_situations_hourly.py
    translate_situations.py
    generate_training_data.py
    generate_training_data_hourly.py
    study_situations.py
    demo_situations_pipeline.py

  documents/                  # Documentacion
    COMMANDS.md                 # << TODOS los comandos
  data/                       # Datos locales (gitignored)
    external/                   # external_datasets/raw/ (Charades, EPIC)
    generated/                  # Salidas de los pipelines
  archive/                    # Legacy congelado (gitignored)
    pipeline_v1_anchored/       # Pipeline V1 (datos 5k del TFG)
    legacy_root_scripts/        # Scripts COOPERA no usados
  configs/
    example_pipeline_config.json
```

### PYTHONPATH

Los scripts en `scripts/` inyectan `src/` en `sys.path` automaticamente.
Para usar el paquete directamente:

```bash
export PYTHONPATH="$(pwd)/src"   # Linux / macOS / Git Bash
$env:PYTHONPATH = "$(Get-Location)\src"  # PowerShell
```

---

## Pipelines

### Pipeline A — dataset-grounded libre (principal)

Situaciones reales de Charades (EPIC opcional), traducidas a accion de robot,
y decision LIBRE (sin forzar labels, sin anchoring).

```
build_situations_from_external  (CPU)  -> situaciones (accion humana)
translate_situations            (GPU)  -> accion de robot
generate_training_data          (GPU)  -> JSONL de entrenamiento
```

Orchestrador: `scripts/run_pipeline.py` (`--skip-*`, `--sample N`, `--dry-run`).

### Pipeline B — horario minimo

Cada situacion se expande a 24 copias (una por hora) con estructura minima.

```
build_situations_hourly         (CPU)  -> situaciones minimas x24h
translate_situations            (GPU)  -> accion de robot
generate_training_data_hourly   (GPU)  -> JSONL
```

### Pipeline V1 — anclado (CONGELADO)

`archive/pipeline_v1_anchored/` — copia autocontenida del pipeline V1 original
(taxonomia antigua de 49 senales). Ver `ABOUT_THIS_FOLDER.md` en esa carpeta.
No editar: es el registro historico que genero los datos 5k del TFG.

---

## Datasets fuente

- **Charades** (por defecto): Sigurdsson et al., ECCV 2016. Actividades
  domesticas. `arXiv:1604.01753`.
- **EPIC-Kitchens-100** (opcional): Damen et al., IJCV 2022. Cocina
  egocentrica. `arXiv:2006.13256`.

## Pruebas sin GPU

Todos los scripts aceptan `--dry-run`. El build y el study corren en CPU.
Ver `documents/COMMANDS.md` para smoke tests concretos.
