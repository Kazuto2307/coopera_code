# pipeline_v1_anchored — FROZEN (pipeline V1, anclado)

Copia **congelada** del pipeline de 3 pasos *anclado* tal y como estaba en el
commit `748d6ae` ("pipeline created"). Es el que genero los datos de ~5k que se
explican en la memoria del TFG. Se conserva aqui, autocontenido, para que siga
siendo reproducible aunque el pipeline nuevo (en el directorio padre) evolucione.

## Por que esta aislado
Usa la **taxonomia VIEJA** (50 senales `prefer_/avoid_` + label `remind`), que ya
NO es la del proyecto actual (la nueva tiene 11 senales jerarquicas y
`tell_the_user`). Esta carpeta trae su PROPIA copia de `preference_taxonomy.py`,
`qwen_labeler.py`, `human_sim_preference_data.py` (V1), etc., asi que al
ejecutar desde aqui los imports resuelven a estas copias (sys.path[0] = esta
carpeta) y NO dependen del codigo nuevo del padre.

## Que es V1 y V2 aqui
- **V1**: `human_sim_preference_data.py` — generador monolitico base.
- **V2**: `human_sim_preference_data_v2_balanced.py` — variante con balanceo de
  label, routine consistency y reintentos.
- Encima, el pipeline de 3 pasos *anclado*:
  `generate_profiles.py` -> `generate_situations.py` (situaciones ancladas a una
  senal, con `differentiating_labels`) -> `generate_training_data.py` (label
  schedule + target + retry + matching por `anchored_signal`).

## Como ejecutarlo (ejemplos)
```bash
python IRI-TFG-PROGRAM/pipeline_v1_anchored/generate_profiles.py --num-profiles 10
python IRI-TFG-PROGRAM/pipeline_v1_anchored/generate_situations.py --all-signals
python IRI-TFG-PROGRAM/pipeline_v1_anchored/generate_training_data.py --target-samples 5000
# o el monolitico balanceado (V2) directamente:
python IRI-TFG-PROGRAM/pipeline_v1_anchored/human_sim_preference_data_v2_balanced.py --target-samples 5000
```

NO editar para evolucionar el proyecto: el desarrollo activo va en el directorio
padre (pipeline nuevo, sin anchoring). Esto es solo el registro V1.
