# IRI-TFG-PROGRAM

Modulo aislado para generar datos sinteticos de entrenamiento del repositorio
de preferencias a partir de planes humanos generados por COOPERA.

La separacion es esta:

- COOPERA genera perfiles humanos, Big Five, resumen de personalidad,
  intenciones y tareas domesticas.
- Este modulo parsea los planes finales de COOPERA.
- Qwen local genera `preference_snapshot` y `label_action` usando el perfil
  humano de COOPERA, no reglas superficiales del contexto.
- Los JSONL sinteticos se guardan en `generated_data/`.

No ejecuta Habitat ni renderiza videos.

## Entrada COOPERA

Lee planes finales como:

```text
results/human/gpt_response/collaboration_1/predicates_reflection_2/<human>/<scene>/<day>/<hour>/predicates_reflection_2.json
```

Y carga contexto de personalidad desde:

```text
results/human/gpt_response/traits_summary/<human>/<timestamp>/traits_summary.json
data/humanoids/humanoid_data/mypersonality_final.csv
```

El nombre `gpt_response` es historico en COOPERA. En este fork puede estar
respaldado por Qwen local.

## Salida

Genera muestras con el contrato del repositorio de preferencias:

```json
{
  "sample_id": "coopera:00003:108736635_177263256:00:...:task1",
  "user_id": 4,
  "user_external_id": "coopera_human_00003",
  "label_action": "do_now",
  "action_input": {
    "action_text": "place 008_pudding_box_:0000 on Food, White Pedestal Bowl",
    "activity": "meal support"
  },
  "context_input": {
    "location_current": "kitchen",
    "objects_nearby": ["Food, White Pedestal Bowl", "008_pudding_box_:0000"],
    "available_objects": ["008_pudding_box_:0000", "Food, White Pedestal Bowl"],
    "raw_conditions": ["cozy", "user nearby"],
    "time_of_day": "morning",
    "weekday": "synthetic_day_00",
    "user_state": ["nearby"],
    "environment_flags": []
  },
  "structured_task_features": {
    "kind": "routine_reminder",
    "urgency": "medium",
    "sensitivity": "low",
    "user_busy": false,
    "quiet_hours": false,
    "conditions": ["user nearby"],
    "context_flags": {}
  },
  "preference_snapshot": [
    {"signal_name": "prefer_non_intrusive_assistance", "polarity": "prefer"}
  ]
}
```

Importante: `preference_snapshot` y `label_action` salen del modo
`qwen_profile`, que recibe el perfil, Big Five/resumen e intencion/tarea. No se
deben inferir por reglas como "desayuno -> prefer_morning_tasks".

## Uso recomendado con Qwen perfil

Desde la raiz de COOPERA:

```bash
python IRI-TFG-PROGRAM/build_training_data_from_coopera.py \
  --response-source gpt_response \
  --collab-type 1 \
  --generation-strategy qwen_profile \
  --qwen-model Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --output IRI-TFG-PROGRAM/generated_data/training_samples_from_coopera_qwen_profile.jsonl \
  --summary-output IRI-TFG-PROGRAM/generated_data/training_samples_from_coopera_qwen_profile_summary.json
```

## Pipeline conectado y controlable

Para lanzar `human_sim.py` y despues construir el JSONL en una sola orden:

```bash
python IRI-TFG-PROGRAM/run_controlled_pipeline.py \
  --scene-indices 1 \
  --profile-indices 3 \
  --max-days 1 \
  --collab-type 1 \
  --generation-strategy qwen_profile \
  --output IRI-TFG-PROGRAM/generated_data/training_samples_scene1_human3_day0.jsonl
```

Esto ejecuta dos etapas:

1. `habitat-lab/coopera_main/human_sim/human_sim.py`, que genera
   `traits_summary` y `predicates_reflection_2`.
2. `IRI-TFG-PROGRAM/build_training_data_from_coopera.py`, que convierte esos
   planes en JSONL para el modelo de preferencias.

Para solo ver los comandos sin ejecutar Habitat/Qwen:

```bash
python IRI-TFG-PROGRAM/run_controlled_pipeline.py \
  --scene-indices 1 \
  --profile-indices 3 \
  --max-days 1 \
  --dry-run
```

Para reutilizar planes ya generados y reconstruir solo el dataset:

```bash
python IRI-TFG-PROGRAM/run_controlled_pipeline.py \
  --scene-indices 1 \
  --profile-indices 3 \
  --max-days 1 \
  --skip-human-sim \
  --generation-strategy qwen_profile
```

El pipeline guarda un `*.manifest.json` junto al JSONL con los comandos usados.

## Human Sim Alternativo Para Tu Dataset

Si no quieres generar primero `predicates_reflection_2` para el simulador, usa:

```bash
python IRI-TFG-PROGRAM/human_sim_preference_data.py \
  --profile-indices 3 \
  --max-days 1 \
  --samples-per-hour 1 \
  --output IRI-TFG-PROGRAM/generated_data/preference_human_sim_human3_day0.jsonl
```

El script busca `mypersonality_final.csv` en estas rutas:

```text
data/humanoids/humanoid_data/mypersonality_final.csv
habitat-lab/data/versioned_data/habitat_humanoids/mypersonality_final.csv
```

Si esta en otra ruta:

```bash
python IRI-TFG-PROGRAM/human_sim_preference_data.py \
  --mypersonality-path /ruta/a/mypersonality_final.csv \
  --profile-indices 3 \
  --max-days 1
```

Este script es el equivalente conceptual de `human_sim.py`, pero su salida no es
para Habitat. En vez de generar planes con `Act: [...]`, genera directamente
muestras del modelo de preferencias:

```text
COOPERA mypersonality + Big Five + traits_summary opcional
        |
        v
profile_summary
        |
        v
preference_profile estable
        |
        v
assistance_situation por hora
        |
        v
decision_reflection
        |
        v
action_input + context_input + structured_task_features
        |
        v
preference_snapshot + label_action
        |
        v
JSONL listo para entrenamiento
```

Para comprobar bucles sin cargar Qwen:

```bash
python IRI-TFG-PROGRAM/human_sim_preference_data.py \
  --profile-indices 3 \
  --max-days 1 \
  --dry-run
```

Este es el camino mas directo si tu objetivo es dataset, no simulacion fisica.

La estructura imita COOPERA: no se pide a Qwen que genere todo de golpe. Primero
compacta la persona, despues infiere preferencias estables, despues propone una
situacion, y finalmente reflexiona la decision del robot. Los intermedios se
guardan en:

```text
IRI-TFG-PROGRAM/generated_data/<run>_intermediate/<human_id>/
  profile_summary.json
  preference_profile.json
```

Para una prueba pequena en servidor:

```bash
python IRI-TFG-PROGRAM/build_training_data_from_coopera.py \
  --response-source gpt_response \
  --collab-type 1 \
  --generation-strategy qwen_profile \
  --human-ids 3 \
  --days 0 \
  --max-files 2 \
  --output IRI-TFG-PROGRAM/generated_data/smoke_qwen_profile.jsonl
```

Si falta el perfil de COOPERA, el script falla por defecto. Eso es intencional:
evita generar preferencias sin personalidad humana. Solo para depuracion puedes
relajar esto con:

```bash
--allow-missing-profile
```

Si el modelo requiere token:

```bash
export HUGGINGFACE_TOKEN=...
```

## Modo de depuracion sin Qwen

Existe `rules_debug` solo para comprobar parser/rutas sin cargar GPU:

```bash
python IRI-TFG-PROGRAM/build_training_data_from_coopera.py \
  --response-source gpt_response \
  --collab-type 1 \
  --generation-strategy rules_debug \
  --max-files 2 \
  --output IRI-TFG-PROGRAM/generated_data/debug_rules.jsonl
```

No usar `rules_debug` como dataset final.

## Entrenar en el repo de preferencias

```bash
python -m src.scripts.training.train_decision_personalization_model \
  --input path/to/training_samples_from_coopera_qwen_profile.jsonl \
  --model-output artifacts/decision_personalization/model_coopera.pt \
  --report-output artifacts/decision_personalization/report_coopera.json
```

## Archivos

- `coopera_plan_parser.py`: parser compatible con `predicates_reflection_2.json`.
- `coopera_profile_loader.py`: carga `traits_summary` y Big Five/perfil original.
- `build_training_data_from_coopera.py`: genera JSONL de entrenamiento.
- `human_sim_preference_data.py`: genera JSONL directamente desde perfiles humanos.
- `qwen_labeler.py`: genera preferencias y target con Qwen local, sin OpenAI.
- `preference_taxonomy.py`: senales cerradas esperadas por el RAG entrenable.
- `generated_data/`: salida sintetica.
