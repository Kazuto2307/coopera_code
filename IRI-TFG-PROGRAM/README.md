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

Durante una ejecucion real, el script muestra un resumen visual del plan y
barras de progreso para perfiles y muestras si `tqdm` esta instalado. Si no lo
esta, usa mensajes simples de progreso. Para desactivar las barras:

```bash
python IRI-TFG-PROGRAM/human_sim_preference_data.py \
  --max-profiles 20 \
  --max-days 5 \
  --samples-per-hour 1 \
  --no-progress
```

Si quieres conservar tambien el plan completo en JSON por terminal:

```bash
--plan-json
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

## Pipeline de datos (situaciones del dataset, sin anchoring)

El flujo NO inventa situaciones ni las ancla a ninguna preferencia: las
situaciones salen de datasets publicos de actividades domesticas (EPIC-Kitchens
+ Charades), se traducen a acciones de robot, y el humano sintetico decide
libremente que querria que hiciera el robot, sin forzar ninguna etiqueta.

1. `generate_profiles.py`              -> humanos sinteticos (perfil + preferencias + description).
2. `build_situations_from_external.py` -> parsea EPIC+Charades a situaciones crudas (CPU).
   `translate_situations.py`           -> traduce cada accion humana a accion de robot (Qwen/GPU).
3. `generate_training_data.py`         -> decision libre: el humano elige label sin forzar nada (Qwen/GPU).

Etiquetas (taxonomia nueva): `do_now`, `do_later`, `tell_the_user`, `no_action`.
Todos aceptan `--dry-run` (plan sin cargar Qwen) y `--no-progress`.

```text
generate_profiles.py        build_situations_from_external.py
  (humanos sinteticos)        (EPIC + Charades -> situaciones)
        |                             |
        |                     translate_situations.py
        |                       (accion humana -> accion de robot)
        +-------------+---------------+
                      v
            generate_training_data.py
          (decision libre, sin forzar labels)
                      |
                      v
        JSONL de entrenamiento (distribucion natural)
```

`run_pipeline.py` encadena los cuatro pasos en una sola orden.

### Paso 1: perfiles

Lee `mypersonality_final.csv` y, si existe, el `traits_summary` de COOPERA.
Para cada perfil genera con Qwen un `profile_summary`, un `preference_profile`
estable y una `description` legible de una frase. Guarda un JSON por humano.

```bash
python IRI-TFG-PROGRAM/generate_profiles.py \
  --num-profiles 10 \
  --output-dir IRI-TFG-PROGRAM/generated_data/profiles/
```

O con indices explicitos:

```bash
python IRI-TFG-PROGRAM/generate_profiles.py \
  --profile-indices 0 3 7 \
  --output-dir IRI-TFG-PROGRAM/generated_data/profiles/
```

Cada humano se persiste como `generated_data/profiles/human_XXXXX.json`:

```json
{
  "human_id": "00003",
  "profile_index": 3,
  "big_five": {"openness": 3.1, "conscientiousness": 3.4, "extroversion": 2.6, "agreeableness": 3.8, "neuroticism": 2.9},
  "profile_summary": {"summary": {}, "source": "generated_from_mypersonality"},
  "preference_profile": {
    "stable_preferences": [{"signal_name": "interruption_sensitivity", "polarity": "prefer"}],
    "profile_level_rationale": "...",
    "uncertain_or_omitted": []
  },
  "description": "Una frase legible que describe al humano sintetico.",
  "generated_at": "2026-06-01T13:46:29+02:00"
}
```

Sin `--overwrite`, los `human_XXXXX.json` ya existentes se saltan y se loguean.
`--num-profiles` y `--profile-indices` son mutuamente excluyentes. Si la
generacion de la `description` falla, queda `null` sin abortar el perfil.

### Paso 2: situaciones (del dataset, traducidas a acciones de robot)

Las situaciones NO se inventan ni se anclan: se construyen de datasets publicos
de actividades de la vida diaria y luego se traducen a acciones de robot.

`build_situations_from_external.py` parsea solo las anotaciones de texto (sin
video) de EPIC-Kitchens-100 (cocina, ~77k acciones verbo+objeto) y Charades
(~157 actividades domesticas en todas las habitaciones), y emite situaciones tal
cual, sin anclaje a ninguna preferencia:

```bash
python IRI-TFG-PROGRAM/build_situations_from_external.py \
  --sources epic charades \
  --output IRI-TFG-PROGRAM/generated_data/situations_external/external_situations.jsonl
```

(Las anotaciones crudas se descargan a `IRI-TFG-PROGRAM/external_datasets/raw/`.)

Despues, `translate_situations.py` reescribe cada `action_text` (accion humana)
como accion de robot con Qwen en GPU. Lo que el robot puede hacer se mantiene
("make breakfast" -> "prepare breakfast") y lo que solo hace la persona se
reformula como asistencia ("consume pills" -> "offer the pills"). Traduce solo
frases unicas y cachea el mapa (`--reuse-map` para reanudar):

```bash
python IRI-TFG-PROGRAM/translate_situations.py \
  --input  IRI-TFG-PROGRAM/generated_data/situations_external/external_situations.jsonl \
  --output IRI-TFG-PROGRAM/generated_data/situations_external/external_situations_robot.jsonl
```

Cada situacion es una linea JSONL (sin anchoring, sin campos de usuario):

```json
{
  "situation_id": "ext:charades:46GP8:c129",
  "source_dataset": "charades",
  "action_input": {"action_text": "offer the pills", "activity": "medication support"},
  "context_input": {"location_current": "bedroom", "objects_nearby": [], "time_of_day": "unknown", "weekday": "unknown", "user_state": []},
  "structured_task_features": {"kind": "medication_support", "urgency": "medium", "sensitivity": "high"},
  "scenario_rationale": "...",
  "source_metadata": {"original_action_text": "taking/consuming some medicine", "source_dataset": "charades"}
}
```

### Paso 3: datos de entrenamiento (decision libre)

Carga los perfiles del paso 1 y las situaciones traducidas del paso 2. Para cada
par (humano, situacion), Qwen decide de forma natural que querria esa persona
que hiciera el robot. NO hay label schedule, ni target label, ni reintentos para
forzar una etiqueta, ni anclaje de preferencias: la distribucion de labels es la
que emerja. Usa la taxonomia nueva (`tell_the_user` y las 11 senales jerarquicas
con su semantica `prefer`/`avoid` inyectada en el prompt).

```bash
python IRI-TFG-PROGRAM/generate_training_data.py \
  --profiles-dir IRI-TFG-PROGRAM/generated_data/profiles/ \
  --situations   IRI-TFG-PROGRAM/generated_data/situations_external/external_situations_robot.jsonl \
  --target-samples 3000 \
  --output IRI-TFG-PROGRAM/generated_data/preference_training.jsonl \
  --summary-output IRI-TFG-PROGRAM/generated_data/preference_training_summary.json
```

El `sample_id` es `synth:<human_id>:<situation_id>:<n>` y el `source_metadata`
incluye `human_id`, `situation_id`, `source_dataset`, `original_action_text` y
`decision_rationale`. `--routine-consistency enforce` (opcional, por defecto
`off`) descarta contradicciones (misma situacion para el mismo humano con label
distinto) sin forzar ninguna etiqueta. Con `--profile-indices` se usa un
subconjunto de perfiles.

### Todo en una sola orden

`run_pipeline.py` encadena perfiles -> build -> translate -> training:

```bash
python IRI-TFG-PROGRAM/run_pipeline.py \
  --num-profiles 25 \
  --target-samples 3000 \
  --output IRI-TFG-PROGRAM/generated_data/preference_training_3k.jsonl \
  --summary-output IRI-TFG-PROGRAM/generated_data/preference_training_3k_summary.json
```

Pasos saltables: `--skip-profiles`, `--skip-build`, `--skip-translate`,
`--skip-training`. Anade `--dry-run` para ver el plan sin cargar Qwen.

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
- `human_sim_preference_data.py`: genera JSONL directamente desde perfiles humanos (v1, funciones base).
- `human_sim_preference_data_v2_balanced.py`: variante v2 con balanceo de label, routine consistency y reintentos.
- `generate_profiles.py`: paso 1, genera humanos sinteticos (perfil + preferencias + description).
- `build_situations_from_external.py`: paso 2a, parsea EPIC-Kitchens + Charades a situaciones (CPU, sin anchoring).
- `translate_situations.py`: paso 2b, traduce las acciones humanas a acciones de robot con Qwen.
- `generate_training_data.py`: paso 3, decision libre del humano sintetico (sin forzar labels), taxonomia nueva.
- `run_pipeline.py`: encadena los cuatro pasos en una sola orden.
- `external_datasets/raw/`: anotaciones crudas descargadas (EPIC, Charades).
- `qwen_labeler.py`: genera preferencias y target con Qwen local, sin OpenAI.
- `preference_taxonomy.py`: senales cerradas esperadas por el RAG entrenable.
- `generated_data/`: salida sintetica.
