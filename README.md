# LocalScribe

Pipeline local/offline para convertir grabaciones de reuniones en transcripciones estructuradas con diarización, nombres de participantes y metadatos de sesión.

## Estado

El desarrollo se gestiona en Linear, proyecto `LocalScribe`. El repositorio usa PRs pequeñas y trazables por tarjeta.

## Bootstrap

```bash
cp .env.example .env
mkdir -p data/{input,processing,output,completed,failed} models
docker compose build
```

La primera vez, con internet disponible y después de aceptar las condiciones de `pyannote/speaker-diarization-community-1`, agrega temporalmente `HF_TOKEN` a `.env` y ejecuta:

```bash
docker compose --profile setup run --rm bootstrap-models
```

Al terminar puedes quitar `HF_TOKEN`. Los modelos quedan persistidos bajo `./models` y la ejecución normal debe hacerse con `ALLOW_MODEL_DOWNLOAD=0`:

```bash
docker compose up -d localscribe
docker compose logs -f localscribe
```

## Uso diario

1. Graba la reunión en WAV, MP3, M4A u otro formato configurado.
2. Copia el archivo a `data/input/`.
3. LocalScribe espera a que termine la copia y mueve el archivo a `data/processing/`.
4. Tras procesarlo, el audio termina en `data/completed/` o `data/failed/`.
5. Los artefactos quedan en un subdirectorio de `data/output/`:
   - `metadata.json`
   - `transcript.json`
   - `transcript.md`
   - `transcript.srt`

Si un job falla, el subdirectorio de salida incluye `ERROR.txt` con el error y traceback.

## Cabecera recomendada de una sesión

Para el MVP, habla de forma natural pero explícita durante los primeros minutos. Por ejemplo:

```text
Hoy es jueves 13 de agosto, son las 5:30 de la tarde.
Sesión de despacho jurídico de Justa Ley.
Yo soy Armando Reyes.
Yo soy Jimena Hernández.
```

Cada auto-presentación debe pronunciarla la persona correspondiente. El sistema conserva siempre el `speaker_id` original de diarización y registra cómo se resolvió cada nombre.

## Smoke test end-to-end real

Después del bootstrap de modelos:

```bash
docker compose up -d localscribe
cp /ruta/a/reunion-prueba.m4a data/input/
```

Verifica que:

- el audio desaparece de `input` y termina en `completed`;
- `processing` queda vacío al finalizar;
- se crean los cuatro artefactos esperados en `output`;
- `metadata.json` contiene fecha/hora, fuente del datetime y participantes;
- `transcript.json` conserva `whisper_segments` raw y `speaker_id`;
- Markdown/SRT muestran nombres cuando las auto-presentaciones fueron claras;
- con la red desconectada después del bootstrap, una segunda grabación sigue procesándose sin descargar modelos.

Este smoke test requiere modelos y un audio real; no se ejecuta en CI.

## Regresión automatizada

La suite ligera prueba la lógica determinística posterior a los modelos sin descargar pesos de Whisper/pyannote:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app scripts
python -m pytest -q
```

GitHub Actions ejecuta esta suite en cada PR y en cada push a `main`.

## Datos persistentes

El contenedor crea y usa estas rutas:

- `data/input`
- `data/processing`
- `data/output`
- `data/completed`
- `data/failed`
- `models`

Los audios, modelos y secretos no deben versionarse.

## Arquitectura

1. watcher de archivos;
2. transcripción con Faster-Whisper;
3. diarización con pyannote;
4. resolución de participantes;
5. extracción de metadatos;
6. generación de Markdown, JSON y SRT;
7. reconocimiento opcional por perfil de voz local.
