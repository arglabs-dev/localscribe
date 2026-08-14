# LocalScribe

Pipeline local/offline para convertir grabaciones de reuniones en transcripciones estructuradas con diarización, nombres de participantes y metadatos de sesión.

## Estado

El desarrollo se gestiona en Linear, proyecto `LocalScribe`. El repositorio usa PRs pequeñas y trazables por tarjeta.

## Bootstrap

```bash
cp .env.example .env
mkdir -p data/{input,processing,output,completed,failed,enrollment,profiles} models
docker compose build
```

La primera vez, con internet disponible, acepta las condiciones de `pyannote/speaker-diarization-community-1` y `pyannote/embedding`. Agrega temporalmente `HF_TOKEN` a `.env` y ejecuta:

```bash
docker compose --profile setup run --rm bootstrap-models
```

Al terminar puedes quitar `HF_TOKEN`. Los modelos quedan persistidos bajo `./models`, `PYANNOTE_METRICS_ENABLED=0` desactiva la telemetría y la ejecución normal usa `ALLOW_MODEL_DOWNLOAD=0`:

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

## Perfiles de voz locales

Puedes registrar una voz una sola vez y reutilizarla en reuniones futuras. Coloca temporalmente una grabación limpia de una sola persona en `data/enrollment/` y ejecuta:

```bash
docker compose run --rm localscribe \
  python -m app.voice_profiles enroll \
  --name "Armando Reyes" \
  --audio /app/data/enrollment/armando-reyes.wav
```

Registrar de nuevo el mismo nombre reemplaza su embedding. LocalScribe guarda únicamente el embedding normalizado y metadatos del perfil en `data/profiles/profiles.json`; no copia ni conserva por su cuenta el audio de enrollment.

Lista y elimina perfiles con:

```bash
docker compose run --rm localscribe python -m app.voice_profiles list
docker compose run --rm localscribe python -m app.voice_profiles delete --name "Armando Reyes"
```

Durante una reunión, LocalScribe compara fragmentos diarizados con los perfiles registrados. Solo asigna una identidad si cumple el umbral máximo de distancia coseno y, cuando existen varios candidatos, un margen mínimo respecto al segundo candidato. Ambos valores son configurables. Si la coincidencia no es suficientemente clara, no fuerza el nombre y usa como fallback la auto-presentación o `SPEAKER_XX`.

La resolución queda trazable como `voice_profile`, `self_introduction` o `unresolved` en los artefactos estructurados. Los perfiles de voz sirven para etiquetado de reuniones, no para autenticación ni identificación forense.

## Cabecera recomendada de una sesión

Aunque existan perfiles de voz, una cabecera explícita sigue siendo útil como fallback y para metadatos:

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
- Markdown/SRT muestran nombres cuando un perfil o auto-presentación resuelve la identidad;
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
- `data/enrollment`
- `data/profiles`
- `models`

Los audios, modelos, perfiles y secretos no deben versionarse.

## Arquitectura

1. watcher de archivos;
2. transcripción con Faster-Whisper;
3. diarización con pyannote;
4. reconocimiento opcional por perfil de voz local;
5. fallback de identidad por auto-presentación;
6. extracción de metadatos;
7. generación de Markdown, JSON y SRT.
