# LocalScribe

Pipeline local/offline para convertir grabaciones de reuniones en transcripciones estructuradas con diarización, nombres de participantes y metadatos de sesión.

## Estado

El desarrollo se gestiona en Linear, proyecto `LocalScribe`. El repositorio usa PRs pequeñas y trazables por tarjeta.

## Bootstrap

```bash
cp .env.example .env
mkdir -p data/{input,processing,output,completed,failed} models
docker compose build
docker compose up -d localscribe
```

El contenedor crea y usa estas rutas persistentes:

- `data/input`
- `data/processing`
- `data/output`
- `data/completed`
- `data/failed`
- `models`

Los audios, modelos y secretos no deben versionarse.

## Arquitectura prevista

1. watcher de archivos;
2. transcripción con Faster-Whisper;
3. diarización con pyannote;
4. resolución de participantes;
5. extracción de metadatos;
6. generación de Markdown, JSON y SRT;
7. reconocimiento opcional por perfil de voz local.
