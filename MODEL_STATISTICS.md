# Model Statistics & OpenAI API Coverage

Este documento proporciona estadísticas sobre los modelos actualmente disponibles en LiteLLM Companion y la cobertura de la API de OpenAI.

> **Nota**: Las estadísticas reflejan el estado actual del sistema y se actualizan automáticamente al sincronizar modelos desde los proveedores configurados.

---

## Estadísticas de Modelos

### Resumen General

- **Total de modelos**: 85
- **Proveedores activos**: OpenAI-compatible (100%)

### Por Tipo de Modelo

| Tipo | Cantidad | Descripción |
|------|----------|-------------|
| **Chat/Completion** | 71 | Modelos de conversación y generación de texto |
| **Embedding** | 9 | Modelos para generar embeddings vectoriales |
| **Audio (TTS/STT)** | 4 | Modelos de síntesis y transcripción de audio |
| **Image Generation** | 1 | Modelos de generación de imágenes |

### Por Capacidad

| Capacidad | Modelos | Descripción |
|-----------|---------|-------------|
| **Vision (Multimodal)** | 19 | Modelos que soportan entrada de imágenes |
| **Function Calling** | 34 | Modelos con soporte para llamadas a funciones/herramientas |
| **Audio Input** | 2 | Modelos que procesan entrada de audio (transcripción) |
| **Assistant Prefill** | 0 | Modelos con soporte para pre-rellenado de respuestas |
| **Prompt Caching** | 0 | Modelos con caché de prompts |
| **Response Schema** | 0 | Modelos con soporte para esquemas de respuesta estructurados |

### Modos de Ollama

Para modelos de Ollama, se soportan los siguientes modos:

| Modo | Cantidad | Descripción |
|------|----------|-------------|
| **openai** | 51 | Formato compatible con OpenAI |
| **chat** | 2 | Formato nativo de chat |
| **embedding** | 1 | Formato de embeddings |
| **audio_speech** | 1 | Síntesis de audio |
| **audio_transcription** | 1 | Transcripción de audio |

---

## Cobertura de la API de OpenAI

LiteLLM Companion proporciona compatibilidad con múltiples endpoints de la API de OpenAI, permitiendo usar proveedores locales (Ollama, LocalAI, etc.) como si fueran OpenAI.

### Endpoints Soportados

| Endpoint | Estado | Modelos | Descripción |
|----------|--------|---------|-------------|
| `/v1/chat/completions` | ✓ | 71 | Chat y generación de texto |
| `/v1/embeddings` | ✓ | 9 | Generación de embeddings |
| `/v1/audio/transcriptions` | ✓ | 2 | Transcripción de audio (Whisper) |
| `/v1/images/generations` | ⚠️ | 1 | Generación de imágenes (soporte limitado) |
| `/v1/audio/speech` | ✗ | 0 | Síntesis de voz (TTS) - no disponible actualmente |

**Leyenda**:
- ✓ = Totalmente soportado
- ⚠️ = Soporte limitado
- ✗ = No disponible actualmente

### Funciones Avanzadas

#### Vision (Multimodal)
**19 modelos** con capacidad de procesar imágenes además de texto.

**Ejemplos**:
- `llava-v1.6-7b-mmproj-f16.gguf`
- `llava-v1.6-mistral-7b.Q5_K_M.gguf`
- `minicpm-v-2_6-mmproj-f16.gguf`
- `moondream2-20250414`
- `moondream2-mmproj-f16-20250414.gguf`
- ... y 14 más

#### Function Calling (Llamadas a Funciones)
**34 modelos** con soporte para herramientas y llamadas a funciones.

**Ejemplos**:
- `LocalAI-functioncall-llama3.2-3b-v0.5`
- `localai-functioncall-phi-4-v0.3-q4_k_m.gguf`
- `devstral:24b`
- `ebdm/gemma3-enhanced:12b`
- `gpt-oss:20b`
- ... y 29 más

#### Embeddings
**9 modelos** para generar embeddings vectoriales.

**Ejemplos**:
- `bert-embeddings`
- `granite-embedding-107m-multilingual`
- `text-embedding-ada-002`
- `bge-large:latest`
- `bge-m3:latest`
- ... y 4 más

#### Chat/Completion
**71 modelos** para conversación y generación de texto.

**Ejemplos**:
- `dolphin3.0-qwen2.5-3b`
- `gemma-3-4b-it-qat`
- `hermes-2-pro-llama-3-8b`
- `llama3.3:latest`
- `qwen2.5:14b`
- ... y 66 más

### Parámetros de OpenAI Soportados

LiteLLM Companion soporta **30 parámetros únicos** de la API de OpenAI, incluyendo los más comunes:

#### Parámetros Principales
- `temperature` - Control de aleatoriedad en las respuestas
- `max_tokens` - Límite de tokens en la respuesta
- `top_p` - Muestreo nucleus
- `stream` - Streaming de respuestas
- `stop` - Secuencias de parada
- `frequency_penalty` - Penalización por frecuencia
- `presence_penalty` - Penalización por presencia

#### Parámetros Avanzados
- `tools` - Definición de herramientas/funciones
- `tool_choice` - Control de selección de herramientas
- `response_format` - Formato de respuesta (JSON, etc.)
- `seed` - Semilla para reproducibilidad
- `logprobs` - Probabilidades logarítmicas
- `logit_bias` - Sesgo de logits

... y **18 parámetros adicionales** para control fino del comportamiento del modelo.

---

## Distribución por Proveedor

Actualmente, todos los modelos están disponibles a través de proveedores compatibles con OpenAI:

- **OpenAI-compatible**: 85 modelos (100%)
  - LocalAI
  - Ollama (en modo OpenAI)
  - Otros proveedores compatibles

---

## Casos de Uso por Tipo de Modelo

### 1. Modelos de Chat (71 modelos)
**Casos de uso**:
- Asistentes conversacionales
- Generación de contenido
- Análisis y resumen de texto
- Traducción y reescritura
- Respuestas a preguntas

### 2. Modelos Vision (19 modelos)
**Casos de uso**:
- Análisis de imágenes
- Descripción de contenido visual
- OCR (reconocimiento de texto en imágenes)
- Clasificación de imágenes
- Q&A sobre imágenes

### 3. Modelos con Function Calling (34 modelos)
**Casos de uso**:
- Integración con APIs externas
- Automatización de tareas
- Extracción estructurada de datos
- Agentes inteligentes
- Workflows complejos

### 4. Modelos de Embedding (9 modelos)
**Casos de uso**:
- Búsqueda semántica
- Clustering de documentos
- Recomendaciones
- Clasificación de texto
- Detección de similitud

### 5. Modelos de Audio (4 modelos)
**Casos de uso**:
- Transcripción de audio a texto
- Subtitulación automática
- Análisis de llamadas
- Accesibilidad

---

## Actualizando las Estadísticas

Para actualizar estas estadísticas con los modelos actuales:

```bash
# 1. Sincronizar modelos desde los proveedores
curl -X POST http://localhost:8000/api/providers/sync-all

# 2. Obtener estadísticas actualizadas
curl -s http://localhost:4000/model/info -H "Authorization: Bearer sk-1234" | \
  python3 -c "import json, sys; data = json.load(sys.stdin); print(f'Total: {len(data[\"data\"])} models')"
```

O desde la interfaz web:
1. Ir a `/sources`
2. Hacer clic en "Sync All Providers"
3. Verificar en `/litellm` los modelos sincronizados

---

## Compatibilidad con Clientes OpenAI

Gracias a la compatibilidad con la API de OpenAI, puedes usar LiteLLM Companion con cualquier cliente que soporte OpenAI:

### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-1234"
)

# Chat
response = client.chat.completions.create(
    model="qwen2.5:14b",
    messages=[{"role": "user", "content": "Hola!"}]
)

# Embeddings
embedding = client.embeddings.create(
    model="bge-m3:latest",
    input="texto para embeddings"
)
```

### JavaScript/TypeScript
```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
    baseURL: 'http://localhost:4000',
    apiKey: 'sk-1234'
});

const completion = await openai.chat.completions.create({
    model: 'qwen2.5:14b',
    messages: [{ role: 'user', content: 'Hola!' }]
});
```

### cURL
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:14b",
    "messages": [{"role": "user", "content": "Hola!"}]
  }'
```

---

## Referencias

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [MIGRATION.md](MIGRATION.md) - Guía de migración
- [README.md](README.md) - Documentación general

---

**Última actualización**: 2025-12-26

**Estadísticas generadas desde**: LiteLLM Proxy en `http://localhost:4000`
