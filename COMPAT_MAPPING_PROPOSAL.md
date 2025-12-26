# Propuesta de Mapeo Mejorado para Modelos Compat

## Análisis de Modelos Disponibles

### Proveedores
- **mks** (Ollama): 81 modelos
- **localai** (OpenAI-compatible): 37 modelos

### Modelos por Características

#### Chat General (con razonamiento)
Los modelos Qwen3 TODOS tienen capacidad de razonamiento según la DB:
- `qwen3:4b` - 32K context, reasoning ✓
- `qwen3:8b` - 32K context, reasoning ✓
- `qwen3:14b` - 32K context, reasoning ✓
- `qwen3:30b` - 32K context, reasoning ✓
- `qwen3:32b` - 32K context, reasoning ✓

Otros modelos de chat:
- `llama3.2:3b` - 128K context
- `llama3.1:8b` - 128K context
- `llama3.3:70b` - 8K context
- `mistral:7b` - 32K context
- `mistral-small:22b` - 32K context

#### Modelos de Razonamiento Dedicados
- `deepseek-r1:7b` - reasoning ✓
- `deepseek-r1:8b` - reasoning ✓
- `deepseek-r1:14b` - reasoning ✓
- `deepseek-r1:32b` - reasoning ✓
- `gpt-oss:20b` - reasoning ✓
- `magistral:24b` - reasoning ✓ (mistral con razonamiento)
- `qwq:32b` - modelo de razonamiento especializado

#### Modelos Multimodales (Vision + Razonamiento)
**Los modelos Qwen3-VL tienen VISION + REASONING:**
- `qwen3-vl:4b` - 32K context, vision ✓, reasoning ✓
- `qwen3-vl:8b` - 32K context, vision ✓, reasoning ✓
- `qwen3-vl:30b` - 32K context, vision ✓, reasoning ✓
- `qwen3-vl:32b` - 32K context, vision ✓, reasoning ✓

Otros modelos de visión:
- `llama3.2-vision:11b` - 128K context, vision ✓
- `gemma3:12b` - 8K context, vision ✓
- `mistral-small3.1:24b` - 32K context, vision ✓
- `mistral-small3.2:24b` - 32K context, vision ✓
- `qwen2.5vl:7b` - 32K context, vision ✓

#### Código
- `qwen2.5-coder:1.5b` - 32K context
- `qwen2.5-coder:7b` - 32K context
- `qwen2.5-coder:14b` - 32K context
- `qwen3-coder:30b` - 32K context

#### Embeddings
- `qwen3-embedding:0.6b` - 32K context
- `qwen3-embedding:8b` - 32K context
- `nomic-embed-text`
- `bge-large`
- `bge-m3`
- `mxbai-embed-large`

---

## 🔥 PROPUESTA DE MAPEO MEJORADO

### 1. CHAT MODELS (Progressive Scale)

#### `gpt-3.5-turbo` → **`qwen3:4b`** (proveedor: `mks`)
**Cambio:** De `qwen3:4b` → sigue igual ✓
- Contexto: 32K (suficiente para GPT-3.5)
- Razonamiento: ✓
- Uso: Fast, general chat

#### `gpt-3.5-turbo-16k` → **`qwen3:8b`** (proveedor: `mks`)
**Cambio:** De `qwen3:4b` → **`qwen3:8b`** (más apropiado para 16K)
- Contexto: 32K
- Razonamiento: ✓
- Más capacidad que el modelo base

#### `gpt-4o-mini` → **`qwen3:8b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Contexto: 32K
- Razonamiento: ✓
- Balanced performance

#### `gpt-4` → **`qwen3:32b`** (proveedor: `mks`)
**Cambio:** De `gpt-oss:20b` → **`qwen3:32b`**
**Razón:** Qwen3:32b tiene razonamiento nativo Y más parámetros
- Contexto: 32K
- Razonamiento: ✓
- Premium quality

#### `gpt-4-32k` → **`llama3.1:8b`** (proveedor: `mks`)
**Cambio:** De `gpt-oss:20b` → **`llama3.1:8b`**
**Razón:** El nombre enfatiza contexto grande (128K > 32K)
- Contexto: 128K ✓✓✓
- Large context capability

#### `gpt-4o` → **`qwen3:32b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Contexto: 32K
- Razonamiento: ✓
- Premium model

#### **NUEVO:** `gpt-4-turbo` → **`llama3.3:70b`** (proveedor: `mks`)
**Razón:** El modelo más grande y potente disponible
- Contexto: 8K (suficiente para la mayoría de casos)
- Parámetros: 70B (máxima calidad)
- Best quality available

---

### 2. VISION MODELS

#### `gpt-4-vision-preview` → **`llama3.2-vision:11b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Contexto: 128K ✓✓
- Visión: ✓
- Estable y bien probado

#### `gpt-4-turbo-vision` → **`qwen3-vl:8b`** (proveedor: `mks`)
**Cambio:** De `qwen3-vl:8b` → sigue igual ✓
**VENTAJA ESPECIAL:** Vision + Reasoning!
- Contexto: 32K
- Visión: ✓
- Razonamiento: ✓ (único!)

#### `gpt-4o-vision` → **`qwen3-vl:32b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
**VENTAJA ESPECIAL:** Vision + Reasoning en modelo grande!
- Contexto: 32K
- Visión: ✓
- Razonamiento: ✓
- Premium quality

#### **NUEVO:** `gpt-4-vision-mini` → **`qwen3-vl:4b`** (proveedor: `mks`)
**Razón:** Opción rápida para visión
- Contexto: 32K
- Visión: ✓
- Razonamiento: ✓
- Fast processing

---

### 3. EMBEDDING MODELS

#### `text-embedding-3-small` → **`nomic-embed-text`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Lightweight, fast
- Bien optimizado

#### `text-embedding-ada-002` → **`nomic-embed-text`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Legacy compatibility

#### `text-embedding-3-large` → **`qwen3-embedding:8b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Contexto: 32K
- High quality
- Large dimensions (4096)

---

### 4. REASONING MODELS

#### `o1-mini` → **`deepseek-r1:7b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Razonamiento: ✓
- Fast, lightweight

#### `o1-preview` → **`deepseek-r1:14b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Razonamiento: ✓
- Balanced

#### `o1` → **`deepseek-r1:32b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Razonamiento: ✓
- Premium reasoning

#### **NUEVO:** `o3-mini` → **`deepseek-r1:8b`** (proveedor: `mks`)
**Razón:** Opción intermedia entre o1-mini y o1-preview
- Razonamiento: ✓
- Balanced performance

#### **NUEVO:** `o1-pro` → **`qwq:32b`** (proveedor: `mks`)
**Razón:** Modelo especializado en razonamiento complejo
- Razonamiento avanzado
- 32B parámetros

---

### 5. CODE MODELS

#### `code-davinci-002` → **`qwen2.5-coder:14b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Contexto: 32K
- Code specialized

#### `gpt-4-code` → **`qwen3-coder:30b`** (proveedor: `mks`)
**Cambio:** Sigue igual ✓
- Contexto: 32K
- Premium code model

#### **NUEVO:** `gpt-3.5-turbo-instruct` → **`qwen2.5-coder:7b`** (proveedor: `mks`)
**Razón:** Modelo rápido para code completion
- Contexto: 32K
- Fast code generation

---

## 📊 RESUMEN DE CAMBIOS

### Cambios Principales
1. **`gpt-4`**: `gpt-oss:20b` → **`qwen3:32b`** (mejor razonamiento, más parámetros)
2. **`gpt-4-32k`**: `gpt-oss:20b` → **`llama3.1:8b`** (128K contexto)
3. **`gpt-3.5-turbo-16k`**: `qwen3:4b` → **`qwen3:8b`** (más apropiado)

### Modelos Nuevos Agregados
1. **`gpt-4-turbo`** → `llama3.3:70b` (máxima calidad)
2. **`gpt-4-vision-mini`** → `qwen3-vl:4b` (visión rápida)
3. **`o3-mini`** → `deepseek-r1:8b` (razonamiento intermedio)
4. **`o1-pro`** → `qwq:32b` (razonamiento premium)
5. **`gpt-3.5-turbo-instruct`** → `qwen2.5-coder:7b` (code completion)

### Ventajas Clave
- ✓ Uso de modelos con **razonamiento nativo** (qwen3 series)
- ✓ Modelos **multimodales únicos** (qwen3-vl con vision + reasoning)
- ✓ Mejor escalado por **tamaño y capacidades**
- ✓ Aprovechamiento de **contextos largos** (llama3 128K)
- ✓ Especialización apropiada (embeddings, code, reasoning, vision)

---

## 🎯 MAPEO RECOMENDADO POR CASO DE USO

| Caso de Uso | Modelo OpenAI | Modelo Ollama | Características |
|-------------|---------------|---------------|-----------------|
| Chat rápido | gpt-3.5-turbo | qwen3:4b | 32K ctx, reasoning |
| Chat balanceado | gpt-4o-mini | qwen3:8b | 32K ctx, reasoning |
| Chat premium | gpt-4, gpt-4o | qwen3:32b | 32K ctx, reasoning |
| Máxima calidad | gpt-4-turbo | llama3.3:70b | 70B params |
| Contexto largo | gpt-4-32k | llama3.1:8b | 128K ctx |
| Visión estable | gpt-4-vision-preview | llama3.2-vision:11b | 128K ctx, vision |
| Visión + razonamiento | gpt-4-turbo-vision | qwen3-vl:8b | vision + reasoning |
| Visión premium | gpt-4o-vision | qwen3-vl:32b | vision + reasoning |
| Visión rápida | gpt-4-vision-mini | qwen3-vl:4b | vision + reasoning |
| Embeddings rápidos | text-embedding-3-small | nomic-embed-text | lightweight |
| Embeddings quality | text-embedding-3-large | qwen3-embedding:8b | 32K ctx, 4096 dim |
| Razonamiento rápido | o1-mini | deepseek-r1:7b | reasoning |
| Razonamiento balanceado | o1-preview | deepseek-r1:14b | reasoning |
| Razonamiento intermedio | o3-mini | deepseek-r1:8b | reasoning |
| Razonamiento premium | o1 | deepseek-r1:32b | reasoning |
| Razonamiento avanzado | o1-pro | qwq:32b | advanced reasoning |
| Código rápido | gpt-3.5-turbo-instruct | qwen2.5-coder:7b | code |
| Código balanceado | code-davinci-002 | qwen2.5-coder:14b | code |
| Código premium | gpt-4-code | qwen3-coder:30b | code |

---

## 🚀 IMPLEMENTACIÓN

Para implementar este mapeo mejorado, se debe actualizar el archivo:
`shared/default_compat_models.py`

Con los cambios mencionados arriba, especialmente:
1. Cambiar los modelos indicados
2. Agregar los 5 nuevos modelos
3. Actualizar las definiciones de `litellm_params` y `model_info` con las características correctas
