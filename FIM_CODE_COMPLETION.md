# Fill-in-the-Middle (FIM) / Code Completion con LiteLLM

## 📋 Resumen

Basado en los issues [#4542](https://github.com/continuedev/continue/issues/4542) y [#9251](https://github.com/BerriAI/litellm/issues/9251), **SÍ es posible** usar FIM (Fill-in-the-Middle) a través de LiteLLM, pero requiere configuración especial.

## ✅ Estado Actual en Nuestro Sistema

### 1. Detección Automática
```python
# Los modelos qwen2.5-coder reportan desde Ollama:
{
  "capabilities": ["completion", "tools", "insert"]
}

# Nuestro código (shared/models.py) automáticamente:
# - Extrae la capability "insert"
# - La convierte a tag: "capability:insert"
# - Mapea a campos: supports_fill_in_middle, supports_code_infilling
```

### 2. Modelos con FIM Detectados
- `qwen2.5-coder:1.5b` ✓
- `qwen2.5-coder:7b` ✓
- `qwen2.5-coder:7b-base` ✓
- `qwen2.5-coder:14b` ✓

## 🔧 Cómo Funciona FIM

### Opción A: Usar Ollama Directo (Más Simple)

**Endpoint:** `/api/generate`

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:7b",
  "prompt": "def compute_gcd(a, b):",
  "suffix": "    return result",
  "stream": false
}'
```

**Ventajas:**
- ✓ Funciona directamente sin configuración adicional
- ✓ Formato nativo de Ollama
- ✓ Soporta `suffix` parameter

**Desventajas:**
- ✗ No pasa por LiteLLM (sin logging/analytics/rate limiting)
- ✗ No puede usar access groups de LiteLLM
- ✗ No aparece en el dashboard de LiteLLM

### Opción B: A través de LiteLLM (Requiere Workaround)

**Endpoint:** `/v1/completions` (NO `/fim/completions`)

**IMPORTANTE:** LiteLLM no tiene endpoint nativo `/fim/completions`, pero podemos usar `/v1/completions` con un truco.

#### Paso 1: Registrar el Modelo con Prefijo Especial

```python
# En LiteLLM, registrar con prefijo text-completion-codestral/
{
  "model_name": "mks-ollama/qwen2.5-coder:7b-fim",
  "litellm_params": {
    "model": "text-completion-codestral/qwen2.5-coder:7b",  # ← Prefijo mágico
    "api_base": "http://ollama:11434"
  },
  "model_info": {
    "mode": "completion",  # NO "chat"
    "supports_fill_in_middle": true
  }
}
```

#### Paso 2: Llamar al Endpoint `/v1/completions`

```bash
curl http://localhost:4000/v1/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mks-ollama/qwen2.5-coder:7b-fim",
    "prompt": "def compute_gcd(a, b):",
    "suffix": "    return result",
    "max_tokens": 100,
    "temperature": 0
  }'
```

**Ventajas:**
- ✓ Pasa por LiteLLM (logging, analytics, rate limiting)
- ✓ Access groups de LiteLLM
- ✓ Dashboard unificado

**Desventajas:**
- ✗ Requiere registrar modelo DOS veces (versión chat + versión FIM)
- ✗ Workaround no oficial (puede cambiar)
- ✗ Más complejo de configurar

## 🎯 Integración con Continue.dev / Cursor

### Continue.dev

**Configuración para Tab Autocomplete:**

```json
{
  "tabAutocompleteModel": {
    "provider": "siliconflow",  // ← Trick: NO usar "openai"
    "model": "qwen2.5-coder:7b-fim",
    "apiBase": "http://localhost:4000/",
    "apiKey": "sk-1234"
  }
}
```

**¿Por qué "siliconflow"?**
- Continue.dev aplica formato chat cuando detecta provider "openai"
- Usando "siliconflow" envía el formato correcto con tokens FIM

### Cursor / Otros IDEs

Similar configuración, depende de cómo el IDE envía las peticiones.

## 📊 Comparativa de Enfoques

| Aspecto | Ollama Directo | LiteLLM Proxy |
|---------|----------------|---------------|
| **Endpoint** | `/api/generate` | `/v1/completions` |
| **Configuración** | Simple | Compleja (prefijo especial) |
| **Parámetro suffix** | ✓ Nativo | ✓ Via workaround |
| **LiteLLM logging** | ✗ | ✓ |
| **Access control** | ✗ | ✓ |
| **Dashboard** | ✗ | ✓ |
| **Rate limiting** | ✗ | ✓ |
| **Estabilidad** | ✓✓✓ | ⚠️ Workaround |

## 🚀 Propuesta de Implementación

### Opción 1: Doble Registro (Automático)

Para cada modelo con `capability:insert`, registrar DOS versiones en LiteLLM:

```python
# Ejemplo: qwen2.5-coder:7b

# 1. Versión Chat (normal)
{
  "model_name": "mks-ollama/qwen2.5-coder:7b",
  "litellm_params": {
    "model": "ollama/qwen2.5-coder:7b",
    "api_base": "http://ollama:11434"
  },
  "model_info": {
    "mode": "chat"
  }
}

# 2. Versión FIM (code completion)
{
  "model_name": "mks-ollama/qwen2.5-coder:7b-fim",  # Sufijo -fim
  "litellm_params": {
    "model": "text-completion-codestral/qwen2.5-coder:7b",  # Prefijo especial
    "api_base": "http://ollama:11434"
  },
  "model_info": {
    "mode": "completion",
    "supports_fill_in_middle": true,
    "supports_code_infilling": true
  }
}
```

### Opción 2: Solo Documentar (Manual)

- Documentar que los usuarios pueden usar Ollama directo para FIM
- Proporcionar ejemplos de configuración para Continue.dev/Cursor
- No registrar automáticamente versiones FIM

### Opción 3: Flag de Usuario

Agregar un flag en la UI del provider:
```
☐ Register FIM variants for code models
```

Si está activado, auto-registrar versiones `-fim` de modelos con `capability:insert`.

## 🔍 Verificación

### Check si un modelo tiene FIM:

```bash
# Via API
curl http://localhost:8000/api/models/123 | jq '.litellm_params.supports_fill_in_middle'

# Via database
sqlite3 data/models.db "
  SELECT model_id, litellm_params
  FROM models m
  JOIN providers p ON m.provider_id = p.id
  WHERE p.name = 'mks'
    AND json_extract(litellm_params, '$.supports_fill_in_middle') = 1
"
```

### Test FIM directo con Ollama:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:7b",
  "prompt": "def fibonacci(n):",
  "suffix": "    return result",
  "stream": false
}' | jq -r '.response'
```

### Test FIM via LiteLLM:

```bash
# Primero registrar el modelo con prefijo text-completion-codestral/
# Luego:
curl http://localhost:4000/v1/completions \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "mks-ollama/qwen2.5-coder:7b-fim",
    "prompt": "def fibonacci(n):",
    "suffix": "    return result",
    "max_tokens": 100
  }' | jq -r '.choices[0].text'
```

## 📚 Referencias

- [Continue.dev Issue #4542](https://github.com/continuedev/continue/issues/4542) - Tab autocomplete no funciona con OpenAI provider
- [LiteLLM Issue #9251](https://github.com/BerriAI/litellm/issues/9251) - FIM/Completions support
- [Ollama API - Generate](https://docs.ollama.com/api/generate) - Documentación oficial del parámetro suffix
- [Ollama FIM Issue #3869](https://github.com/ollama/ollama/issues/3869) - API para FIM tasks

## 🎬 Próximos Pasos

1. ✅ **HECHO:** Agregar detección de capability "insert" → `supports_fill_in_middle`
2. ⏳ **PENDIENTE:** Decidir estrategia de registro (automático vs manual)
3. ⏳ **PENDIENTE:** Actualizar UI para mostrar modelos con FIM
4. ⏳ **PENDIENTE:** Documentar configuración para Continue.dev/Cursor
5. ⏳ **PENDIENTE:** Tests de integración con FIM

---

**Última actualización:** 2025-12-25
**Estado:** Campo `supports_fill_in_middle` implementado, pendiente estrategia de registro dual
