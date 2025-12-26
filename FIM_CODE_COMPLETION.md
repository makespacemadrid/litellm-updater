# Fill-in-the-Middle (FIM) / Code Completion con LiteLLM

## 📋 Resumen

LiteLLM Companion detecta automáticamente modelos con capacidad FIM (Fill-in-the-Middle) y marca esta capacidad en los metadatos del modelo. Los clientes como Continue.dev y Cursor pueden usar los modelos directamente sin necesidad de configuración especial.

## ✅ Estado Actual en Nuestro Sistema

### 1. Detección Automática

```python
# Los modelos qwen2.5-coder reportan desde Ollama:
{
  "capabilities": ["completion", "tools", "insert"]
}

# Nuestro código (shared/models.py) automáticamente:
# - Extrae la capability "insert" / "fill_in_middle" / "fim"
# - La convierte a campos: supports_fill_in_middle, supports_code_infilling
# - Los marca en model_info al registrar en LiteLLM
```

### 2. Modelos con FIM Detectados

- `qwen2.5-coder:1.5b` ✓
- `qwen2.5-coder:7b` ✓
- `qwen2.5-coder:7b-base` ✓
- `qwen2.5-coder:14b` ✓
- Otros modelos que reporten "insert" en capabilities

## 🔧 Cómo Funciona

### Registro en LiteLLM

Cuando un proveedor tiene `auto_detect_fim=true` (por defecto), el sistema:

1. **Detecta** modelos con `supports_fill_in_middle` en sus capacidades
2. **Marca** en `model_info`:
   ```json
   {
     "supports_fill_in_middle": true,
     "supports_code_infilling": true
   }
   ```
3. **Registra** el modelo normalmente en LiteLLM (sin duplicados ni prefijos especiales)

### Ejemplo de Modelo Registrado

```json
{
  "model_name": "mks-ollama/qwen2.5-coder:7b",
  "litellm_params": {
    "model": "ollama_chat/qwen2.5-coder:7b",
    "api_base": "http://ollama:11434",
    "tags": ["capability:fill-in-middle", "capability:code-infilling", ...]
  },
  "model_info": {
    "litellm_provider": "ollama",
    "mode": "ollama_chat",
    "supports_fill_in_middle": true,
    "supports_code_infilling": true
  }
}
```

## 🎯 Integración con Continue.dev / Cursor

### Continue.dev

**Configuración para Tab Autocomplete:**

```json
{
  "tabAutocompleteModel": {
    "provider": "siliconflow",
    "model": "qwen2.5-coder:7b",
    "apiBase": "http://localhost:4000/",
    "apiKey": "sk-1234"
  }
}
```

**¿Por qué "siliconflow"?**
- Continue.dev usa formato FIM nativo cuando detecta ciertos providers
- `siliconflow` es uno de los providers que soporta FIM automáticamente
- Esto evita que Continue.dev aplique formato chat a las peticiones

**Alternativa (si el provider soporta OpenAI + FIM):**

```json
{
  "tabAutocompleteModel": {
    "provider": "openai",
    "model": "qwen2.5-coder:7b",
    "apiBase": "http://localhost:4000/v1",
    "apiKey": "sk-1234"
  }
}
```

### Cursor

Similar configuración en `settings.json`:

```json
{
  "cursor.cpp.fimModel": {
    "provider": "siliconflow",
    "model": "qwen2.5-coder:7b",
    "apiBase": "http://localhost:4000/",
    "apiKey": "sk-1234"
  }
}
```

## 📊 Ventajas del Nuevo Enfoque

| Aspecto | Enfoque Anterior | Nuevo Enfoque |
|---------|------------------|---------------|
| **Modelos duplicados** | ✗ Requería versión -fim separada | ✓ Un solo modelo |
| **Configuración** | ✗ Prefijo text-completion-codestral | ✓ Modo normal |
| **Detección FIM** | ✓ Automática | ✓ Automática |
| **Metadatos** | ⚠️ En modelo separado | ✓ En modelo principal |
| **Simplicidad** | ✗ Compleja | ✓ Simple |
| **Mantenimiento** | ✗ Dos modelos | ✓ Un modelo |

## 🔍 Verificación

### Check si un modelo tiene FIM:

```bash
# Via LiteLLM API
curl http://localhost:4000/model/info \
  -H "Authorization: Bearer sk-1234" | \
  jq '.data[] | select(.model_info.supports_fill_in_middle == true) | .model_name'

# Via API local
curl http://localhost:8000/api/models/123 | \
  jq '{
    model: .model_id,
    fim: .litellm_params.supports_fill_in_middle,
    infilling: .litellm_params.supports_code_infilling
  }'
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

### Test via LiteLLM con Continue.dev:

1. Configura Continue.dev con `provider: "siliconflow"`
2. El autocompletado debería funcionar automáticamente
3. LiteLLM rutea la petición a Ollama preservando el contexto FIM

## ⚙️ Configuración del Provider

### Habilitar/Deshabilitar Auto-detección FIM

En la UI de Admin (`/admin`), al editar un provider:

```
☑ Auto-detect FIM
  Automatically detect and mark Fill-in-the-Middle capability
  for code models with insert/infilling support
```

Por defecto está **habilitado**. Si lo deshabilitas:
- No se detectará FIM automáticamente
- Puedes marcar manualmente `supports_fill_in_middle` en los parámetros del modelo

## 📚 Referencias

- [Continue.dev Issue #4542](https://github.com/continuedev/continue/issues/4542) - Tab autocomplete configuration
- [LiteLLM Issue #9251](https://github.com/BerriAI/litellm/issues/9251) - FIM/Completions support
- [Ollama API - Generate](https://docs.ollama.com/api/generate) - Documentación del parámetro suffix
- [Ollama FIM Issue #3869](https://github.com/ollama/ollama/issues/3869) - FIM API support

## 🎬 Estado de Implementación

1. ✅ **HECHO:** Detección de capability "insert" → `supports_fill_in_middle`
2. ✅ **HECHO:** Marcado automático de capacidad FIM en model_info
3. ✅ **HECHO:** Eliminado workaround de text-completion-codestral
4. ✅ **HECHO:** Simplificado a un solo modelo por versión
5. ⏳ **PENDIENTE:** Documentar configuración específica para otros IDEs
6. ⏳ **PENDIENTE:** Tests de integración con Continue.dev/Cursor

---

**Última actualización:** 2025-12-26
**Estado:** Sistema simplificado - FIM detectado automáticamente como capacidad del modelo
