# Revisión Semántica de Nomenclatura - LiteLLM Updater

**Fecha:** 2025-11-27
**Objetivo:** Identificar nombres confusos o inconsistentes que puedan inducir a error

---

## ❌ Problemas Críticos

### 1. Inconsistencia Terminológica: "Source" vs "Provider"

**Severidad:** Alta
**Ubicaciones:** A lo largo del proyecto

**Problema:**
El modelo de datos usa consistentemente "source" (SourceEndpoint, SourceType, SourceModels), pero la UI y algunas rutas usan "provider":

- `/providers` ruta HTTP
- `providers.html` template
- `providers_page()` función en web.py:152
- `refresh_provider_models()` función en web.py:305

**Impacto:**
Genera confusión sobre si "provider" y "source" son conceptos diferentes o el mismo concepto con nombres distintos.

**Recomendación:**
Elegir UN término y usarlo consistentemente:
- **Opción A:** Cambiar todo a "source" (más preciso, ya que son fuentes upstream)
- **Opción B:** Cambiar todo a "provider" (más común en contexto de APIs)

**Opción recomendada:** Mantener "source" y renombrar:
- `providers.html` → `sources.html`
- `/providers` → `/sources`
- `providers_page` → `sources_page`
- `refresh_provider_models` → `refresh_source_models`

---

### 2. Nombre de Propiedad Engañoso: `litellm_mappable`

**Severidad:** Media-Alta
**Ubicación:** `models.py:400`

**Problema:**
```python
@property
def litellm_mappable(self) -> dict[str, Any]:
    """Return LiteLLM-compatible fields from the raw payload, omitting nulls."""
```

El nombre `litellm_mappable` sugiere un valor booleano (¿es mapeable?), pero retorna un diccionario.

**Impacto:**
```python
# Lectura confusa:
if model.litellm_mappable:  # ¿Es esto un check o estoy usando el dict?

# Uso actual:
fields = model.litellm_mappable  # No queda claro que es un dict
```

**Recomendación:**
Renombrar a uno de estos:
- `to_litellm_dict()` (método, más explícito)
- `litellm_fields` (propiedad, sustantivo claro)
- `litellm_model_dict` (propiedad, muy explícito)

**Preferencia:** `litellm_fields`

---

### 3. Clase `LitellmTarget` - Nombre Ambiguo

**Severidad:** Media
**Ubicación:** `models.py:37`

**Problema:**
`LitellmTarget` representa el **destino** de sincronización, no una fuente. En el contexto del sistema:
- `SourceEndpoint` = de donde se LEEN modelos
- `LitellmTarget` = a donde se ESCRIBEN modelos

El término "target" es válido pero menos claro que "destination".

**Recomendación:**
Renombrar a `LitellmDestination` o `SyncDestination` para mayor claridad semántica.

---

## ⚠️ Problemas Moderados

### 4. Inconsistencia en Nombres de Funciones de Rutas

**Severidad:** Media
**Ubicación:** `web.py`

**Problema:**
Falta de patrón consistente en nombres de funciones de ruta:

| Ruta | Función | Patrón |
|------|---------|---------|
| `/` | `index` | Simple ✓ |
| `/admin` | `admin_page` | Con sufijo |
| `/providers` | `providers_page` | Con sufijo |
| `/litellm` | `litellm_page` | Con sufijo |
| `/models` | `models_endpoint` | Con sufijo diferente |
| `/models/show` | `model_details` | Simple |
| `/sync` | `manual_sync` | Con prefijo |
| `/admin/sources` | `add_source_form` | Con sufijo diferente |

**Recomendación:**
Adoptar patrón consistente:

**Para vistas HTML:**
- `/` → `index`
- `/admin` → `admin`
- `/sources` → `sources` (renombrado)
- `/litellm` → `litellm`

**Para endpoints API:**
- `/api/sources` → `api_sources` ✓
- `/api/models` → `api_models` ✓
- `/models` → `models_redirect` (ya que redirige)
- `/models/show` → `model_details`

**Para acciones POST:**
- `/admin/sources` → `add_source`
- `/admin/sources/delete` → `delete_source`
- `/admin/litellm` → `update_litellm` ✓
- `/admin/interval` → `update_interval` ✓
- `/sync` → `run_sync`
- `/sources/refresh` → `refresh_source`

---

### 5. Confusión entre `model_type` y `mode`

**Severidad:** Media
**Ubicación:** `models.py:350-365`

**Problema:**
`ModelMetadata` tiene dos campos similares:
```python
model_type: str | None  # "embedding", "completion", "image"
mode: str | None        # "chat", "embeddings", "audio_transcription"
```

Ambos describen el tipo/modo del modelo pero con valores diferentes y sin documentación clara de cuándo usar cada uno.

**Uso actual:**
- `model_type`: Se extrae/infiere de los datos upstream
- `mode`: Se usa para compatibilidad con LiteLLM

**Recomendación:**
Mejorar la documentación o renombrar:
```python
model_type: str | None  # Tipo upstream original
litellm_mode: str | None  # Modo LiteLLM (chat, embeddings, etc.)
```

---

### 6. Función `_extract_supported_openai_params` - Nombre Incompleto

**Severidad:** Media-Baja
**Ubicación:** `models.py:454`

**Problema:**
```python
def _extract_supported_openai_params(self) -> list[str]:
    """Extract supported OpenAI parameters from Ollama's parameters field."""
```

El nombre dice "extract" pero la función también **mapea** parámetros de Ollama a equivalentes OpenAI (ej: `repeat_penalty` → `frequency_penalty`).

**Recomendación:**
Renombrar a `_extract_and_map_openai_params` o `_get_openai_compatible_params`

---

## ℹ️ Mejoras Sugeridas (Prioridad Baja)

### 7. Función `_human_source_type` en Lugar Incorrecto

**Ubicación:** `web.py:111`

**Problema:**
```python
def _human_source_type(source_type: SourceType) -> str:
    return "Ollama" if source_type is SourceType.OLLAMA else "LiteLLM / OpenAI"
```

Esta función de formateo/presentación está en `web.py` pero podría ser útil en otros contextos. Debería estar en `models.py` como método de `SourceType` o en un módulo de utilidades.

**Recomendación:**
Mover a `models.py`:
```python
class SourceType(str, Enum):
    OLLAMA = "ollama"
    LITELLM = "litellm"

    def display_name(self) -> str:
        """Return human-readable name for UI display."""
        return "Ollama" if self is SourceType.OLLAMA else "LiteLLM / OpenAI"
```

---

### 8. Ordenamiento de Definiciones en `sources.py`

**Ubicación:** `sources.py:139`

**Problema:**
La función `_clean_ollama_payload` se define en la línea 139 pero se usa en la línea 62. Aunque Python lo permite (por ser async), puede dificultar la lectura.

**Recomendación:**
Mover definiciones de funciones helper privadas antes de las funciones públicas que las usan.

---

### 9. Acceso Directo a Campo Opcional

**Ubicación:** `sync.py:62`

**Problema:**
```python
await _register_model_with_litellm(
    client, config.litellm.base_url, config.litellm.api_key, model
)
```

Se accede a `base_url` directamente cuando podría ser `None`. Aunque hay un check previo en línea 55, sería más seguro usar `normalized_base_url` que valida.

**Código actual (líneas 55-63):**
```python
if not config.litellm.configured:
    logger.info("LiteLLM target not configured; skipping registration for %s", source.name)
    continue

for model in source_models.models:
    try:
        await _register_model_with_litellm(
            client, config.litellm.base_url, config.litellm.api_key, model
        )
```

**Recomendación:**
```python
await _register_model_with_litellm(
    client, config.litellm.normalized_base_url, config.litellm.api_key, model
)
```

---

## 📊 Resumen de Cambios Recomendados

### Críticos (Hacer ASAP)
1. ✅ Resolver inconsistencia source/provider
2. ✅ Renombrar `litellm_mappable` → `litellm_fields`
3. ✅ Renombrar `LitellmTarget` → `LitellmDestination`

### Importantes (Hacer pronto)
4. ✅ Estandarizar nombres de funciones de rutas
5. ✅ Clarificar `model_type` vs `mode`
6. ✅ Renombrar `_extract_supported_openai_params`

### Opcionales (Cuando haya tiempo)
7. Mover `_human_source_type` a método de enum
8. Reordenar definiciones en `sources.py`
9. Usar `normalized_base_url` en `sync.py`

---

## 🎯 Plan de Implementación Sugerido

### Fase 1: Resolver Inconsistencias Críticas
1. Decidir: ¿"source" o "provider"?
2. Refactorizar nombres a través del proyecto
3. Actualizar tests

### Fase 2: Mejorar Claridad de API Interna
4. Renombrar propiedades/métodos confusos
5. Mejorar documentación de campos similares

### Fase 3: Limpieza y Optimización
6. Reorganizar código
7. Mejorar type hints
8. Actualizar documentación

---

## ✅ Aspectos Bien Nombrados (Felicitaciones)

- ✅ `SourceEndpoint` - Claro y descriptivo
- ✅ `ModelMetadata` - Preciso
- ✅ `SyncState` - Obvio su propósito
- ✅ `ModelDetailsCache` - Muy claro
- ✅ `fetch_*` prefijo para funciones de fetching
- ✅ `_make_auth_headers` - Verbo claro
- ✅ Uso de `_` para funciones privadas
- ✅ Constantes en UPPER_CASE (DEFAULT_TIMEOUT, DEFAULT_CONFIG_PATH)
- ✅ Enums con valores descriptivos

---

**Conclusión:** El código tiene buena estructura general, pero sufre de inconsistencias terminológicas que pueden generar confusión. Resolver el problema "source vs provider" tendría el mayor impacto en la claridad del código.
