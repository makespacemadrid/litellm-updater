# Guía de Migración - LiteLLM Companion

## Migración desde Versiones Anteriores

Esta guía explica cómo actualizar desde versiones anteriores de LiteLLM Companion.

## ⚠️ IMPORTANTE: Haz Backup Antes de Migrar

```bash
# Crear backup de la base de datos
cp data/models.db data/models.db.backup.$(date +%Y%m%d_%H%M%S)
```

## Migración Automática (Recomendado)

El sistema detecta automáticamente si existe una base de datos y aplica las migraciones necesarias:

1. **Detener los servicios:**
   ```bash
   docker compose down
   ```

2. **Actualizar el código:**
   ```bash
   git pull origin main
   ```

3. **Reconstruir las imágenes:**
   ```bash
   docker compose build --no-cache litellm-companion-web litellm-companion-backend
   ```

4. **Iniciar los servicios:**
   ```bash
   docker compose up -d
   ```

El script de inicio (`start_web.sh`) detectará la base de datos existente y ejecutará las migraciones automáticamente.

## Migración Manual (Si la automática falla)

Si la migración automática falla, puedes ejecutar el script de migración manual:

```bash
docker compose run --rm litellm-companion-web bash /app/scripts/migrate_from_previous.sh
```

Este script:
1. Crea un backup automático de la base de datos
2. Intenta ejecutar las migraciones de Alembic
3. Si falla, usa el método de inicialización alternativo
4. Te proporciona instrucciones para restaurar el backup si algo sale mal

## Verificación Post-Migración

Después de la migración, verifica que todo funciona correctamente:

1. **Verifica que el servicio está corriendo:**
   ```bash
   docker compose ps
   ```

2. **Verifica los logs:**
   ```bash
   docker compose logs litellm-companion-web --tail=50
   ```

3. **Verifica que la columna auto_detect_fim existe:**
   ```bash
   docker compose exec litellm-companion-web sqlite3 /app/data/models.db \
     "PRAGMA table_info(providers);" | grep auto_detect
   ```

   Deberías ver algo como:
   ```
   14|auto_detect_fim|INTEGER|1|1|0
   ```

4. **Accede a la interfaz web:**
   ```
   http://localhost:4001
   ```

## Secuencia de Migraciones

Las migraciones se ejecutan en este orden:

- **001**: Esquema inicial (providers, models)
- **002**: Agregar tags (provider tags, model system_tags, user_tags)
- **003**: Renombrar tipo 'litellm' → 'openai', agregar tipo 'compat'
- **004**: Agregar campos de pricing
- **005**: Agregar campo auto_detect_fim

## Cambios Importantes en Esta Versión

### Campo `auto_detect_fim`
- Nuevo campo en la tabla `providers`
- Permite detección automática de capacidades Fill-in-the-Middle (FIM)
- Por defecto: `True`

### Corrección de Inicialización de Base de Datos
- Se corrigió un bug crítico donde `ensure_minimum_schema()` causaba rollback de transacciones
- Ahora usa `engine.connect()` con commit manual en lugar de `engine.begin()`
- Las migraciones son idempotentes y seguras de ejecutar múltiples veces

### Migraciones Compatibles con SQLite
- Las migraciones 002 y 003 ahora usan SQL directo con try/except
- Eliminados los comandos no soportados por SQLite (ALTER COLUMN, DROP CONSTRAINT)

## Restaurar desde Backup

Si algo sale mal durante la migración, puedes restaurar desde el backup:

```bash
# Detener servicios
docker compose down

# Restaurar backup (reemplaza YYYYMMDD_HHMMSS con la fecha de tu backup)
cp data/models.db.backup.YYYYMMDD_HHMMSS data/models.db

# Reiniciar servicios
docker compose up -d
```

## Instalación Desde Cero

Si prefieres empezar desde cero:

```bash
# Detener servicios
docker compose down

# Eliminar base de datos (¡asegúrate de tener backup!)
rm data/models.db*

# Reiniciar servicios
docker compose up -d
```

El sistema creará automáticamente una base de datos nueva con el esquema completo.

## Problemas Comunes

### Error: "no such column: providers.auto_detect_fim"

**Causa**: La migración no se completó correctamente.

**Solución**:
```bash
docker compose down
docker compose run --rm litellm-companion-web bash /app/scripts/migrate_from_previous.sh
docker compose up -d
```

### Error: "duplicate column name: auto_detect_fim"

**Causa**: La columna ya existe pero las migraciones intentan agregarla de nuevo.

**Solución**: Las migraciones son idempotentes, este error es esperado y no causa problemas. Si persiste:
```bash
docker compose exec litellm-companion-web python -m alembic stamp head
docker compose restart litellm-companion-web
```

### Servicios no inician después de la migración

**Solución**:
1. Verifica los logs: `docker compose logs litellm-companion-web`
2. Verifica que la base de datos no está corrupta: `docker compose exec litellm-companion-web sqlite3 /app/data/models.db "PRAGMA integrity_check;"`
3. Si hay problemas, restaura desde backup

## Soporte

Si encuentras problemas durante la migración:

1. Revisa los logs detalladamente
2. Verifica que tienes un backup antes de cualquier acción destructiva
3. Reporta el issue en: https://github.com/makespacemadrid/litellm-companion/issues

---

## Ver También

- [MODEL_STATISTICS.md](MODEL_STATISTICS.md) - Estadísticas de modelos y cobertura de API de OpenAI
- [README.md](README.md) - Documentación general del proyecto
- [CLAUDE.md](CLAUDE.md) - Guía de desarrollo

---

**Última actualización**: 2025-12-26
