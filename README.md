# ML Comedor - Planificación de Raciones con Inteligencia Artificial

Sistema de predicción de raciones para comedores institucionales mediante Machine Learning. Integra con Supabase para predecir necesidades diarias de raciones según datos históricos.

## Características

- **Predicción automática**: Genera estimaciones de raciones necesarias para los próximos 7 días
- **Análisis histórico**: Compara predicciones con datos reales para medir precisión
- **Integración con Supabase**: Almacena los resultados en un esquema dedicado
- **Dockerizado**: Despliegue fácil en cualquier entorno

## Diagrama de Arquitectura

```
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│               │      │               │      │               │
│  Supabase DB  │◄────►│   ML-Comedor  │◄────►│  Frontend/BI  │
│ (PostgreSQL)  │      │  (Predictor)  │      │  (Opcional)   │
│               │      │               │      │               │
└───────────────┘      └───────────────┘      └───────────────┘
```

## Requisitos Previos

- Docker y Docker Compose
- Cuenta en Supabase o PostgreSQL
- Base de datos con esquema de tickets existente

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/ml-comedor.git
   cd ml-comedor
   ```

2. **Configurar variables de entorno**:
   ```bash
   cp example.env .env
   ```
   Edita el archivo `.env` con tus credenciales de base de datos.

3. **Inicializar la base de datos**:
   - Ejecuta el script `init-db/01_schema.sql` en tu instancia de Supabase/PostgreSQL.

4. **Construir y ejecutar con Docker**:
   ```bash
   docker-compose up -d --build
   ```

## Uso

El servicio se ejecuta automáticamente a las 2:00 AM todos los días (configurable). También puedes ejecutarlo manualmente:

```bash
# Ejecutar una vez y terminar
docker-compose run --rm -e RUN_ONCE=true ml-comedor

# Ver los logs
docker-compose logs -f ml-comedor
```

## Estructura de Datos

### Tabla de Resultados

La predicción y datos reales se almacenan en:

```sql
ml.planificacion_diaria (
    id,
    fecha,
    comida,
    raciones_predichas,
    raciones_reales,
    error_absoluto,  -- Calculado automáticamente
    modelo_version,
    actualizado_en
)
```

## Consultas Útiles

Ver predicciones para la próxima semana:
```sql
SELECT fecha, comida, raciones_predichas 
FROM ml.planificacion_diaria
WHERE fecha > CURRENT_DATE 
ORDER BY fecha ASC, comida;
```

Ver comparación predicción vs. realidad:
```sql
SELECT fecha, comida, raciones_predichas, raciones_reales, error_absoluto
FROM ml.planificacion_diaria 
WHERE raciones_reales IS NOT NULL
ORDER BY fecha DESC;
```

## Personalización

- **Número de días de predicción**: Modifica la variable `DIAS_A_PREDECIR` en tu `.env`
- **Hora de ejecución**: Modifica la variable `RUN_SCHEDULE` en tu `.env`
- **Algoritmo**: El código usa RandomForestRegressor, pero puedes modificar el modelo en `etl_ml.py`

## Licencia

MIT
