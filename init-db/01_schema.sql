-- Crear el esquema ML si no existe
CREATE SCHEMA IF NOT EXISTS ml;

-- Asegurarse que existe el tipo enum (si no existe, crearlo)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tipo_comida') THEN
        CREATE TYPE tipo_comida AS ENUM ('DESAYUNO', 'ALMUERZO', 'CENA');
    END IF;
END
$$;

-- Crear la tabla de planificación (Resultados del ML)
CREATE TABLE IF NOT EXISTS ml.planificacion_diaria (
    id BIGSERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    comida tipo_comida NOT NULL, -- Usa el ENUM que ya definiste en public

    -- Predicción (Lo que dice la IA hoy para mañana)
    raciones_predichas INT,
    
    -- Realidad (Se llena al día siguiente con datos reales)
    raciones_reales INT,
    
    -- Métricas de calidad
    error_absoluto INT GENERATED ALWAYS AS (ABS(raciones_predichas - raciones_reales)) STORED,
    
    -- Auditoría
    modelo_version TEXT DEFAULT 'v1_random_forest',
    actualizado_en TIMESTAMPTZ DEFAULT now(),

    -- Clave única vital para el UPSERT (evita duplicados)
    CONSTRAINT uq_planificacion_fecha_comida UNIQUE (fecha, comida)
);

-- Índices para consultas rápidas desde el Frontend/BI
CREATE INDEX IF NOT EXISTS idx_ml_fecha ON ml.planificacion_diaria(fecha);

-- Vista para agregación de ventas diarias (esta vista debería existir en el esquema principal)
-- Si esta vista ya existe en tu esquema public, no es necesario crearla de nuevo
CREATE OR REPLACE VIEW public.vw_ventas_diarias AS
SELECT 
    t.fecha::date AS fecha,
    t.tipo_comida AS comida,
    COUNT(*) AS cantidad_tickets,
    SUM(t.cantidad_raciones) AS raciones_servidas_total
FROM 
    public.tickets t
WHERE 
    t.estado = 'UTILIZADO'
GROUP BY 
    t.fecha::date, t.tipo_comida
ORDER BY 
    t.fecha::date, t.tipo_comida;

-- Permisos (ajustar según tu modelo de usuarios)
GRANT USAGE ON SCHEMA ml TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml TO postgres;
