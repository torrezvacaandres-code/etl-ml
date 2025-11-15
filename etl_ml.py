import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta, date
from dotenv import load_dotenv
import schedule
import time
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ml-comedor')

# Cargar variables de entorno
load_dotenv()

# Configuración
DB_URL = os.getenv("DATABASE_URL")
DIAS_A_PREDECIR = int(os.getenv("DIAS_A_PREDECIR", "7"))
RUN_SCHEDULE = os.getenv("RUN_SCHEDULE", "02:00")  # Hora de ejecución por defecto (2 AM)
RUN_ONCE = os.getenv("RUN_ONCE", "false").lower() == "true"

def get_db_connection():
    """Establece conexión con la base de datos."""
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        raise

def extraer_datos():
    """Lee la vista agregada del esquema public."""
    conn = get_db_connection()
    query = """
    SELECT fecha, tipo_comida, raciones_servidas 
    FROM public.vw_ventas_diarias 
    ORDER BY fecha ASC;
    """
    try:
        df = pd.read_sql(query, conn)
        logger.info(f"Datos extraídos correctamente: {len(df)} registros")
        return df
    except Exception as e:
        logger.error(f"Error al extraer datos: {e}")
        raise
    finally:
        conn.close()

def preprocesar_datos(df):
    """Transforma fechas y crea variables para el modelo (Features)."""
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Crear un índice completo para asegurar que no falten días (rellenar con 0 si cerró)
    # Esto es vital para que los cálculos de 'ayer' (lag) funcionen bien
    fechas = pd.date_range(start=df['fecha'].min(), end=df['fecha'].max())
    comidas = df['tipo_comida'].unique()
    
    # Producto cartesiano de fechas x comidas
    full_index = pd.MultiIndex.from_product([fechas, comidas], names=['fecha', 'tipo_comida'])
    df = df.set_index(['fecha', 'tipo_comida']).reindex(full_index, fill_value=0).reset_index()
    
    # Feature Engineering
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['es_finde'] = df['dia_semana'].isin([5, 6]).astype(int)
    df['mes'] = df['fecha'].dt.month
    
    # Lags (Raciones de hace 1 día, promedio de hace 7 días)
    # Agrupamos por tipo de comida para que el desayuno no se mezcle con la cena
    df['lag_1'] = df.groupby('tipo_comida')['raciones_servidas_total'].shift(1)
    df['media_movil_7'] = df.groupby('tipo_comida')['raciones_servidas_total'].transform(lambda x: x.rolling(7).mean())
    
    # Eliminar filas vacías generadas por los lags (los primeros 7 días)
    df = df.dropna()
    
    logger.info(f"Datos preprocesados: {len(df)} registros después del preprocesamiento")
    return df

def entrenar_y_predecir(df):
    """Entrena el modelo y genera predicciones futuras."""
    
    predicciones_futuras = []
    
    # Entrenamos un modelo INDEPENDIENTE por cada tipo de comida (Desayuno, Almuerzo, Cena)
    # Esto suele ser más preciso que un modelo general.
    for comida in df['tipo_comida'].unique():
        df_comida = df[df['tipo_comida'] == comida].copy()
        
        # Variables predictoras (X) y Objetivo (y)
        features = ['dia_semana', 'es_finde', 'mes', 'lag_1', 'media_movil_7']
        X = df_comida[features]
        y = df_comida['raciones_servidas_total']
        
        # Entrenar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        logger.info(f"Modelo entrenado para '{comida}' con {len(df_comida)} registros")
        
        # --- Generar futuro ---
        ultima_fecha = df_comida['fecha'].max()
        ultimo_valor_real = df_comida.iloc[-1]['raciones_servidas_total']
        
        # Simulamos los próximos N días
        # Nota: Para simplificar, usaremos el último valor real como lag estático 
        # (en prod real se hace una predicción recursiva)
        input_lag = ultimo_valor_real
        input_media = df_comida.iloc[-1]['media_movil_7'] # Simplificación
        
        for i in range(1, DIAS_A_PREDECIR + 1):
            fecha_futura = ultima_fecha + timedelta(days=i)
            
            # Crear features para el futuro
            future_features = pd.DataFrame([{
                'dia_semana': fecha_futura.dayofweek,
                'es_finde': 1 if fecha_futura.dayofweek >= 5 else 0,
                'mes': fecha_futura.month,
                'lag_1': input_lag,         # Usamos el valor del día anterior
                'media_movil_7': input_media
            }])
            
            prediccion = int(model.predict(future_features)[0])
            
            predicciones_futuras.append({
                'fecha': fecha_futura.date(),
                'comida': comida,
                'raciones_predichas': prediccion,
                'raciones_reales': None # Aún no ha pasado
            })
            
            # Actualizamos el lag para el siguiente día del bucle
            input_lag = prediccion 

    df_predicciones = pd.DataFrame(predicciones_futuras)
    logger.info(f"Se generaron {len(df_predicciones)} predicciones para los próximos {DIAS_A_PREDECIR} días")
    return df_predicciones

def preparar_actualizacion_historica(df_historico):
    """Prepara los datos reales de ayer para actualizar la tabla ML."""
    # Solo nos interesa subir el dato real para compararlo con la predicción que ya hicimos
    return df_historico[['fecha', 'tipo_comida', 'raciones_servidas_total']].copy()

def guardar_en_bd(df_predicciones, df_historico):
    """Usa UPSERT para guardar predicciones y actualizar datos reales."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Preparar datos históricos (Actualizar raciones_reales)
    # Convertimos a lista de tuplas
    datos_reales = []
    for _, row in df_historico.iterrows():
        datos_reales.append((
            row['fecha'], 
            row['tipo_comida'], 
            row['raciones_servidas_total']
        ))

    # Query para UPSERT de datos reales (Si existe la fecha, actualiza el campo real)
    query_real = """
    INSERT INTO ml.planificacion_diaria (fecha, comida, raciones_reales)
    VALUES %s
    ON CONFLICT (fecha, comida) 
    DO UPDATE SET 
        raciones_reales = EXCLUDED.raciones_reales,
        actualizado_en = NOW();
    """
    
    # 2. Preparar datos futuros (Insertar raciones_predichas)
    datos_futuros = []
    for _, row in df_predicciones.iterrows():
        datos_futuros.append((
            row['fecha'], 
            row['comida'], 
            row['raciones_predichas']
        ))

    # Query para UPSERT de predicciones (Si ya predijimos, actualizamos la predicción)
    query_futuro = """
    INSERT INTO ml.planificacion_diaria (fecha, comida, raciones_predichas)
    VALUES %s
    ON CONFLICT (fecha, comida) 
    DO UPDATE SET 
        raciones_predichas = EXCLUDED.raciones_predichas,
        actualizado_en = NOW();
    """

    try:
        # Ejecutar en batch
        if datos_reales:
            execute_values(cursor, query_real, datos_reales)
            logger.info(f"Actualizados {len(datos_reales)} registros históricos")
        if datos_futuros:
            execute_values(cursor, query_futuro, datos_futuros)
            logger.info(f"Insertadas {len(datos_futuros)} predicciones futuras")
        
        conn.commit()
        logger.info(f"✅ Éxito: {len(datos_reales)} registros históricos actualizados y {len(datos_futuros)} predicciones generadas.")
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Error al guardar datos en BD: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def ejecutar_etl():
    """Función principal que ejecuta todo el proceso ETL."""
    try:
        logger.info("🔄 Iniciando proceso ETL de ML...")
        
        # 1. Extraer
        df_raw = extraer_datos()
        
        if df_raw.empty:
            logger.warning("⚠️ No hay datos suficientes en vw_ventas_diarias para entrenar.")
            return
            
        # 2. Transformar
        df_procesado = preprocesar_datos(df_raw)
        
        # 3. Predecir (Futuro)
        df_preds = entrenar_y_predecir(df_procesado)
        
        # 4. Datos Reales (Pasado reciente para actualizar)
        # Tomamos los últimos 30 días para asegurar que actualizamos cualquier dato rezagado
        ultima_fecha = df_raw['fecha'].max() # Fecha original sin procesar
        df_realidad = df_raw[pd.to_datetime(df_raw['fecha']) >= (pd.to_datetime(ultima_fecha) - timedelta(days=30))]
        logger.info(f"Preparando actualización con {len(df_realidad)} registros históricos recientes")
        
        # 5. Cargar
        guardar_en_bd(df_preds, df_realidad)
        logger.info("Proceso ETL completado con éxito!")
    except Exception as e:
        logger.error(f"Error en el proceso ETL: {e}")

def verificar_conexion():
    """Verifica que la conexión a la base de datos funcione"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        logger.info("Conexión a la base de datos establecida correctamente")
        return True
    except Exception as e:
        logger.error(f"Error de conexión a la base de datos: {e}")
        return False

def verificar_tablas():
    """Verifica que existan las tablas necesarias"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verificar si existe la vista de ventas
        cursor.execute("SELECT to_regclass('public.vw_ventas_diarias')")
        existe_vista = cursor.fetchone()[0] is not None
        
        # Verificar si existe la tabla de planificación
        cursor.execute("SELECT to_regclass('ml.planificacion_diaria')")
        existe_tabla = cursor.fetchone()[0] is not None
        
        cursor.close()
        conn.close()
        
        if not existe_vista:
            logger.error("No existe la vista 'public.vw_ventas_diarias'. Es necesaria para el ETL.")
        if not existe_tabla:
            logger.error("No existe la tabla 'ml.planificacion_diaria'. Es necesaria para guardar resultados.")
            
        return existe_vista and existe_tabla
        
    except Exception as e:
        logger.error(f"Error al verificar tablas: {e}")
        return False

def run_job():
    """Función principal que se ejecuta en el schedule"""
    logger.info(f"Ejecutando job programado a las {RUN_SCHEDULE}")
    if verificar_conexion() and verificar_tablas():
        ejecutar_etl()
    else:
        logger.error("No se pudo ejecutar el ETL debido a errores en la configuración")

if __name__ == "__main__":
    logger.info(f"Iniciando servicio de ML-Comedor (v1.0)")
    
    if not DB_URL:
        logger.error("No se ha configurado DATABASE_URL en las variables de entorno")
        exit(1)
        
    # Verificaciones iniciales
    verificar_conexion()
    verificar_tablas()
    
    # Modo de ejecución
    if RUN_ONCE:
        logger.info("Ejecutando en modo único (run once)")
        ejecutar_etl()
    else:
        # Programar ejecución diaria
        logger.info(f"Programando ejecución diaria a las {RUN_SCHEDULE}")
        schedule.every().day.at(RUN_SCHEDULE).do(run_job)
        
        # También ejecutamos una vez al inicio
        run_job()
        
        # Mantener el programa corriendo
        while True:
            schedule.run_pending()
            time.sleep(60)
