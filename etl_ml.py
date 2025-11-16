import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta, date
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
DB_URL = os.getenv("DATABASE_URL")
DIAS_A_PREDECIR = 7
MODELO_VERSION = 'v1_random_forest'

def get_db_connection():
    return psycopg2.connect(DB_URL)

def extraer_datos():
    """
    MODIFICADO: Lee la vista y extrae los desgloses de becados/regulares.
    """
    conn = get_db_connection()
    query = """
    SELECT fecha, tipo_comida, 
           raciones_servidas,
           raciones_becados,
           raciones_regulares
    FROM public.vw_ventas_diarias 
    ORDER BY fecha ASC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocesar_datos(df):
    """Transforma fechas y crea variables para el modelo (Features)."""
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Crear un índice completo para asegurar que no falten días (rellenar con 0 si cerró)
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
    df['lag_1'] = df.groupby('tipo_comida')['raciones_servidas'].shift(1)
    df['media_movil_7'] = df.groupby('tipo_comida')['raciones_servidas'].transform(lambda x: x.rolling(7).mean())
    
    # Eliminar filas vacías generadas por los lags (los primeros 7 días)
    df = df.dropna()
    
    return df

def entrenar_y_predecir(df):
    """Entrena el modelo y genera predicciones futuras."""
    
    predicciones_futuras = []
    
    for comida in df['tipo_comida'].unique():
        df_comida = df[df['tipo_comida'] == comida].copy()
        
        # Variables predictoras (X) y Objetivo (y)
        features = ['dia_semana', 'es_finde', 'mes', 'lag_1', 'media_movil_7']
        X = df_comida[features]
        y = df_comida['raciones_servidas']
        
        # Entrenar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # --- Generar futuro ---
        ultima_fecha = df_comida['fecha'].max()
        ultimo_valor_real = df_comida.iloc[-1]['raciones_servidas']
        
        input_lag = ultimo_valor_real
        input_media = df_comida.iloc[-1]['media_movil_7'] # Simplificación
        
        for i in range(1, DIAS_A_PREDECIR + 1):
            fecha_futura = ultima_fecha + timedelta(days=i)
            
            # Crear features para el futuro
            future_features = pd.DataFrame([{
                'dia_semana': fecha_futura.dayofweek,
                'es_finde': 1 if fecha_futura.dayofweek >= 5 else 0,
                'mes': fecha_futura.month,
                'lag_1': input_lag,
                'media_movil_7': input_media
            }])
            
            prediccion = int(model.predict(future_features)[0])
            
            predicciones_futuras.append({
                'fecha': fecha_futura.date(),
                'comida': comida,
                'raciones_predichas': prediccion,
                # NOTA: No podemos predecir confianza_prediccion con RandomForest
                # de forma sencilla, lo dejamos NULO.
            })
            
            input_lag = prediccion 

    return pd.DataFrame(predicciones_futuras)

def preparar_actualizacion_historica(df_historico):
    """
    MODIFICADO: Prepara todos los datos reales para actualizar la tabla ML.
    """
    # Renombramos 'raciones_servidas' para claridad en el siguiente paso
    df_historico = df_historico.rename(columns={
        'raciones_servidas': 'raciones_consumidas_real'
    })
    
    # Seleccionamos las columnas que coinciden con la tabla de destino
    columnas_reales = [
        'fecha', 
        'tipo_comida', 
        'raciones_consumidas_real', 
        'raciones_becados', 
        'raciones_regulares'
    ]
    return df_historico[columnas_reales].copy()

def guardar_en_bd(df_predicciones, df_historico):
    """
    MODIFICADO: Usa UPSERT con los nombres de columna correctos de tu esquema.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Preparar datos históricos (Actualizar datos reales)
    datos_reales = []
    for _, row in df_historico.iterrows():
        datos_reales.append((
            row['fecha'], 
            row['tipo_comida'], 
            row['raciones_consumidas_real'],
            row['raciones_becados'],
            row['raciones_regulares']
        ))

    # Query para UPSERT de datos reales
    # OJO: Los nombres de columna coinciden con tu SQL
    query_real = """
    INSERT INTO ml.planificacion_diaria (
        fecha, comida, 
        raciones_consumidas_real, 
        raciones_becados_real, 
        raciones_regulares_real
    )
    VALUES %s
    ON CONFLICT (fecha, comida) 
    DO UPDATE SET 
        raciones_consumidas_real = EXCLUDED.raciones_consumidas_real,
        raciones_becados_real = EXCLUDED.raciones_becados_real,
        raciones_regulares_real = EXCLUDED.raciones_regulares_real,
        actualizado_en = NOW();
    """
    
    # 2. Preparar datos futuros (Insertar raciones_predichas)
    datos_futuros = []
    for _, row in df_predicciones.iterrows():
        datos_futuros.append((
            row['fecha'], 
            row['comida'], 
            row['raciones_predichas'],
            MODELO_VERSION
        ))

    # Query para UPSERT de predicciones
    # OJO: Los nombres de columna coinciden con tu SQL
    query_futuro = """
    INSERT INTO ml.planificacion_diaria (
        fecha, comida, 
        raciones_predichas, 
        modelo_version
    )
    VALUES %s
    ON CONFLICT (fecha, comida) 
    DO UPDATE SET 
        raciones_predichas = EXCLUDED.raciones_predichas,
        modelo_version = EXCLUDED.modelo_version,
        actualizado_en = NOW();
    """

    try:
        # Ejecutar en batch
        if datos_reales:
            execute_values(cursor, query_real, datos_reales)
        if datos_futuros:
            execute_values(cursor, query_futuro, datos_futuros)
        
        conn.commit()
        print(f"✅ Éxito: {len(datos_reales)} registros históricos actualizados y {len(datos_futuros)} predicciones generadas.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    print("🔄 Iniciando proceso ETL de ML...")
    
    # 1. Extraer
    df_raw = extraer_datos()
    
    if df_raw.empty or len(df_raw) < 10: # Necesitamos suficientes datos
        print("⚠️ No hay datos suficientes en vw_ventas_diarias para entrenar.")
    else:
        # 2. Transformar
        df_procesado = preprocesar_datos(df_raw)
        
        # 3. Predecir (Futuro)
        df_preds = entrenar_y_predecir(df_procesado)
        
        # 4. Datos Reales (Pasado reciente para actualizar)
        # Tomamos los últimos 30 días para asegurar que actualizamos cualquier dato rezagado
        ultima_fecha = df_raw['fecha'].max()
        df_realidad_raw = df_raw[pd.to_datetime(df_raw['fecha']) >= (pd.to_datetime(ultima_fecha) - timedelta(days=30))]
        df_realidad = preparar_actualizacion_historica(df_realidad_raw)

        # 5. Cargar
        guardar_en_bd(df_preds, df_realidad)