import pandas as pd
import numpy as np
import joblib
import os
from sqlalchemy import create_engine, text
from datetime import datetime
from dateutil.relativedelta import relativedelta
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- CONFIGURACIÓN ---
N_MESES = 12  # Número de meses hacia el futuro que deseas predecir
CARPETA_MODELOS = 'models'  # Carpeta donde están los modelos
MIN_CLIENTES_PREDICCION = 5  # Mínimo número de clientes a predecir
MIN_PRODUCTOS_PREDICCION = 5  # Mínimo número de productos a predecir

# Parámetros para reglas de asociación
MIN_SUPPORT = 0.01  # Soporte mínimo para itemsets frecuentes
MIN_LIFT = 1.0      # Lift mínimo para reglas de asociación

# --- Detalles de conexión a tu base de datos PostgreSQL local ---
db_user = 'postgres'
db_password = 'postgres'
db_host = 'localhost'
db_port = '5433'
db_name = 'Tipvos' # Nombre de la base de datos

# Construir la cadena de conexión para PostgreSQL local
engine_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(engine_string)

print("=== INICIANDO SISTEMA DE PREDICCIÓN Y ANÁLISIS (TIPVOS) ===")
print(f"📅 Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- CARGA DE DATOS ---
print("\n--- Cargando Datos Históricos desde PostgreSQL Local ---")
try:
    df_ventas = pd.read_sql("SELECT * FROM fact_ventas", engine)
    df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
    df_ventas = df_ventas[(df_ventas['cantidad'] > 0) & (df_ventas['precio_unitario'] > 0)]
    df_ventas['monto_venta'] = df_ventas['cantidad'] * df_ventas['precio_unitario']
    df_ventas['año_mes'] = df_ventas['fecha'].dt.to_period('M')

    ultima_fecha_historica_global = df_ventas['fecha'].max()
    fecha_inicio_predicciones_global = (ultima_fecha_historica_global + relativedelta(months=1)).replace(day=1)
    print(f"✅ Datos de ventas cargados: {len(df_ventas)} registros")

    # Agrupación mensual por producto
    df_mensual_productos = df_ventas.groupby(['stockcode', 'año_mes'])['cantidad'].sum().reset_index()
    df_mensual_productos['año_mes'] = df_mensual_productos['año_mes'].dt.to_timestamp()
    print(f"✅ Datos mensuales por producto preparados.")

    # Obtener top productos por cantidad total (para predicciones alternativas)
    top_productos_por_cantidad = df_ventas.groupby('stockcode')['cantidad'].sum().nlargest(30).index.tolist()
    print(f"✅ Top productos por cantidad identificados.")

    # Agrupación mensual por cliente (para montos)
    df_mensual_clientes = df_ventas.groupby(['customer_id', 'año_mes'])['monto_venta'].sum().reset_index()
    df_mensual_clientes['año_mes'] = df_mensual_clientes['año_mes'].dt.to_timestamp()
    print(f"✅ Datos mensuales por cliente preparados.")

    # Obtener top clientes por monto total (para predicciones alternativas)
    top_clientes_por_monto = df_ventas.groupby('customer_id')['monto_venta'].sum().nlargest(20).index.tolist()
    print(f"✅ Top clientes por monto identificados.")

except Exception as e:
    print(f"❌ Error al cargar o preparar datos históricos: {e}")
    print("Asegúrate de que la base de datos PostgreSQL local esté accesible y la tabla 'fact_ventas' exista.")
    raise # Detener la ejecución si los datos no se pueden cargar

# --- SELECCIÓN DE TIPO DE PREDICCIÓN ---
print("\n=== SELECTOR DE PREDICCIONES ===")
print("Selecciona qué tipo de predicción/análisis deseas realizar:")
print("1. Solo predicciones de productos (cantidades)")
print("2. Solo predicciones de clientes (montos)")
print("3. Solo análisis de reglas de asociación (recomendaciones)")
print("4. Solo segmentación de clientes (clustering)")
print("5. Productos + Clientes")
print("6. Productos + Reglas de asociación")
print("7. Clientes + Reglas de asociación")
print("8. Productos + Clustering")
print("9. Clientes + Clustering")
print("10. Reglas de asociación + Clustering")
print("11. Productos + Clientes + Clustering")
print("12. Todas las opciones")

while True:
    try:
        opcion = int(input("\nIngresa tu opción (1-12): "))
        if opcion in range(1, 13):
            break
        else:
            print("❌ Por favor ingresa un número entre 1 y 12")
    except ValueError:
        print("❌ Por favor ingresa un número válido")

realizar_productos = opcion in [1, 5, 6, 8, 11, 12]
realizar_clientes = opcion in [2, 5, 7, 9, 11, 12]
realizar_asociaciones = opcion in [3, 6, 7, 10, 12]
realizar_clustering = opcion in [4, 8, 9, 10, 11, 12]

print(f"\n=== CONFIGURACIÓN SELECCIONADA ===")
print(f"📦 Predicciones de productos: {'SÍ' if realizar_productos else 'NO'}")
print(f"👥 Predicciones de clientes: {'SÍ' if realizar_clientes else 'NO'}")
print(f"🔗 Reglas de asociación: {'SÍ' if realizar_asociaciones else 'NO'}")
print(f"🎯 Segmentación de clientes: {'SÍ' if realizar_clustering else 'NO'}")
print(f"📅 Meses a predecir: {N_MESES}")
print(f"💾 Modo de inserción: LIMPIAR FUTURAS (automático)")

input("\nPresiona Enter para continuar...")


# --- LIMPIAR DATOS FUTUROS (AUTOMÁTICO) ---
print("\n--- Limpiando Predicciones Futuras en PostgreSQL Local ---")
# Usar la fecha fija para la demo
ultima_fecha_historica = df_ventas['fecha'].max().date()
# La predicción debe comenzar DESPUÉS de la última fecha histórica.
# Si quieres borrar todo lo que está más allá de tu último dato histórico, usa esa fecha.
# Si quieres borrar TODAS las predicciones existentes para regenerarlas, puedes usar una fecha muy antigua.
# Para este escenario, queremos borrar las predicciones que no corresponden al periodo futuro real.
# Vamos a borrar todo lo que sea posterior o igual al inicio del primer mes que deberías predecir.
fecha_inicio_prediccion_deseada = '2009-01-01'  # Cambia esto si quieres un inicio diferente


with engine.begin() as conn:
    if realizar_productos:
        try:
            # Eliminar todas las predicciones a partir del inicio del primer mes de predicción
            result = conn.execute(text("DELETE FROM predicciones_mensuales WHERE fecha_prediccion >= :fecha"), {"fecha": fecha_inicio_prediccion_deseada})
            print(f"🧹 Eliminadas {result.rowcount} predicciones de productos a partir de {fecha_inicio_prediccion_deseada}.")
        except Exception as e:
            print(f"❌ Error al limpiar 'predicciones_mensuales': {e}. Asegúrate de que la tabla existe.")
    
    if realizar_clientes:
        try:
            # Eliminar todas las predicciones a partir del inicio del primer mes de predicción
            result = conn.execute(text("DELETE FROM predicciones_montos_clientes WHERE fecha_prediccion >= :fecha"), {"fecha": fecha_inicio_prediccion_deseada})
            print(f"🧹 Eliminadas {result.rowcount} predicciones de clientes a partir de {fecha_inicio_prediccion_deseada}.")
        except Exception as e:
            print(f"❌ Error al limpiar 'predicciones_montos_clientes': {e}. Asegúrate de que la tabla existe.")
    
    if realizar_asociaciones:
        try:
            # Primero: Eliminar la vista dependiente si existe (con CASCADE para mayor seguridad si hubiera más dependencias)
            conn.execute(text("DROP VIEW IF EXISTS vista_top_reglas_asociacion CASCADE;"))
            print("🧹 Eliminada la vista 'vista_top_reglas_asociacion' si existía.")
            
            # Segundo: Eliminar la tabla de reglas de asociación si existe (para un reemplazo limpio con to_sql)
            conn.execute(text("DROP TABLE IF EXISTS reglas_asociacion;")) # CAMBIO CLAVE AQUÍ
            print("🧹 Eliminada la tabla 'reglas_asociacion' si existía.")
            
            # El DELETE FROM ya no es necesario aquí si usas if_exists='replace' en to_sql,
            # ya que el DROP TABLE lo manejará. Si no estás usando 'replace', entonces mantendrías el DELETE.
            # Pero para el flujo que hemos discutido, el DROP TABLE es lo correcto.
            
        except Exception as e:
            print(f"ℹ️ No se pudo limpiar 'reglas_asociacion' o su vista: {e}. Asegúrate de que tengas permisos.")
    
    if realizar_clustering:
        try:
            result = conn.execute(text("DELETE FROM segmentacion_clientes WHERE fecha_actualizacion >= :fecha"), {"fecha": fecha_referencia_demo})
            print(f"🧹 Eliminadas {result.rowcount} segmentaciones anteriores.")
        except Exception as e:
            print(f"ℹ️ No se pudo limpiar 'segmentacion_clientes': {e}. Asegúrate de que la tabla existe.")

print("\n=== INICIANDO PREDICCIONES Y ANÁLISIS ===")

# --- 1. PREDICCIONES DE PRODUCTOS (CANTIDADES) ---
if realizar_productos:
    print("\n--- Procesando Predicciones de Productos ---")
    predicciones_productos = []
    productos_con_modelo = []

    # Primero: Procesar productos con modelos ARIMA entrenados
    for archivo in os.listdir(CARPETA_MODELOS):
        if archivo.startswith('arima_model_') and archivo.endswith('.pkl'):
            stockcode = archivo.split('_')[-1].replace('.pkl', '')
            modelo_path = os.path.join(CARPETA_MODELOS, archivo)
            
            try:
                model = joblib.load(modelo_path)
                
                # Serie histórica del producto
                serie = df_mensual_productos[df_mensual_productos['stockcode'] == stockcode].set_index('año_mes').sort_index()
                
                if serie.empty:
                    print(f"⚠️ No hay datos históricos para producto {stockcode}")
                    continue
                    
                serie = serie.reindex(pd.date_range(start=serie.index.min(), end=serie.index.max(), freq='MS'), fill_value=0)
                ts_log = np.log1p(serie['cantidad'])
                
                # Predicción múltiple
                forecast = model.predict(start=len(ts_log), end=len(ts_log) + N_MESES - 1, dynamic=False)
                
                for i, val in enumerate(forecast):
                    pred = np.expm1(val)
                    pred = max(0, round(pred))  # evitar negativos
                    fecha_prediccion = serie.index[-1] + relativedelta(months=i + 1)
                    
                    predicciones_productos.append({
                        'stockcode': stockcode,
                        'fecha_prediccion': fecha_prediccion.date(),
                        'cantidad_predicha': pred
                    })
                    
                productos_con_modelo.append(stockcode)
                print(f"✅ Producto {stockcode}: {N_MESES} predicciones generadas (ARIMA)")
                
            except Exception as e:
                print(f"❌ Error con producto {stockcode}: {e}")

    # Segundo: Si no tenemos suficientes productos, agregar predicciones con promedio histórico
    if len(productos_con_modelo) < MIN_PRODUCTOS_PREDICCION:
        productos_faltantes = MIN_PRODUCTOS_PREDICCION - len(productos_con_modelo)
        print(f"\n⚠️ Solo se encontraron {len(productos_con_modelo)} modelos de productos.")
        print(f"Agregando {productos_faltantes} predicciones adicionales usando promedio histórico...")
        
        # Seleccionar productos adicionales de los top productos que no tengan modelo
        productos_adicionales = [p for p in top_productos_por_cantidad if p not in productos_con_modelo][:productos_faltantes]
        
        for stockcode in productos_adicionales:
            try:
                # Serie histórica del producto
                serie = df_mensual_productos[df_mensual_productos['stockcode'] == stockcode].set_index('año_mes').sort_index()
                
                if serie.empty:
                    continue
                
                # Calcular promedio móvil de los últimos 6 meses (o todos los disponibles)
                ultimo_periodo = serie.tail(6)['cantidad']
                promedio_producto = ultimo_periodo.mean()
                
                if not np.isfinite(promedio_producto) or promedio_producto <= 0:
                    promedio_producto = serie['cantidad'].mean()
                    if not np.isfinite(promedio_producto) or promedio_producto <= 0:
                        continue
                
                # Generar predicciones para N_MESES
                ultima_fecha = serie.index[-1]
                for i in range(N_MESES):
                    fecha_prediccion = ultima_fecha + relativedelta(months=i + 1)
                    
                    # Aplicar una pequeña variación aleatoria al promedio (±15%)
                    variacion = np.random.uniform(0.85, 1.15)
                    pred = max(0, round(promedio_producto * variacion))
                    
                    predicciones_productos.append({
                        'stockcode': stockcode,
                        'fecha_prediccion': fecha_prediccion.date(),
                        'cantidad_predicha': pred
                    })
                
                print(f"✅ Producto {stockcode}: {N_MESES} predicciones generadas (Promedio Histórico)")
                
            except Exception as e:
                print(f"❌ Error con producto adicional {stockcode}: {e}")

    print(f"\n📊 Total de productos con predicciones: {len(set([p['stockcode'] for p in predicciones_productos]))}")

else:
    predicciones_productos = []
    print("⏭️ Saltando predicciones de productos...")

# --- 2. PREDICCIONES DE CLIENTES (MONTOS) ---
if realizar_clientes:
    print("\n--- Procesando Predicciones de Clientes ---")
    predicciones_clientes = []
    clientes_con_modelo = []

    # Primero: Procesar clientes con modelos ARIMA entrenados
    for archivo in os.listdir(CARPETA_MODELOS):
        if archivo.startswith('arima_cliente_') and archivo.endswith('.pkl'):
            customer_id = archivo.split('_')[-1].replace('.pkl', '')
            modelo_path = os.path.join(CARPETA_MODELOS, archivo)
            
            try:
                model = joblib.load(modelo_path)
                
                # Serie histórica del cliente
                serie = df_mensual_clientes[df_mensual_clientes['customer_id'] == int(customer_id)].set_index('año_mes').sort_index()
                
                if serie.empty:
                    print(f"⚠️ No hay datos históricos para cliente {customer_id}")
                    continue
                    
                serie = serie.reindex(pd.date_range(start=serie.index.min(), end=serie.index.max(), freq='MS'), fill_value=0)
                ts_log = np.log1p(serie['monto_venta'])
                
                # Predicción múltiple
                forecast = model.predict(start=len(ts_log), end=len(ts_log) + N_MESES - 1, dynamic=False)
                
                for i, val in enumerate(forecast):
                    pred = np.expm1(val)
                    pred = max(0, round(pred, 2))  # evitar negativos y redondear a 2 decimales
                    fecha_prediccion = serie.index[-1] + relativedelta(months=i + 1)
                    
                    # Validación adicional para montos
                    if not np.isfinite(pred) or pred > serie['monto_venta'].max() * 3:
                        pred = serie['monto_venta'].mean()
                        if not np.isfinite(pred) or pred < 0:
                            pred = 0
                        pred = round(pred, 2)
                    
                    predicciones_clientes.append({
                        'client_id': customer_id,
                        'fecha_prediccion': fecha_prediccion.date(),
                        'monto_predicho': pred
                    })
                    
                clientes_con_modelo.append(int(customer_id))
                print(f"✅ Cliente {customer_id}: {N_MESES} predicciones generadas (ARIMA)")
                
            except Exception as e:
                print(f"❌ Error con cliente {customer_id}: {e}")

    # Segundo: Si no tenemos suficientes clientes, agregar predicciones con promedio histórico
    if len(clientes_con_modelo) < MIN_CLIENTES_PREDICCION:
        clientes_faltantes = MIN_CLIENTES_PREDICCION - len(clientes_con_modelo)
        print(f"\n⚠️ Sólo se encontraron {len(clientes_con_modelo)} modelos de clientes.")
        print(f"Agregando {clientes_faltantes} predicciones adicionales usando promedio histórico...")
        
        # Seleccionar clientes adicionales de los top clientes que no tengan modelo
        clientes_adicionales = [c for c in top_clientes_por_monto if c not in clientes_con_modelo][:clientes_faltantes]
        
        for customer_id in clientes_adicionales:
            try:
                # Serie histórica del cliente
                serie = df_mensual_clientes[df_mensual_clientes['customer_id'] == customer_id].set_index('año_mes').sort_index()
                
                if serie.empty:
                    continue
                
                # Calcular promedio móvil de los últimos 6 meses (o todos los disponibles)
                ultimo_periodo = serie.tail(6)['monto_venta']
                promedio_cliente = ultimo_periodo.mean()
                
                if not np.isfinite(promedio_cliente) or promedio_cliente <= 0:
                    promedio_cliente = serie['monto_venta'].mean()
                    if not np.isfinite(promedio_cliente) or promedio_cliente <= 0:
                        continue
                
                # Generar predicciones para N_MESES
                ultima_fecha = serie.index[-1]
                for i in range(N_MESES):
                    fecha_prediccion = ultima_fecha + relativedelta(months=i + 1)
                    
                    # Aplicar una pequeña variación aleatoria al promedio (±10%)
                    variacion = np.random.uniform(0.9, 1.1)
                    pred = round(promedio_cliente * variacion, 2)
                    
                    predicciones_clientes.append({
                        'client_id': str(customer_id),
                        'fecha_prediccion': fecha_prediccion.date(),
                        'monto_predicho': pred
                    })
                
                print(f"✅ Cliente {customer_id}: {N_MESES} predicciones generadas (Promedio Histórico)")
                
            except Exception as e:
                print(f"❌ Error con cliente adicional {customer_id}: {e}")

    print(f"\n📊 Total de clientes con predicciones: {len(set([p['client_id'] for p in predicciones_clientes]))}")

else:
    predicciones_clientes = []
    print("⏭️ Saltando predicciones de clientes...")

# --- 3. ANÁLISIS DE REGLAS DE ASOCIACIÓN ---
reglas_asociacion = [] # Cambiamos el nombre para evitar confusión

if realizar_asociaciones:
    print("\n--- Procesando Reglas de Asociación ---")
    
    try:
        # Cargar información de productos para descripciones (Asegúrate de que 'descripcion' esté disponible)
        df_productos = pd.read_sql("SELECT stockcode, descripcion FROM dim_producto", engine)
        product_description_map = df_productos.set_index('stockcode')['descripcion'].to_dict()

        modelo_path = os.path.join(CARPETA_MODELOS, 'market_basket_model.pkl')
        
        if os.path.exists(modelo_path):
            print("📂 Cargando modelo de reglas de asociación existente...")
            modelo_asociacion = joblib.load(modelo_path)
            rules = modelo_asociacion['association_rules']
            print(f"✅ Modelo cargado: {len(rules)} reglas encontradas")
            
        else:
            print("🔧 Generando nuevas reglas de asociación (no se encontró modelo pre-entrenado)...")
            # ... (tu código para generar rules usando apriori y association_rules) ...
            # Asegúrate de que 'rules' sea un DataFrame de pandas aquí

        # Preparar reglas para subir a base de datos
        if len(rules) > 0:
            print("    Preparando reglas para la base de datos...")
            
            # Filtrar solo reglas con un antecedente y un consecuente para simplificar el Power BI
            # O manejar como listas si quieres mostrarlas en Power BI, pero la relación con dim_producto será difícil.
            # Para la plantilla de Power BI, lo más fácil es 1 a 1.
            filtered_rules = rules[
                (rules['antecedents'].apply(lambda x: len(x) == 1)) & 
                (rules['consequents'].apply(lambda x: len(x) == 1))
            ]

            # Tomar un número razonable de reglas, por ejemplo, las 500 más fuertes por lift y confianza
            filtered_rules = filtered_rules.sort_values(['lift', 'confidence'], ascending=[False, False]).head(2000)
            regla_id_counter = 1
            for idx, rule in filtered_rules.iterrows():
                antecedent_id = list(rule['antecedents'])[0]
                consequent_id = list(rule['consequents'])[0]
                
                reglas_asociacion.append({
                    'regla_id': regla_id_counter,
                    'antecedent_product_id': antecedent_id,
                    'antecedent_product_description': product_description_map.get(antecedent_id, f"Desconocido ({antecedent_id})"),
                    'consequent_product_id': consequent_id,
                    'consequent_product_description': product_description_map.get(consequent_id, f"Desconocido ({consequent_id})"),
                    'support': round(float(rule['support']), 4),
                    'confidence': round(float(rule['confidence']), 4),
                    'lift': round(float(rule['lift']), 4),
                    'fecha_generacion': datetime.now().date()
                })
            
            print(f"✅ {len(reglas_asociacion)} reglas preparadas para subir.")
        else:
            print("⚠️ No hay reglas de asociación para subir.")
            
    except Exception as e:
        print(f"❌ Error en análisis de reglas de asociación: {e}")
        reglas_asociacion_para_db = []

else:
    print("⏭️ Saltando análisis de reglas de asociación...")

# --- 4. SEGMENTACIÓN DE CLIENTES (CLUSTERING) ---
segmentacion_clientes = []

if realizar_clustering:
    print("\n--- Procesando Segmentación de Clientes ---")
    
    try:
        # Cargar modelo de clustering y scaler
        modelo_kmeans_path = os.path.join(CARPETA_MODELOS, 'kmeans_model_latest.pkl')
        scaler_path = os.path.join(CARPETA_MODELOS, 'scaler_latest.pkl')
        metadata_path = os.path.join(CARPETA_MODELOS, 'model_metadata_latest.pkl')
        
        if os.path.exists(modelo_kmeans_path) and os.path.exists(scaler_path):
            print("📂 Cargando modelo de clustering existente...")
            
            # Cargar componentes del modelo
            modelo_kmeans = joblib.load(modelo_kmeans_path)
            scaler = joblib.load(scaler_path)
            
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                print(f"✅ Modelo cargado - K={metadata.get('best_k', 'N/A')}, Score={metadata.get('silhouette_score', 'N/A'):.3f}")
            else:
                print("✅ Modelo cargado (sin metadatos específicos).")
            
            # Calcular métricas RFM actualizadas para todos los clientes
            print("   Calculando métricas RFM actualizadas para todos los clientes...")
            # Usar la fecha fija de la demo para los cálculos RFM
            fecha_ref = datetime(2011, 12, 1) + pd.Timedelta(days=1)
            
            rfm_actual = df_ventas.groupby("customer_id").agg({
                "fecha": lambda x: (fecha_ref - x.max()).days,   # Recencia
                "invoice": "nunique",                            # Frecuencia
                "monto_venta": "sum"                             # Monto
            }).reset_index()
            
            rfm_actual.columns = ["customer_id", "recencia", "frecuencia", "monto"]
            
            # Normalizar usando el scaler entrenado
            rfm_scaled = scaler.transform(rfm_actual[["recencia", "frecuencia", "monto"]])
            
            # Predecir clusters
            clusters = modelo_kmeans.predict(rfm_scaled)
            rfm_actual['cluster'] = clusters
            
            # Calcular información adicional para Power BI
            print("   Preparando datos detallados por cliente para Power BI...")
            
            # Obtener información adicional de cada cliente
            info_clientes = df_ventas.groupby('customer_id').agg(
                primera_compra=('fecha', 'min'),    # Primera compra
                ultima_compra=('fecha', 'max'),     # Última compra
                total_ordenes=('invoice', 'nunique'), # Número de órdenes
                total_productos=('cantidad', 'sum'), # Productos totales comprados
                monto_total=('monto_venta', 'sum'),
                monto_promedio=('monto_venta', 'mean'),
                monto_std=('monto_venta', 'std')
            ).reset_index()
            
            # Calcular días desde primera compra
            info_clientes['dias_como_cliente'] = (fecha_ref - info_clientes['primera_compra']).dt.days

            # Obtener país del cliente (desde dim_cliente)
            try:
                df_paises = pd.read_sql("SELECT customer_id, pais FROM dim_cliente", engine)
                info_clientes = info_clientes.merge(df_paises, on='customer_id', how='left')
            except Exception as e:
                print(f"⚠️ No se pudo cargar dim_cliente para obtener países: {e}. País se establecerá como 'Unknown'.")
                info_clientes['pais'] = 'Unknown'
            
            # Combinar con RFM y clusters
            rfm_completo = rfm_actual.merge(info_clientes, on='customer_id', how='left')
            
            # Asignar etiquetas de clusters más inteligentes basadas en las métricas
            cluster_stats = rfm_actual.groupby('cluster').agg({
                'recencia': 'mean',
                'frecuencia': 'mean',
                'monto': 'mean'
            }).round(2)
            
            print("   Estadísticas promedio por cluster:")
            print(cluster_stats)
            
            # Lógica mejorada para asignar etiquetas, basándose en la percentiles globales o en la interpretación de los clusters entrenados
            def get_cluster_label(row_stats):
                recency_threshold_high = rfm_actual['recencia'].quantile(0.75)
                recency_threshold_low = rfm_actual['recencia'].quantile(0.25)
                frequency_threshold_high = rfm_actual['frecuencia'].quantile(0.75)
                monetary_threshold_high = rfm_actual['monto'].quantile(0.75)

                if row_stats['monto'] > monetary_threshold_high and row_stats['frecuencia'] > frequency_threshold_high:
                    return "Clientes VIP"
                elif row_stats['recencia'] > recency_threshold_high and row_stats['frecuencia'] < frequency_threshold_high:
                    return "Clientes en Riesgo de Fuga"
                elif row_stats['recencia'] <= recency_threshold_low and row_stats['frecuencia'] > rfm_actual['frecuencia'].median():
                    return "Clientes Activos y Frecuentes"
                elif row_stats['monto'] < rfm_actual['monto'].quantile(0.25):
                    return "Clientes de Bajo Valor"
                elif row_stats['recencia'] > recency_threshold_high:
                    return "Clientes Inactivos"
                else:
                    return "Clientes Regulares"

            cluster_labels_inteligentes = {
                cluster_id: get_cluster_label(stats)
                for cluster_id, stats in cluster_stats.iterrows()
            }
            
            # Preparar datos para subir a base de datos
            for idx, row in rfm_completo.iterrows():
                cluster_id = int(row['cluster'])
                cluster_label = cluster_labels_inteligentes.get(cluster_id, f"Cluster {cluster_id}")
                
                segmentacion_clientes.append({
                    'customer_id': int(row['customer_id']),
                    'cluster_id': cluster_id,
                    'cluster_nombre': cluster_label,
                    'recencia': int(row['recencia']),
                    'frecuencia': int(row['frecuencia']),
                    'monto_total': round(float(row['monto']), 2),
                    'primera_compra': row.get('primera_compra', pd.NaT).date() if pd.notna(row.get('primera_compra')) else None,
                    'ultima_compra': row.get('ultima_compra', pd.NaT).date() if pd.notna(row.get('ultima_compra')) else None,
                    'total_ordenes': int(row.get('total_ordenes', 0)),
                    'total_productos': int(row.get('total_productos', 0)),
                    'monto_promedio': round(float(row.get('monto_promedio', 0)), 2),
                    'dias_como_cliente': int(row.get('dias_como_cliente', 0)),
                    'pais': str(row.get('pais', 'Unknown')),
                    'fecha_actualizacion': datetime.now().date(),
                    'activo': True
                })
            
            print(f"✅ {len(segmentacion_clientes)} clientes segmentados y preparados para subir.")
            
            # Mostrar distribución de clusters
            distribucion = rfm_actual['cluster'].value_counts().sort_index()
            print("\n   Distribución de clientes por cluster (con etiquetas):")
            for cluster_id, count in distribucion.items():
                label = cluster_labels_inteligentes.get(cluster_id, f"Cluster {cluster_id}")
                porcentaje = (count / len(rfm_actual)) * 100
                print(f"   {label} (ID: {cluster_id}): {count} clientes ({porcentaje:.1f}%)")
                
        else:
            print("❌ No se encontró modelo de clustering entrenado ('kmeans_model_latest.pkl' o 'scaler_latest.pkl').")
            print("💡 Por favor, ejecuta primero el script de entrenamiento de clustering para generar estos archivos.")
            
    except Exception as e:
        print(f"❌ Error crítico en segmentación de clientes: {e}")
        segmentacion_clientes = []

else:
    print("⏭️ Saltando segmentación de clientes...")

# --- 5. SUBIR RESULTADOS A POSTGRESQL LOCAL ---
print("\n--- Subiendo Resultados a PostgreSQL Local ---")



if realizar_productos:
    MAX_CANTIDAD = 1_00000  # Límite razonable para cantidades predichas
    # Filtrar predicciones antes de subir
    df_pred_productos = pd.DataFrame(predicciones_productos)
    if not df_pred_productos.empty:
        df_pred_productos = df_pred_productos[
            (df_pred_productos['cantidad_predicha'] >= 0) &
            (df_pred_productos['cantidad_predicha'] <= MAX_CANTIDAD)
        ]

        # Opcional: Reemplazar valores fuera de rango por la media histórica o por 0
        df_pred_productos.loc[df_pred_productos['cantidad_predicha'] > MAX_CANTIDAD, 'cantidad_predicha'] = 0

        for idx, row in df_pred_productos[df_pred_productos['cantidad_predicha'] > MAX_CANTIDAD].iterrows():
            stockcode = row['stockcode']
            # Calcular la media histórica del producto
            media_historica = df_mensual_productos[df_mensual_productos['stockcode'] == stockcode]['cantidad'].mean()
            # Si la media no es válida, usa 0
            if not np.isfinite(media_historica) or media_historica < 0:
                media_historica = 0
            df_pred_productos.at[idx, 'cantidad_predicha'] = round(media_historica)

        try:
            df_pred_productos.to_sql('predicciones_mensuales', engine, if_exists='append', index=False)
            print(f"✅ {len(df_pred_productos)} predicciones de productos insertadas para {N_MESES} meses.")
        except Exception as e:
            print(f"❌ Error al subir predicciones de productos: {e}")
            print("💡 Asegúrate de que la tabla 'predicciones_mensuales' existe y tiene las columnas correctas.")
    else:
        print("⚠️ No se generaron predicciones de productos para subir.")

# Subir predicciones de clientes
if realizar_clientes and predicciones_clientes:
    try:
        df_pred_clientes = pd.DataFrame(predicciones_clientes)
        df_pred_clientes.to_sql('predicciones_montos_clientes', engine, if_exists='append', index=False)
        print(f"✅ {len(df_pred_clientes)} predicciones de montos de clientes insertadas para {N_MESES} meses.")
    except Exception as e:
        print(f"❌ Error al subir predicciones de clientes: {e}")
        print("💡 Asegúrate de que la tabla 'predicciones_montos_clientes' existe y tiene las columnas correctas.")
elif realizar_clientes:
    print("⚠️ No se generaron predicciones de clientes para subir.")

# Subir reglas de asociación
if realizar_asociaciones and reglas_asociacion:
    try:
        df_reglas_asociacion = pd.DataFrame(reglas_asociacion)
        # Asegúrate de que las columnas coincidan con la tabla de DB.
        # Creamos la tabla si no existe o la sobrescribimos para cada ejecución.
        df_reglas_asociacion.to_sql('reglas_asociacion', engine, if_exists='replace', index=False) 
        print(f"✅ {len(df_reglas_asociacion)} reglas de asociación insertadas/actualizadas.")
    except Exception as e:
        print(f"❌ Error al subir reglas de asociación: {e}")
        print("💡 Asegúrate de que la tabla 'reglas_asociacion' pueda ser creada/sobrescrita y tenga las columnas correctas.")
else:
    print("⚠️ No se generaron reglas de asociación para subir.")

# Subir segmentación de clientes
if realizar_clustering and segmentacion_clientes:
    try:
        df_segmentacion = pd.DataFrame(segmentacion_clientes)
        df_segmentacion.to_sql('segmentacion_clientes', engine, if_exists='append', index=False)
        print(f"✅ {len(df_segmentacion)} segmentaciones de clientes insertadas.")
        
        # Crear tabla resumen por cluster para Power BI
        resumen_clusters = df_segmentacion.groupby(['cluster_id', 'cluster_nombre']).agg(
            total_clientes=('customer_id', 'count'),
            recencia_promedio=('recencia', 'mean'),
            frecuencia_promedio=('frecuencia', 'mean'),
            monto_promedio=('monto_total', 'mean'),
            monto_total_cluster=('monto_total', 'sum'),
            ordenes_promedio=('total_ordenes', 'mean'),
            dias_promedio=('dias_como_cliente', 'mean')
        ).round(2).reset_index()
        
        resumen_clusters['fecha_actualizacion'] = datetime.now().date()
        resumen_clusters['porcentaje_clientes'] = (resumen_clusters['total_clientes'] / len(df_segmentacion) * 100).round(2)
        
        # Subir resumen de clusters (usar 'replace' para esta tabla, ya que es un resumen)
        resumen_clusters.to_sql('resumen_clusters', engine, if_exists='replace', index=False)
        print(f"✅ Resumen de {len(resumen_clusters)} clusters actualizado para Power BI.")
        
    except Exception as e:
        print(f"❌ Error al subir segmentación o resumen de clientes: {e}")
        print("💡 Asegúrate de que las tablas 'segmentacion_clientes' y 'resumen_clusters' existen y tienen las columnas correctas.")
elif realizar_clustering:
    print("⚠️ No se generaron segmentaciones de clientes para subir.")

# --- 6. CREAR VISTAS ADICIONALES PARA POWER BI ---
print("\n--- Creando/Actualizando Vistas Adicionales para Power BI en PostgreSQL ---")

try:
    with engine.begin() as conn:
        # Vista consolidada de métricas por cliente
        conn.execute(text("""
        CREATE OR REPLACE VIEW vista_metricas_clientes AS
        SELECT 
            s.customer_id,
            s.cluster_nombre,
            s.recencia,
            s.frecuencia,
            s.monto_total,
            s.total_ordenes,
            s.dias_como_cliente,
            s.pais,
            CASE 
                WHEN s.recencia <= 30 THEN 'Activo'
                WHEN s.recencia <= 90 THEN 'En Riesgo'
                ELSE 'Inactivo'
            END AS estado_actividad,
            CASE
                WHEN s.monto_total >= (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY monto_total) FROM segmentacion_clientes) THEN 'Alto Valor'
                WHEN s.monto_total >= (SELECT PERCENTILE_CONT(0.4) WITHIN GROUP (ORDER BY monto_total) FROM segmentacion_clientes) THEN 'Valor Medio'
                ELSE 'Bajo Valor'
            END AS etiqueta_valor_monto
        FROM segmentacion_clientes s;
        """))
        print("✅ Vista 'vista_metricas_clientes' creada/actualizada.")

        # Vista de las principales reglas de asociación (ejemplo, top 100)
        conn.execute(text("""
         CREATE OR REPLACE VIEW vista_top_reglas_asociacion AS
            SELECT
                regla_id,                         -- Asegúrate de que exista en la tabla
                antecedent_product_id,
                antecedent_product_description,   -- Nuevo campo
                consequent_product_id,
                consequent_product_description,   -- Nuevo campo
                support,
                confidence,
                lift,
                fecha_generacion
            FROM reglas_asociacion
            -- WHERE activa = TRUE -- Solo si has añadido y gestionas la columna 'activa'
            ORDER BY lift DESC, confidence DESC
            LIMIT 2000;
        """))
        print("✅ Vista 'vista_top_reglas_asociacion' creada/actualizada.")

except Exception as e:
    print(f"❌ Error al crear/actualizar vistas para Power BI: {e}")
    print("💡 Asegúrate de tener los permisos necesarios para crear/reemplazar vistas en tu base de datos.")

print("\n=== PROCESO COMPLETADO ===")
print("Todas las operaciones seleccionadas han finalizado.")