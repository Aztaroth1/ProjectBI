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
MIN_LIFT = 1.0     # Lift mínimo para reglas de asociación

# --- CONEXIÓN A SUPABASE ---
engine = create_engine("postgresql://postgres.wrwpkkyeukjuisjlbihn:postgres@aws-0-us-east-2.pooler.supabase.com:6543/postgres")

# --- CARGA DE DATOS ---
df_ventas = pd.read_sql("SELECT * FROM fact_ventas", engine)
df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
df_ventas = df_ventas[(df_ventas['cantidad'] > 0) & (df_ventas['precio_unitario'] > 0)]
df_ventas['monto_venta'] = df_ventas['cantidad'] * df_ventas['precio_unitario']
df_ventas['año_mes'] = df_ventas['fecha'].dt.to_period('M')

# Agrupación mensual por producto
df_mensual_productos = df_ventas.groupby(['stockcode', 'año_mes'])['cantidad'].sum().reset_index()
df_mensual_productos['año_mes'] = df_mensual_productos['año_mes'].dt.to_timestamp()

# Obtener top productos por cantidad total (para predicciones alternativas)
top_productos_por_cantidad = df_ventas.groupby('stockcode')['cantidad'].sum().nlargest(30).index.tolist()

# Agrupación mensual por cliente (para montos)
df_mensual_clientes = df_ventas.groupby(['customer_id', 'año_mes'])['monto_venta'].sum().reset_index()
df_mensual_clientes['año_mes'] = df_mensual_clientes['año_mes'].dt.to_timestamp()

# Obtener top clientes por monto total (para predicciones alternativas)
top_clientes_por_monto = df_ventas.groupby('customer_id')['monto_venta'].sum().nlargest(20).index.tolist()

# --- SELECCIÓN DE TIPO DE PREDICCIÓN ---
print("=== SELECTOR DE PREDICCIONES ===")
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
print("\n--- Limpiando Predicciones Futuras ---")
fecha_actual = '2011-12-01'

with engine.begin() as conn:
    if realizar_productos:
        result = conn.execute(text("DELETE FROM predicciones_mensuales WHERE fecha_prediccion >= :fecha"), {"fecha": fecha_actual})
        print(f"🧹 Eliminadas {result.rowcount} predicciones futuras de productos")
    
    if realizar_clientes:
        result = conn.execute(text("DELETE FROM predicciones_montos_clientes WHERE fecha_prediccion >= :fecha"), {"fecha": fecha_actual})
        print(f"🧹 Eliminadas {result.rowcount} predicciones futuras de clientes")
    
    if realizar_asociaciones:
        try:
            result = conn.execute(text("DELETE FROM reglas_asociacion WHERE fecha_generacion >= :fecha"), {"fecha": fecha_actual})
            print(f"🧹 Eliminadas {result.rowcount} reglas de asociación anteriores")
        except Exception as e:
            print(f"ℹ️ No se pudo limpiar reglas de asociación: {e}")
    
    if realizar_clustering:
        try:
            result = conn.execute(text("DELETE FROM segmentacion_clientes WHERE fecha_actualizacion >= :fecha"), {"fecha": fecha_actual})
            print(f"🧹 Eliminadas {result.rowcount} segmentaciones anteriores")
        except Exception as e:
            print(f"ℹ️ No se pudo limpiar segmentaciones: {e}")

print("\n=== INICIANDO PREDICCIONES ===")

# --- 1. PREDICCIONES DE PRODUCTOS (CANTIDADES) ---
if realizar_productos:
    print("\n--- Procesando Predicciones de Productos ---")
    predicciones_productos = []
    productos_con_modelo = []

    # Primero: Procesar productos con modelos ARIMA entrenados
    for archivo in os.listdir(CARPETA_MODELOS):
        if archivo.startswith('arima_producto_') and archivo.endswith('.pkl'):
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
        print(f"\n⚠️ Solo se encontraron {len(clientes_con_modelo)} modelos de clientes.")
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
reglas_asociacion = []

if realizar_asociaciones:
    print("\n--- Procesando Reglas de Asociación ---")
    
    try:
        # Cargar modelo de reglas de asociación si existe
        modelo_path = os.path.join(CARPETA_MODELOS, 'market_basket_model.pkl')
        
        if os.path.exists(modelo_path):
            print("📂 Cargando modelo de reglas de asociación existente...")
            modelo_asociacion = joblib.load(modelo_path)
            rules = modelo_asociacion['association_rules']
            print(f"✅ Modelo cargado: {len(rules)} reglas encontradas")
            
        else:
            print("🔧 Generando nuevas reglas de asociación...")
            
            # Cargar información de productos para descripciones
            try:
                df_productos = pd.read_sql("SELECT stockcode, descripcion FROM dim_producto", engine)
            except:
                df_productos = pd.DataFrame({'stockcode': df_ventas['stockcode'].unique(), 
                                           'descripcion': df_ventas['stockcode'].unique()})
            
            # Preparar transacciones
            print("   Preparando transacciones...")
            transactions = df_ventas.groupby('invoice')['stockcode'].apply(list).tolist()
            transactions_filtered = [trans for trans in transactions if len(trans) >= 2]
            
            print(f"   Transacciones con 2+ productos: {len(transactions_filtered)}")
            
            if len(transactions_filtered) < 100:
                print("⚠️ Muy pocas transacciones para análisis de asociación")
                rules = pd.DataFrame()
            else:
                # Codificar transacciones
                te = TransactionEncoder()
                te_ary = te.fit_transform(transactions_filtered)
                basket_sets = pd.DataFrame(te_ary, columns=te.columns_)
                
                # Encontrar itemsets frecuentes
                print("   Encontrando itemsets frecuentes...")
                frequent_itemsets = apriori(basket_sets, min_support=MIN_SUPPORT, use_colnames=True)
                
                if len(frequent_itemsets) > 1:
                    # Generar reglas de asociación
                    print("   Generando reglas de asociación...")
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=MIN_LIFT)
                    
                    if len(rules) > 0:
                        # Convertir frozensets a listas
                        rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
                        rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
                        rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
                        
                        print(f"✅ {len(rules)} reglas de asociación generadas")
                    else:
                        print("⚠️ No se generaron reglas de asociación")
                        rules = pd.DataFrame()
                else:
                    print("⚠️ No hay suficientes itemsets frecuentes")
                    rules = pd.DataFrame()
        
        # Preparar reglas para subir a base de datos
        if len(rules) > 0:
            print("   Preparando reglas para base de datos...")
            
            for idx, rule in rules.head(100).iterrows():  # Limitar a top 100 reglas
                # Convertir listas a strings
                antecedents_str = ','.join(map(str, rule['antecedents']))
                consequents_str = ','.join(map(str, rule['consequents']))
                
                reglas_asociacion.append({
                    'regla_id': idx + 1,
                    'antecedentes': antecedents_str,
                    'consecuentes': consequents_str,
                    'soporte': round(float(rule['support']), 4),
                    'confianza': round(float(rule['confidence']), 4),
                    'lift': round(float(rule['lift']), 4),
                    'fecha_generacion': datetime.now().date(),
                    'activa': True
                })
            
            print(f"✅ {len(reglas_asociacion)} reglas preparadas para subir")
        else:
            print("⚠️ No hay reglas de asociación para subir")
            
    except Exception as e:
        print(f"❌ Error en análisis de reglas de asociación: {e}")
        reglas_asociacion = []

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
                print("✅ Modelo cargado (sin metadatos)")
            
            # Calcular métricas RFM actualizadas para todos los clientes
            print("   Calculando métricas RFM actualizadas...")
            fecha_ref = df_ventas["fecha"].max() + pd.Timedelta(days=1)
            
            rfm_actual = df_ventas.groupby("customer_id").agg({
                "fecha": lambda x: (fecha_ref - x.max()).days,   # Recencia
                "invoice": "nunique",                            # Frecuencia
                "monto_venta": "sum"                            # Monto
            }).reset_index()
            
            rfm_actual.columns = ["customer_id", "recencia", "frecuencia", "monto"]
            
            # Normalizar usando el scaler entrenado
            rfm_scaled = scaler.transform(rfm_actual[["recencia", "frecuencia", "monto"]])
            
            # Predecir clusters
            clusters = modelo_kmeans.predict(rfm_scaled)
            rfm_actual['cluster'] = clusters
            
            # Calcular información adicional para Power BI
            print("   Preparando datos para Power BI...")
            
            # Obtener información del país de cada cliente
            info_clientes = df_ventas.groupby('customer_id').agg({
                'fecha': ['min', 'max'],  # Primera y última compra
                'invoice': 'nunique',     # Número de órdenes
                'cantidad': 'sum',        # Productos totales comprados
                'monto_venta': ['sum', 'mean', 'std']  # Estadísticas de monto
            }).reset_index()
            
            # Aplanar columnas multi-nivel
            info_clientes.columns = ['customer_id', 'primera_compra', 'ultima_compra', 
                                   'total_ordenes', 'total_productos', 'monto_total', 
                                   'monto_promedio', 'monto_std']
            
            # Calcular días desde primera compra
           # Asegúrate que 'primera_compra' es datetime
            info_clientes['primera_compra'] = pd.to_datetime(info_clientes['primera_compra'], errors='coerce')

# Calcula los días correctamente
            info_clientes['dias_como_cliente'] = (fecha_ref - info_clientes['primera_compra']).dt.days

            # Obtener país del cliente (tomar el más frecuente)
            paises_clientes = df_ventas.groupby('customer_id').agg({
                'fecha': 'max'  # Para join con dim_cliente
            }).reset_index()
            
            # Obtener país desde dim_cliente
            try:
                df_paises = pd.read_sql("SELECT customer_id, pais FROM dim_cliente", engine)
                info_clientes = info_clientes.merge(df_paises, on='customer_id', how='left')
            except:
                info_clientes['pais'] = 'Unknown'
            
            # Combinar con RFM y clusters
            rfm_completo = rfm_actual.merge(info_clientes, on='customer_id', how='left')
            
            # Definir etiquetas de clusters más descriptivas
            cluster_labels = {
                0: "Clientes VIP",
                1: "Clientes Regulares", 
                2: "Clientes en Riesgo",
                3: "Nuevos Clientes",
                4: "Clientes Perdidos"
            }
            
            # Calcular estadísticas por cluster para definir mejor las etiquetas
            cluster_stats = rfm_actual.groupby('cluster').agg({
                'recencia': 'mean',
                'frecuencia': 'mean',
                'monto': 'mean'
            }).round(2)
            
            print("   Estadísticas por cluster:")
            print(cluster_stats)
            
            # Asignar etiquetas más inteligentes basadas en las métricas
            def asignar_etiqueta_cluster(row):
                if row['monto'] > rfm_actual['monto'].quantile(0.8) and row['frecuencia'] > rfm_actual['frecuencia'].median():
                    return "Clientes VIP"
                elif row['recencia'] > rfm_actual['recencia'].quantile(0.8):
                    return "Clientes Perdidos"
                elif row['recencia'] < rfm_actual['recencia'].quantile(0.3) and row['frecuencia'] > rfm_actual['frecuencia'].median():
                    return "Clientes Activos"
                elif row['monto'] < rfm_actual['monto'].quantile(0.3):
                    return "Clientes de Bajo Valor"
                else:
                    return "Clientes Regulares"
            
            cluster_labels_inteligentes = {}
            for cluster_id in cluster_stats.index:
                cluster_labels_inteligentes[cluster_id] = asignar_etiqueta_cluster(cluster_stats.loc[cluster_id])
            
            # Preparar datos para Supabase
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
                    'primera_compra': row.get('primera_compra', fecha_ref).date(),
                    'ultima_compra': row.get('ultima_compra', fecha_ref).date(),
                    'total_ordenes': int(row.get('total_ordenes', 0)),
                    'total_productos': int(row.get('total_productos', 0)),
                    'monto_promedio': round(float(row.get('monto_promedio', 0)), 2),
                    'dias_como_cliente': int(row.get('dias_como_cliente', 0)),
                    'pais': str(row.get('pais', 'Unknown')),
                    'fecha_actualizacion': datetime.now().date(),
                    'activo': True
                })
            
            print(f"✅ {len(segmentacion_clientes)} clientes segmentados")
            
            # Mostrar distribución de clusters
            distribucion = rfm_actual['cluster'].value_counts().sort_index()
            print("\n   Distribución por cluster:")
            for cluster_id, count in distribucion.items():
                label = cluster_labels_inteligentes.get(cluster_id, f"Cluster {cluster_id}")
                porcentaje = (count / len(rfm_actual)) * 100
                print(f"   {label}: {count} clientes ({porcentaje:.1f}%)")
                
        else:
            print("❌ No se encontró modelo de clustering entrenado")
            print("💡 Ejecuta primero el script de entrenamiento de clustering")
            
    except Exception as e:
        print(f"❌ Error en segmentación de clientes: {e}")
        segmentacion_clientes = []

else:
    print("⏭️ Saltando segmentación de clientes...")

# --- 5. SUBIR PREDICCIONES A SUPABASE ---
print("\n--- Subiendo Datos a Supabase ---")

# Subir predicciones de productos
if realizar_productos and predicciones_productos:
    df_pred_productos = pd.DataFrame(predicciones_productos)
    df_pred_productos.to_sql('predicciones_mensuales', engine, if_exists='append', index=False)
    print(f"✅ {len(df_pred_productos)} predicciones de productos insertadas para {N_MESES} meses.")
elif realizar_productos:
    print("⚠️ No se generaron predicciones de productos.")

# Subir predicciones de clientes
if realizar_clientes and predicciones_clientes:
    df_pred_clientes = pd.DataFrame(predicciones_clientes)
    df_pred_clientes.to_sql('predicciones_montos_clientes', engine, if_exists='append', index=False)
    print(f"✅ {len(df_pred_clientes)} predicciones de montos de clientes insertadas para {N_MESES} meses.")
elif realizar_clientes:
    print("⚠️ No se generaron predicciones de clientes.")

# Subir reglas de asociación
if realizar_asociaciones and reglas_asociacion:
    try:
        df_reglas = pd.DataFrame(reglas_asociacion)
        df_reglas.to_sql('reglas_asociacion', engine, if_exists='append', index=False)
        print(f"✅ {len(df_reglas)} reglas de asociación insertadas.")
    except Exception as e:
        print(f"❌ Error al subir reglas de asociación: {e}")
        print("💡 Asegúrate de que la tabla 'reglas_asociacion' existe en la base de datos")
elif realizar_asociaciones:
    print("⚠️ No se generaron reglas de asociación.")

# Subir segmentación de clientes
if realizar_clustering and segmentacion_clientes:
    try:
        df_segmentacion = pd.DataFrame(segmentacion_clientes)
        df_segmentacion.to_sql('segmentacion_clientes', engine, if_exists='append', index=False)
        print(f"✅ {len(df_segmentacion)} segmentaciones de clientes insertadas.")
        
        # Crear tabla resumen por cluster para Power BI
        resumen_clusters = df_segmentacion.groupby(['cluster_id', 'cluster_nombre']).agg({
            'customer_id': 'count',
            'recencia': 'mean',
            'frecuencia': 'mean', 
            'monto_total': ['mean', 'sum'],
            'total_ordenes': 'mean',
            'dias_como_cliente': 'mean'
        }).round(2)
        
        # Aplanar columnas multi-nivel
        resumen_clusters.columns = ['total_clientes', 'recencia_promedio', 'frecuencia_promedio', 
                                  'monto_promedio', 'monto_total_cluster', 'ordenes_promedio', 'dias_promedio']
        resumen_clusters = resumen_clusters.reset_index()
        resumen_clusters['fecha_actualizacion'] = datetime.now().date()
        resumen_clusters['porcentaje_clientes'] = (resumen_clusters['total_clientes'] / len(df_segmentacion) * 100).round(2)
        
        # Subir resumen de clusters
        resumen_clusters.to_sql('resumen_clusters', engine, if_exists='replace', index=False)
        print(f"✅ Resumen de {len(resumen_clusters)} clusters creado para Power BI.")
        
    except Exception as e:
        print(f"❌ Error al subir segmentación de clientes: {e}")
        print("💡 Asegúrate de que las tablas 'segmentacion_clientes' y 'resumen_clusters' existen")
elif realizar_clustering:
    print("⚠️ No se generaron segmentaciones de clientes.")

# --- 6. CREAR VISTAS ADICIONALES PARA POWER BI ---
print("\n--- Creando Vistas Adicionales para Power BI ---")

try:
    with engine.begin() as conn:
        # Vista consolidada de métricas por cliente
        vista_metricas_cliente = """
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
            d.pais as pais_dimension,
            CASE 
                WHEN s.recencia <= 30 THEN 'Activo'
                WHEN s.recencia <= 90 THEN 'En Riesgo'
                ELSE 'Inactivo'
            END as estado_actividad,
            CASE 
                WHEN s.monto_total >= (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY monto_total) FROM segmentacion_clientes) THEN 'Alto Valor'
                WHEN s.monto_total >= (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY monto_total) FROM segmentacion_clientes) THEN 'Medio Valor'
                ELSE 'Bajo Valor'
            END as segmento_valor
        FROM segmentacion_clientes s
        LEFT JOIN dim_cliente d ON s.customer_id = d.customer_id
        WHERE s.activo = true;
        """
        
        conn.execute(text(vista_metricas_cliente))
        print("✅ Vista 'vista_metricas_clientes' creada.")
        
        # Vista de evolución temporal de ventas por cluster
        if realizar_clustering and segmentacion_clientes:
            vista_evolucion_clusters = """
            CREATE OR REPLACE VIEW vista_evolucion_clusters AS
            SELECT 
                DATE_TRUNC('month', f.fecha) as año_mes,
                s.cluster_nombre,
                s.cluster_id,
                COUNT(DISTINCT f.customer_id) as clientes_activos,
                SUM(f.cantidad * f.precio_unitario) as ventas_totales,
                AVG(f.cantidad * f.precio_unitario) as venta_promedio,
                COUNT(f.invoice) as total_transacciones
            FROM fact_ventas f
            JOIN segmentacion_clientes s ON f.customer_id = s.customer_id
            WHERE s.activo = true
            GROUP BY DATE_TRUNC('month', f.fecha), s.cluster_nombre, s.cluster_id
            ORDER BY año_mes DESC;
            """
            
            conn.execute(text(vista_evolucion_clusters))
            print("✅ Vista 'vista_evolucion_clusters' creada.")
        
        # Vista de productos más vendidos por cluster
        if realizar_clustering and segmentacion_clientes:
            vista_productos_cluster = """
            CREATE OR REPLACE VIEW vista_productos_por_cluster AS
            SELECT 
                s.cluster_nombre,
                s.cluster_id,
                f.stockcode,
                p.descripcion,
                SUM(f.cantidad) as cantidad_total,
                SUM(f.cantidad * f.precio_unitario) as ventas_totales,
                COUNT(DISTINCT f.customer_id) as clientes_compradores,
                AVG(f.precio_unitario) as precio_promedio
            FROM fact_ventas f
            JOIN segmentacion_clientes s ON f.customer_id = s.customer_id
            JOIN dim_producto p ON f.stockcode = p.stockcode
            WHERE s.activo = true
            GROUP BY s.cluster_nombre, s.cluster_id, f.stockcode, p.descripcion
            ORDER BY s.cluster_id, cantidad_total DESC;
            """
            
            conn.execute(text(vista_productos_cluster))
            print("✅ Vista 'vista_productos_por_cluster' creada.")
        
        # Vista consolidada de predicciones vs reales
        if realizar_productos:
            vista_predicciones_productos = """
            CREATE OR REPLACE VIEW vista_predicciones_vs_reales AS
            SELECT 
                p.stockcode,
                pr.descripcion,
                p.fecha_prediccion,
                p.cantidad_predicha,
                COALESCE(SUM(f.cantidad), 0) as cantidad_real,
                ABS(p.cantidad_predicha - COALESCE(SUM(f.cantidad), 0)) as diferencia_absoluta,
                CASE 
                    WHEN COALESCE(SUM(f.cantidad), 0) > 0 THEN 
                        ABS(p.cantidad_predicha - COALESCE(SUM(f.cantidad), 0)) / COALESCE(SUM(f.cantidad), 1) * 100
                    ELSE NULL
                END as error_porcentual
            FROM predicciones_mensuales p
            LEFT JOIN dim_producto pr ON p.stockcode = pr.stockcode
            LEFT JOIN fact_ventas f ON p.stockcode = f.stockcode 
                AND DATE_TRUNC('month', f.fecha) = DATE_TRUNC('month', p.fecha_prediccion)
            GROUP BY p.stockcode, pr.descripcion, p.fecha_prediccion, p.cantidad_predicha;
            """
            
            conn.execute(text(vista_predicciones_productos))
            print("✅ Vista 'vista_predicciones_vs_reales' creada.")

except Exception as e:
    print(f"⚠️ Error creando vistas adicionales: {e}")

# --- 7. GENERAR TABLA DE KPIS PARA POWER BI ---
print("\n--- Generando KPIs para Power BI ---")

try:
    kpis_data = []
    fecha_actual = datetime.now().date()
    
    # KPIs generales
    total_clientes = len(df_ventas['customer_id'].unique())
    total_productos = len(df_ventas['stockcode'].unique())
    total_ventas = df_ventas['monto_venta'].sum()
    
    kpis_data.extend([
        {'kpi_nombre': 'Total Clientes', 'valor': total_clientes, 'categoria': 'General', 'fecha_actualizacion': fecha_actual},
        {'kpi_nombre': 'Total Productos', 'valor': total_productos, 'categoria': 'General', 'fecha_actualizacion': fecha_actual},
        {'kpi_nombre': 'Ventas Totales', 'valor': round(total_ventas, 2), 'categoria': 'General', 'fecha_actualizacion': fecha_actual}
    ])
    
    # KPIs de predicciones
    if predicciones_productos:
        total_pred_productos = len(predicciones_productos)
        productos_unicos_pred = len(set([p['stockcode'] for p in predicciones_productos]))
        kpis_data.extend([
            {'kpi_nombre': 'Predicciones Productos', 'valor': total_pred_productos, 'categoria': 'Predicciones', 'fecha_actualizacion': fecha_actual},
            {'kpi_nombre': 'Productos con Predicción', 'valor': productos_unicos_pred, 'categoria': 'Predicciones', 'fecha_actualizacion': fecha_actual}
        ])
    
    if predicciones_clientes:
        total_pred_clientes = len(predicciones_clientes)
        clientes_unicos_pred = len(set([p['client_id'] for p in predicciones_clientes]))
        kpis_data.extend([
            {'kpi_nombre': 'Predicciones Clientes', 'valor': total_pred_clientes, 'categoria': 'Predicciones', 'fecha_actualizacion': fecha_actual},
            {'kpi_nombre': 'Clientes con Predicción', 'valor': clientes_unicos_pred, 'categoria': 'Predicciones', 'fecha_actualizacion': fecha_actual}
        ])
    
    # KPIs de clustering
    if segmentacion_clientes:
        total_segmentados = len(segmentacion_clientes)
        clusters_unicos = len(set([s['cluster_id'] for s in segmentacion_clientes]))
        kpis_data.extend([
            {'kpi_nombre': 'Clientes Segmentados', 'valor': total_segmentados, 'categoria': 'Clustering', 'fecha_actualizacion': fecha_actual},
            {'kpi_nombre': 'Número de Clusters', 'valor': clusters_unicos, 'categoria': 'Clustering', 'fecha_actualizacion': fecha_actual}
        ])
    
    # KPIs de reglas de asociación
    if reglas_asociacion:
        total_reglas = len(reglas_asociacion)
        kpis_data.append({'kpi_nombre': 'Reglas de Asociación', 'valor': total_reglas, 'categoria': 'Recomendaciones', 'fecha_actualizacion': fecha_actual})
    
    # Subir KPIs a Supabase
    if kpis_data:
        df_kpis = pd.DataFrame(kpis_data)
        
        # Limpiar KPIs anteriores del día
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM kpis_dashboard WHERE fecha_actualizacion = :fecha"), {"fecha": fecha_actual})
        
        df_kpis.to_sql('kpis_dashboard', engine, if_exists='append', index=False)
        print(f"✅ {len(df_kpis)} KPIs actualizados para Power BI.")

except Exception as e:
    print(f"⚠️ Error generando KPIs: {e}")

# --- 8. RESUMEN FINAL ---
print(f"\n=== RESUMEN FINAL ===")
print(f"📦 Predicciones de productos: {len(predicciones_productos)}")
print(f"🔢 Productos únicos con predicciones: {len(set([p['stockcode'] for p in predicciones_productos])) if predicciones_productos else 0}")
print(f"👥 Predicciones de clientes: {len(predicciones_clientes)}")
print(f"🔢 Clientes únicos con predicciones: {len(set([p['client_id'] for p in predicciones_clientes])) if predicciones_clientes else 0}")
print(f"🔗 Reglas de asociación: {len(reglas_asociacion)}")
print(f"🎯 Clientes segmentados: {len(segmentacion_clientes)}")
print(f"📅 Meses predichos: {N_MESES}")
print("🎉 Proceso completado exitosamente!")

# Mostrar algunas reglas de asociación si se generaron
if reglas_asociacion:
    print(f"\n=== TOP 5 REGLAS DE ASOCIACIÓN ===")
    for i, regla in enumerate(reglas_asociacion[:5]):
        print(f"{i+1}. Si compra [{regla['antecedentes']}] → entonces [{regla['consecuentes']}]")
        print(f"   Confianza: {regla['confianza']:.3f} | Lift: {regla['lift']:.3f}")

# Mostrar distribución de clusters si se generaron
if segmentacion_clientes:
    print(f"\n=== DISTRIBUCIÓN DE CLUSTERS ===")
    clusters_dist = {}
    for cliente in segmentacion_clientes:
        cluster_nombre = cliente['cluster_nombre']
        if cluster_nombre not in clusters_dist:
            clusters_dist[cluster_nombre] = 0
        clusters_dist[cluster_nombre] += 1
    
    total_clientes_seg = len(segmentacion_clientes)
    for cluster_nombre, count in clusters_dist.items():
        porcentaje = (count / total_clientes_seg) * 100
        print(f"{cluster_nombre}: {count} clientes ({porcentaje:.1f}%)")

# Información para conectar Power BI
print(f"\n=== INFORMACIÓN PARA POWER BI ===")
print("📊 Tablas principales disponibles:")
print("   • predicciones_mensuales - Predicciones de productos")
print("   • predicciones_montos_clientes - Predicciones de clientes")
print("   • reglas_asociacion - Reglas de recomendación")
print("   • segmentacion_clientes - Segmentación de clientes")
print("   • resumen_clusters - Resumen estadístico por cluster")
print("   • kpis_dashboard - KPIs principales")
print("\n📈 Vistas para análisis avanzado:")
print("   • vista_metricas_clientes - Métricas consolidadas")
print("   • vista_evolucion_clusters - Evolución temporal por cluster")
print("   • vista_productos_por_cluster - Productos preferidos por segmento")
print("   • vista_predicciones_vs_reales - Comparación predicciones vs reales")
print("\n🔗 Usa estas tablas y vistas en Power BI para crear dashboards interactivos")