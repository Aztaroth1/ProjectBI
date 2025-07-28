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
MIN_PRODUCTOS_PREDICCION = 10  # Mínimo número de productos a predecir

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
print("4. Productos + Clientes")
print("5. Productos + Reglas de asociación")
print("6. Clientes + Reglas de asociación")
print("7. Todas las opciones")

while True:
    try:
        opcion = int(input("\nIngresa tu opción (1-7): "))
        if opcion in range(1, 8):
            break
        else:
            print("❌ Por favor ingresa un número entre 1 y 7")
    except ValueError:
        print("❌ Por favor ingresa un número válido")

realizar_productos = opcion in [1, 4, 5, 7]
realizar_clientes = opcion in [2, 4, 6, 7]
realizar_asociaciones = opcion in [3, 5, 6, 7]

print(f"\n=== CONFIGURACIÓN SELECCIONADA ===")
print(f"📦 Predicciones de productos: {'SÍ' if realizar_productos else 'NO'}")
print(f"👥 Predicciones de clientes: {'SÍ' if realizar_clientes else 'NO'}")
print(f"🔗 Reglas de asociación: {'SÍ' if realizar_asociaciones else 'NO'}")
print(f"📅 Meses a predecir: {N_MESES}")
print(f"💾 Modo de inserción: LIMPIAR FUTURAS (automático)")

input("\nPresiona Enter para continuar...")

# --- LIMPIAR DATOS FUTUROS (AUTOMÁTICO) ---
print("\n--- Limpiando Predicciones Futuras ---")
fecha_actual = datetime.now().date()

with engine.begin() as conn:
    if realizar_productos:
        result = conn.execute(text("DELETE FROM predicciones_mensuales WHERE fecha_prediccion >= :fecha"), {"fecha": fecha_actual})
        print(f"🧹 Eliminadas {result.rowcount} predicciones futuras de productos")
    
    if realizar_clientes:
        result = conn.execute(text("DELETE FROM predicciones_montos_clientes WHERE fecha_prediccion >= :fecha"), {"fecha": fecha_actual})
        print(f"🧹 Eliminadas {result.rowcount} predicciones futuras de clientes")
    
    if realizar_asociaciones:
        # Limpiar reglas de asociación anteriores
        try:
            result = conn.execute(text("DELETE FROM reglas_asociacion WHERE fecha_generacion >= :fecha"), {"fecha": fecha_actual})
            print(f"🧹 Eliminadas {result.rowcount} reglas de asociación anteriores")
        except Exception as e:
            print(f"ℹ️ No se pudo limpiar reglas de asociación: {e}")

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

# --- 4. SUBIR PREDICCIONES A SUPABASE ---
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

# --- 5. RESUMEN FINAL ---
print(f"\n=== RESUMEN FINAL ===")
print(f"📦 Predicciones de productos: {len(predicciones_productos)}")
print(f"🔢 Productos únicos con predicciones: {len(set([p['stockcode'] for p in predicciones_productos])) if predicciones_productos else 0}")
print(f"👥 Predicciones de clientes: {len(predicciones_clientes)}")
print(f"🔢 Clientes únicos con predicciones: {len(set([p['client_id'] for p in predicciones_clientes])) if predicciones_clientes else 0}")
print(f"🔗 Reglas de asociación: {len(reglas_asociacion)}")
print(f"📅 Meses predichos: {N_MESES}")
print("🎉 Proceso completado exitosamente!")

# Mostrar algunas reglas de asociación si se generaron
if reglas_asociacion:
    print(f"\n=== TOP 5 REGLAS DE ASOCIACIÓN ===")
    for i, regla in enumerate(reglas_asociacion[:5]):
        print(f"{i+1}. Si compra [{regla['antecedentes']}] → entonces [{regla['consecuentes']}]")
        print(f"   Confianza: {regla['confianza']:.3f} | Lift: {regla['lift']:.3f}")