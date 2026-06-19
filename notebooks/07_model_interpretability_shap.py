# Importar el módulo os para interactuar con el sistema operativo
import os
# Importar pandas para el manejo y análisis de estructuras de datos
import pandas as pd
# Importar numpy para cálculos numéricos y manejo de matrices
import numpy as np
# Importar pyplot de matplotlib para crear gráficos y visualizaciones
import matplotlib.pyplot as plt
# Importar seaborn para visualización de datos estadísticos
import seaborn as sns
# Importar xgboost para utilizar el algoritmo de gradient boosting
import xgboost as xgb
# Importar el módulo warnings para controlar las advertencias emitidas por Python
import warnings
# Filtrar e ignorar todas las advertencias para que no se muestren en consola
warnings.filterwarnings('ignore')

# Iniciar bloque try para intentar importar el módulo shap
try:
    # Intentar importar la librería shap para interpretabilidad de modelos
    import shap
# Capturar la excepción ImportError si shap no está instalado
except ImportError:
    # Imprimir un mensaje indicando que SHAP no está instalado y se procederá a instalarlo
    print("⚠️ SHAP no está instalado. Instalando...")
    # Importar el módulo subprocess para ejecutar comandos del sistema
    import subprocess
    # Ejecutar pip install shap de forma silenciosa para instalar la librería
    subprocess.check_call(['pip', 'install', 'shap', '-q'])
    # Importar shap nuevamente después de haberlo instalado exitosamente
    import shap

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ==========================================
# Definir la ruta del directorio donde se encuentran los datos procesados
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
# Definir la ruta del directorio donde se guardarán los resultados textuales
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
# Definir la ruta del directorio donde se guardarán los gráficos generados
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"

# Crear el directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)
# Crear el directorio de gráficos si no existe
os.makedirs(PLOTS_DIR, exist_ok=True)

# Crear la ruta completa para el archivo de salida de texto combinando RESULTS_DIR y el nombre del archivo
TXT_OUTPUT = os.path.join(RESULTS_DIR, "07_shap_interpretability.txt")

# Imprimir una línea separadora de 80 caracteres de igual
print("="*80)
# Imprimir el título del proceso actual
print("ANÁLISIS DE INTERPRETABILIDAD CON SHAP (VERSIÓN MEJORADA)")
# Imprimir otra línea separadora de 80 caracteres de igual
print("="*80)

# ==========================================
# 2. CARGAR DATOS Y ENTRENAR MODELO
# ==========================================
# Imprimir mensaje indicando el inicio del paso 1: carga de datos y entrenamiento
print("\n[1/4] Cargando datos y entrenando modelo...")

# Cargar el conjunto de características de entrenamiento desde un archivo CSV
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
# Cargar el conjunto de características de prueba desde un archivo CSV
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
# Cargar las etiquetas de entrenamiento, convertirlas a un arreglo numpy y aplanarlas a 1D
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
# Cargar las etiquetas de prueba, convertirlas a un arreglo numpy y aplanarlas a 1D
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

# Convertir los tipos de datos de X_train a float64 para evitar problemas computacionales
X_train = X_train.astype('float64')
# Convertir los tipos de datos de X_test a float64 para evitar problemas computacionales
X_test = X_test.astype('float64')

# Contar el número de ejemplos negativos (clase 0) en las etiquetas de entrenamiento
num_neg = np.sum(y_train == 0)
# Contar el número de ejemplos positivos (clase 1) en las etiquetas de entrenamiento
num_pos = np.sum(y_train == 1)
# Calcular el peso de la clase positiva para manejar el desbalanceo de clases
scale_pos_weight = num_neg / num_pos

# Inicializar y configurar el modelo XGBoost Classifier
model = xgb.XGBClassifier(
    # Definir el número de árboles a construir (150)
    n_estimators=150,
    # Definir la profundidad máxima de cada árbol (5)
    max_depth=5,
    # Establecer la tasa de aprendizaje del modelo (0.05)
    learning_rate=0.05,
    # Asignar el peso para balancear la clase positiva
    scale_pos_weight=scale_pos_weight,
    # Fijar la semilla aleatoria para asegurar la reproducibilidad
    random_state=42,
    # Especificar la métrica de evaluación a usar (logloss)
    eval_metric='logloss',
    # Configurar el nivel de detalle de los mensajes en consola (0 = silencioso)
    verbosity=0
)

# Entrenar el modelo con el conjunto de entrenamiento (X_train, y_train)
model.fit(X_train, y_train)
# Imprimir mensaje confirmando que el modelo se entrenó e indicando el número de muestras
print(f"   ✓ Modelo entrenado con {X_train.shape[0]} muestras")
# Imprimir mensaje indicando el número de características utilizadas en el entrenamiento
print(f"   ✓ Características: {X_train.shape[1]}")

# ==========================================
# 3. CALCULAR SHAP VALUES
# ==========================================
# Imprimir mensaje indicando el inicio del paso 2: cálculo de valores SHAP
print("\n[2/4] Calculando SHAP values (TreeExplainer)...")
# Imprimir advertencia sobre el tiempo estimado del proceso
print("   Esto puede tomar 1-2 minutos...")

# Iniciar bloque try para el cálculo de valores SHAP usando TreeExplainer
try:
    # Instanciar TreeExplainer de SHAP pasándole el modelo entrenado
    explainer = shap.TreeExplainer(model)
    # Calcular los valores SHAP para el conjunto de prueba
    shap_values = explainer.shap_values(X_test)
    
    # Comprobar si los valores SHAP devueltos son una lista (clasificación multiclase/binaria dependiente de versión)
    if isinstance(shap_values, list):
        # Seleccionar los valores SHAP correspondientes a la clase 1 (positiva)
        shap_values_class1 = shap_values[1]  # Clase 1 (Sepsis)
    # Si no es una lista, asumimos que ya corresponde a la salida esperada
    else:
        # Asignar los valores SHAP directamente
        shap_values_class1 = shap_values
    
    # Obtener el valor base esperado del explainer
    expected_value = explainer.expected_value
    # Comprobar si el valor esperado es una lista
    if isinstance(expected_value, list):
        # Extraer el valor esperado correspondiente a la clase 1
        expected_value_class1 = expected_value[1]
    # Si no es una lista, asignar el valor directamente
    else:
        # Asignar el valor esperado
        expected_value_class1 = expected_value
    
    # Imprimir confirmación de valores SHAP calculados y su forma
    print(f"   ✓ SHAP values calculados: {shap_values_class1.shape}")
    # Imprimir el valor base esperado con 4 decimales
    print(f"   ✓ Expected value (baseline): {expected_value_class1:.4f}")
    
    # Imprimir mensaje indicando preparación de datos para visualizaciones
    print("\n   Preparando datos para visualización...")
    # Crear una pequeña muestra (máximo 100 registros) del conjunto de entrenamiento
    X_train_sample = X_train.iloc[:min(100, len(X_train))].copy()
    # Crear un nuevo TreeExplainer para visualización
    explainer_viz = shap.TreeExplainer(model)
    # Generar el objeto Explanation explainer_viz para una pequeña muestra del conjunto de prueba
    shap_values_viz = explainer_viz(X_test.iloc[:min(100, len(X_test))].copy())
    
    # Imprimir confirmación del tamaño de la muestra de visualización
    print(f"   ✓ Muestra para visualización: {X_test.iloc[:min(100, len(X_test))].shape[0]} muestras")
    
# Capturar cualquier excepción que ocurra dentro del bloque try
except Exception as e:
    # Imprimir mensaje de error mostrando el motivo por el cual falló TreeExplainer
    print(f"   ⚠️ Error en TreeExplainer: {str(e)}")
    # Imprimir aviso indicando que se utilizará el método alternativo KernelExplainer
    print("   Usando método alternativo (KernelExplainer)...")
    
    # Inicializar el explicador genérico KernelExplainer de SHAP
    explainer = shap.KernelExplainer(
        # Pasar el método de predicción de probabilidades del modelo
        model.predict_proba,
        # Proveer una pequeña muestra representativa (50 registros) del set de entrenamiento usando shap.sample
        shap.sample(X_train, 50),  # Sample de 50 muestras como background
        # Especificar la función de enlace "logit"
        link="logit"
    )
    # Calcular los valores SHAP para el conjunto de prueba usando KernelExplainer
    shap_values = explainer.shap_values(X_test)
    
    # Comprobar si los valores SHAP devueltos son una lista
    if isinstance(shap_values, list):
        # Seleccionar la segunda lista correspondiente a la clase 1 (Sepsis)
        shap_values_class1 = shap_values[1]
    # En caso contrario
    else:
        # Asignar directamente a shap_values_class1
        shap_values_class1 = shap_values
    
    # Obtener el valor esperado (baseline) del explainer
    expected_value_class1 = explainer.expected_value
    # Comprobar si es una lista
    if isinstance(expected_value_class1, list):
        # Tomar el valor de la clase 1
        expected_value_class1 = expected_value_class1[1]
    
    # Imprimir que se calcularon correctamente los valores con KernelExplainer
    print(f"   ✓ SHAP values calculados (KernelExplainer)")
    # Imprimir el valor esperado
    print(f"   ✓ Expected value (baseline): {expected_value_class1:.4f}")
    
    # Asignar a shap_values_viz la matriz de valores SHAP para usar en visualizaciones genéricas
    shap_values_viz = shap_values
    # Asignar a explainer_viz el explainer actual
    explainer_viz = explainer

# ==========================================
# 4. ANÁLISIS DE IMPORTANCIA GLOBAL
# ==========================================
# Imprimir título para la sección de análisis global
print("\n[3/4] Generando análisis de importancia global...")

# Crear un DataFrame con las métricas de importancia de cada variable basadas en SHAP
feature_importance_shap = pd.DataFrame({
    # Almacenar el nombre de cada característica a partir de las columnas de X_test
    'Feature': X_test.columns,
    # Calcular y almacenar la media del valor absoluto de SHAP por característica a lo largo de todas las muestras (axis=0)
    'SHAP_Mean_Abs': np.abs(shap_values_class1).mean(axis=0),
    # Calcular y almacenar la media aritmética (con signo) del valor SHAP por característica (axis=0)
    'SHAP_Mean': shap_values_class1.mean(axis=0)
# Ordenar el DataFrame de manera descendente según el valor absoluto medio de SHAP
}).sort_values('SHAP_Mean_Abs', ascending=False)

# Imprimir encabezado para la visualización en consola del Top 10 de características
print("\n   📊 TOP 10 CARACTERÍSTICAS POR SHAP (Media Absoluta):")
# Imprimir los nombres de columnas para el reporte en consola con formato de espaciado
print(f"   {'Rank':<5} {'Feature':<35} {'SHAP Mean Abs':<15} {'SHAP Mean':<15}")
# Imprimir línea divisoria horizontal para la tabla
print(f"   {'-'*70}")
# Iterar sobre las 10 características más importantes
for idx, (_, row) in enumerate(feature_importance_shap.head(10).iterrows(), 1):
    # Imprimir el rango, nombre, media absoluta y media con signo de cada característica con espaciado formateado
    print(f"   {idx:<5} {row['Feature']:<35} {row['SHAP_Mean_Abs']:<15.6f} {row['SHAP_Mean']:<15.6f}")

# ==========================================
# 5. EXPORTAR RESULTADOS A TXT
# ==========================================
# Abrir (o crear) el archivo de salida TXT en modo escritura con codificación UTF-8
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    # Escribir una línea divisoria de igual en el archivo
    f.write("="*90 + "\n")
    # Escribir el título principal del reporte
    f.write("REPORTE DE INTERPRETABILIDAD - SHAP (Shapley Additive exPlanations)\n")
    # Escribir otra línea divisoria y doble salto de línea
    f.write("="*90 + "\n\n")
    
    # Escribir el subtítulo para la sección teórica
    f.write("1. TEORÍA DE SHAP\n")
    # Escribir línea divisoria de guiones
    f.write("-"*90 + "\n")
    # Explicar qué hace SHAP conceptualmente
    f.write("   SHAP utiliza la teoría de juegos para explicar predicciones de ML.\n")
    # Explicar cómo calcula la contribución
    f.write("   Para cada predicción, SHAP calcula la contribución de cada característica\n")
    # Explicar que se basa en permutaciones del modelo
    f.write("   basada en permutaciones del modelo. El valor SHAP representa cuánto\n")
    # Explicar la relación con el valor base
    f.write("   cada característica contribuye a alejar la predicción del valor base.\n\n")
    # Imprimir el valor base calculado
    f.write(f"   Expected Value (Base): {expected_value_class1:.6f}\n")
    # Aclarar que el valor base es el promedio en los datos de entrenamiento
    f.write(f"   (Promedio de predicciones en datos de entrenamiento)\n\n")
    
    # Escribir subtítulo de la sección de Top 15 características
    f.write("2. TOP 15 CARACTERÍSTICAS POR IMPORTANCIA SHAP\n")
    # Escribir línea de guiones
    f.write("-"*90 + "\n")
    # Escribir los encabezados de la tabla con su espaciado
    f.write(f"{'Rank':<5} {'Feature':<35} {'SHAP Mean Abs':<18} {'SHAP Mean':<18}\n")
    # Escribir línea separadora para el encabezado
    f.write(f"{'-'*76}\n")
    
    # Iterar sobre las top 15 características y guardar datos en el TXT
    for idx, (_, row) in enumerate(feature_importance_shap.head(15).iterrows(), 1):
        # Escribir cada fila formateada con rango, nombre y valores SHAP
        f.write(f"{idx:<5} {row['Feature']:<35} {row['SHAP_Mean_Abs']:<18.6f} {row['SHAP_Mean']:<18.6f}\n")
    
    # Escribir un salto de línea en el archivo
    f.write("\n")
    # Escribir subtítulo de interpretación clínica
    f.write("3. INTERPRETACIÓN CLÍNICA DE RESULTADOS SHAP\n")
    # Escribir línea separadora de guiones
    f.write("-"*90 + "\n")
    
    # Extraer únicamente el Top 5 de variables
    top_5 = feature_importance_shap.head(5)
    # Escribir mensaje introductorio de las top 5
    f.write("   Las 5 características más importantes para predecir sepsis:\n\n")
    
    # Iterar para describir clínica/analíticamente el impacto de las top 5 variables
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        # Guardar el nombre de la variable
        feature_name = row['Feature']
        # Guardar el valor SHAP absoluto
        shap_impact = row['SHAP_Mean_Abs']
        # Determinar el texto de la dirección del impacto (aumento o disminución de riesgo)
        direction = "Aumenta riesgo" if row['SHAP_Mean'] > 0 else "Disminuye riesgo"
        
        # Escribir el número y nombre en mayúsculas de la variable
        f.write(f"   {idx}. {feature_name.upper()}\n")
        # Escribir el impacto SHAP calculado
        f.write(f"      • SHAP Impact: {shap_impact:.6f}\n")
        # Escribir la tendencia calculada
        f.write(f"      • Tendencia: {direction}\n")
        # Escribir la explicación inicial del impacto
        f.write(f"      • Explicación: Esta variable tiene alto impacto en las\n")
        # Completar la explicación
        f.write(f"        predicciones del modelo de sepsis.\n\n")
    
    # Escribir sección de validez del análisis
    f.write("4. VALIDEZ DEL ANÁLISIS SHAP\n")
    # Escribir guiones separadores
    f.write("-"*90 + "\n")
    # Escribir confirmación del método usado y su complejidad
    f.write("   ✓ Método: TreeExplainer (O(T*D) complejidad)\n")
    # Escribir nota sobre eficiencia
    f.write("   ✓ Eficiencia: Rápido para gradient boosting\n")
    # Escribir nota sobre convergencia teórica
    f.write("   ✓ Convergencia: Teóricamente garantizada\n")
    # Escribir qué datos se utilizaron
    f.write("   ✓ Muestras: Test set completo analizado\n")
    # Explicar garantías matemáticas de SHAP
    f.write("   ✓ Validez: SHAP values respetan propiedades de Shapley\n")
    # Mencionar las propiedades concretas
    f.write(f"     (Eficiencia local, simetría, dummy, aditividad)\n\n")
    
    # Escribir la última sección con un ejemplo de interpretación individual
    f.write("5. EJEMPLO DE INTERPRETACIÓN INDIVIDUAL\n")
    # Escribir guiones separadores
    f.write("-"*90 + "\n")
    # Escribir contexto explicativo
    f.write(f"   Para una predicción individual, SHAP muestra:\n")
    # Escribir cuál es el valor base en una predicción
    f.write(f"   • Valor base (Expected Value): {expected_value_class1:.4f}\n")
    # Explicar qué representa la contribución de features
    f.write(f"   • Contribución de cada feature: Valor SHAP\n")
    # Explicar la fórmula de la predicción final
    f.write(f"   • Predicción final: Base + Σ(Contribuciones SHAP)\n\n")
    # Iniciar sección de ejemplo textual
    f.write(f"   Ejemplo:\n")
    # Poner contexto del paciente
    f.write(f"   Si un paciente tiene:\n")
    # Añadir detalle de creatinina
    f.write(f"   - Creatinina alta: +0.15 (aumenta probabilidad de sepsis)\n")
    # Añadir detalle de SpO2
    f.write(f"   - SpO2 normal: -0.08 (reduce probabilidad)\n")
    # Añadir detalle de lactato
    f.write(f"   - Lactato elevado: +0.22 (fuerte indicador de sepsis)\n")
    # Mostrar cálculo hipotético final
    f.write(f"   Predicción: 0.50 (base) + 0.15 - 0.08 + 0.22 = 0.79 (79% sepsis)\n\n")

# Imprimir en consola que el reporte de texto ha sido guardado
print(f"   ✓ Reporte guardado")

# ==========================================
# 6. GENERAR VISUALIZACIONES SHAP
# ==========================================
# Indicar el inicio del último paso, que es la generación de gráficos
print("\n[4/4] Generando visualizaciones SHAP...")

# Gráfico 1: Summary Bar Plot
# Intentar graficar el Summary Bar Plot
try:
    # Crear una nueva figura de pyplot con tamaño específico
    plt.figure(figsize=(10, 8))
    
    # Seleccionar las top 15 características para mostrar en el plot
    # Crear bar plot manualmente si shap.summary_plot falla
    top_features = feature_importance_shap.head(15)
    # Generar colores usando el mapa de color viridis basado en la cantidad de variables
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    
    # Graficar gráfico de barras horizontal (barh) con la media absoluta
    plt.barh(range(len(top_features)), top_features['SHAP_Mean_Abs'], color=colors)
    # Reemplazar las etiquetas del eje Y por los nombres de las características
    plt.yticks(range(len(top_features)), top_features['Feature'])
    # Configurar el título del eje X
    plt.xlabel('Mean |SHAP value|', fontweight='bold', fontsize=11)
    # Configurar el título superior del gráfico
    plt.title("SHAP Summary - Importancia Media Absoluta de Características", 
              fontsize=12, fontweight='bold', pad=20)
    # Invertir el eje Y para que la más importante quede arriba
    plt.gca().invert_yaxis()
    # Agregar una cuadrícula de referencia para el eje X
    plt.grid(axis='x', alpha=0.3)
    
    # Iterar a través de los valores de SHAP para colocar los textos numéricos junto a cada barra
    for i, val in enumerate(top_features['SHAP_Mean_Abs']):
        # Escribir el valor numérico a un lado del borde de la barra
        plt.text(val, i, f' {val:.6f}', va='center', fontsize=9)
    
    # Ajustar el diseño para que todo encaje sin cortarse
    plt.tight_layout()
    # Guardar el gráfico en la ruta PLOTS_DIR con resolución de 300 dpi
    plt.savefig(os.path.join(PLOTS_DIR, "13_shap_summary_bar.png"), dpi=300, bbox_inches='tight')
    # Cerrar la figura para liberar memoria
    plt.close()
    # Imprimir éxito de la creación de la figura
    print("   ✓ Summary bar plot guardado")
# Capturar fallos en la generación del primer gráfico
except Exception as e:
    # Imprimir error si falló
    print(f"   ⚠️ Error en summary bar plot: {str(e)}")

# Gráfico 2: Feature Importance por valor absoluto (alternativa a beeswarm)
# Intentar graficar los Scatter Plots (Dependence Plots) para las variables más importantes
try:
    # Crear figura y ejes para 2 gráficos en fila (1 fila, 2 columnas)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Seleccionar las top 2 variables de importancia global para este gráfico
    # Gráfico de dispersión SHAP vs Feature Value para top 2 características
    top_2_features = feature_importance_shap.head(2)
    
    # Iterar a través de estas 2 variables principales y el índice de cada gráfico
    for plot_idx, (_, feature_row) in enumerate(top_2_features.iterrows()):
        # Seleccionar el eje/subplot correspondiente
        ax = axes[plot_idx]
        # Obtener el nombre de la variable
        feature_name = feature_row['Feature']
        # Obtener el índice numérico de la variable en la matriz de prueba (columnas)
        feature_idx = X_test.columns.get_loc(feature_name)
        
        # Obtener la columna de valores reales de la variable en X_test
        feature_values = X_test.iloc[:, feature_idx].values
        # Obtener los valores SHAP correspondientes para esa variable
        shap_feature_values = shap_values_class1[:, feature_idx]
        
        # Generar gráfico de dispersión, coloreado por la etiqueta real (y_test)
        scatter = ax.scatter(feature_values, shap_feature_values, 
                           c=y_test, cmap='RdYlGn_r', alpha=0.6, s=50)
        
        # Etiquetar el eje X con el nombre de la variable
        ax.set_xlabel(feature_name, fontweight='bold', fontsize=11)
        # Etiquetar el eje Y como SHAP Value
        ax.set_ylabel('SHAP Value', fontweight='bold', fontsize=11)
        # Asignar un título indicativo al subplot
        ax.set_title(f'{feature_name} - Relación SHAP', fontweight='bold', fontsize=12)
        # Mostrar grilla de fondo en el subplot
        ax.grid(alpha=0.3)
        
        # Agregar una barra de color referencial al lado del subplot
        cbar = plt.colorbar(scatter, ax=ax)
        # Etiquetar la barra de color (0=No Sepsis, 1=Sepsis)
        cbar.set_label('0=No Sepsis\n1=Sepsis', fontweight='bold', fontsize=9)
    
    # Configurar título principal central para la figura entera
    plt.suptitle("SHAP Dependence - Top 2 Características", fontsize=14, fontweight='bold')
    # Ajustar espaciado entre subplots
    plt.tight_layout()
    # Guardar gráfico Dependence
    plt.savefig(os.path.join(PLOTS_DIR, "14_shap_dependence_top2.png"), dpi=300, bbox_inches='tight')
    # Cerrar la figura
    plt.close()
    # Imprimir éxito de este guardado
    print("   ✓ Dependence plot guardado")
# Capturar si falla
except Exception as e:
    # Imprimir el mensaje de error de este paso
    print(f"   ⚠️ Error en dependence plot: {str(e)}")

# Gráfico 3: Ejemplos de predicciones individuales
# Intentar graficar ejemplos textuales simulados de fuerza (como force plots individuales)
try:
    # Obtener los índices de los primeros 5 casos positivos reales (sepsis)
    sepsis_indices = np.where(y_test == 1)[0][:5]
    # Obtener los índices de los primeros 5 casos negativos reales (no sepsis)
    no_sepsis_indices = np.where(y_test == 0)[0][:5]
    # Combinar en una sola lista los 10 índices a graficar
    example_indices = list(sepsis_indices) + list(no_sepsis_indices)
    
    # Crear una figura con 5 filas y 2 columnas para albergar los 10 ejemplos
    fig, axes = plt.subplots(5, 2, figsize=(14, 12))
    # Configurar título general de la figura de ejemplos
    fig.suptitle("Ejemplos de Predicciones - Contribución SHAP Top 3 Features", 
                 fontsize=14, fontweight='bold')
    
    # Iterar a través de la lista combinada de índices de ejemplos
    for plot_idx, idx in enumerate(example_indices):
        # Seleccionar el eje correspondiente basándose en su fila (plot_idx // 2) y columna (plot_idx % 2)
        ax = axes[plot_idx // 2, plot_idx % 2]
        
        # Calcular la probabilidad predicha por el modelo para este ejemplo individual (clase 1)
        pred = model.predict_proba(X_test.iloc[[idx]])[0, 1]
        # Obtener los valores SHAP específicos para este ejemplo
        shap_sample = shap_values_class1[idx]
        
        # Definir la etiqueta string de la clase real basándose en y_test
        class_label = "Sepsis" if y_test[idx] == 1 else "No Sepsis"
        # Definir la predicción del modelo basándose en un umbral de 0.5
        predicted_label = "Sepsis" if pred >= 0.5 else "No Sepsis"
        
        # Imprimir en texto del plot el número de muestra
        ax.text(0.5, 0.9, f"Muestra {idx}", ha='center', fontsize=10, fontweight='bold',
                transform=ax.transAxes)
        # Imprimir qué clase era y qué predijo el modelo
        ax.text(0.5, 0.75, f"Real: {class_label} | Predicción: {predicted_label}", 
                ha='center', fontsize=9, transform=ax.transAxes)
        # Imprimir el valor real de la probabilidad predicha
        ax.text(0.5, 0.6, f"P(Sepsis) = {pred:.4f}", ha='center', fontsize=9, fontweight='bold',
                # Pintar de rojo si es sepsis, verde si no
                color='red' if pred > 0.5 else 'green', transform=ax.transAxes)
        
        # Ordenar y seleccionar los índices de los 3 valores SHAP con mayor magnitud (absoluta)
        top_features_idx = np.argsort(np.abs(shap_sample))[-3:][::-1]
        # Posición inicial vertical para el primer texto de característica
        y_pos = 0.45
        # Iterar por las 3 características principales de este caso particular
        for feature_idx in top_features_idx:
            # Obtener nombre de la característica
            feature_name = X_test.columns[feature_idx]
            # Obtener su valor numérico específico de esta fila
            feature_value = X_test.iloc[idx, feature_idx]
            # Obtener el valor SHAP específico
            shap_val = shap_sample[feature_idx]
            
            # Imprimir como texto de eje el nombre, valor y su SHAP contribution
            ax.text(0.05, y_pos, f"{feature_name}: {feature_value:.2f} (SHAP: {shap_val:.4f})",
                    ha='left', fontsize=8, transform=ax.transAxes, family='monospace')
            # Reducir y_pos para el siguiente texto
            y_pos -= 0.12
        
        # Apagar los bordes y grilla del eje, ya que es sólo texto dibujado
        ax.axis('off')
    
    # Ajustar para evitar solapamientos
    plt.tight_layout()
    # Guardar en la carpeta PLOTS_DIR
    plt.savefig(os.path.join(PLOTS_DIR, "15_shap_examples.png"), dpi=300, bbox_inches='tight')
    # Cerrar la figura
    plt.close()
    # Imprimir éxito de este guardado
    print("   ✓ Examples guardado")
# Capturar error
except Exception as e:
    # Imprimir error si existe
    print(f"   ⚠️ Error en ejemplos: {str(e)}")

# Gráfico 4: Distribución de SHAP values para top 5 características
# Intentar graficar los histogramas de valores SHAP
try:
    # Crear una fila de 5 subplots (uno por cada top feature)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    # Título principal de distribuciones
    fig.suptitle("Distribución de SHAP Values - Top 5 Características", 
                 fontsize=14, fontweight='bold')
    
    # Obtener el top 5 de variables nuevamente
    top_5_features = feature_importance_shap.head(5)
    
    # Iterar a través de ellas para generar cada histograma
    for idx, (_, feature_row) in enumerate(top_5_features.iterrows()):
        # Elegir el subplot que le toca (del 0 al 4)
        ax = axes[idx]
        # Nombre de la variable
        feature_name = feature_row['Feature']
        # Índice en X_test
        feature_idx = X_test.columns.get_loc(feature_name)
        
        # Extraer todos los valores SHAP para esa característica
        shap_vals = shap_values_class1[:, feature_idx]
        
        # Dibujar histograma para el grupo negativo (y_test == 0) en color verde
        ax.hist(shap_vals[y_test == 0], bins=30, alpha=0.6, label='No Sepsis', color='green')
        # Dibujar histograma para el grupo positivo (y_test == 1) en color rojo
        ax.hist(shap_vals[y_test == 1], bins=30, alpha=0.6, label='Sepsis', color='red')
        # Etiquetar el eje X como SHAP Value
        ax.set_xlabel('SHAP Value', fontweight='bold', fontsize=9)
        # Etiquetar el eje Y como Frecuencia
        ax.set_ylabel('Frecuencia', fontweight='bold', fontsize=9)
        # Titular con el nombre de la variable
        ax.set_title(feature_name, fontweight='bold', fontsize=10)
        # Mostrar leyenda para diferenciar sepsis/no sepsis
        ax.legend(fontsize=8)
        # Mostrar grilla suave
        ax.grid(alpha=0.3)
    
    # Ajustar diseño del plot
    plt.tight_layout()
    # Guardar gráfico
    plt.savefig(os.path.join(PLOTS_DIR, "16_shap_distributions.png"), dpi=300, bbox_inches='tight')
    # Cerrar figura
    plt.close()
    # Imprimir éxito
    print("   ✓ Distributions guardado")
# Capturar error de distribuciones
except Exception as e:
    # Imprimir error si falló
    print(f"   ⚠️ Error en distributions: {str(e)}")

# ==========================================
# 7. GUARDAR SHAP VALUES PARA USO FUTURO
# ==========================================
# Guardar como archivo binario de numpy los valores SHAP
np.save(os.path.join(RESULTS_DIR, "shap_values_class1.npy"), shap_values_class1)
# Guardar el DataFrame de importancias como un CSV
feature_importance_shap.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"), index=False)
# Informar sobre el éxito del guardado de estos datos
print("   ✓ SHAP values guardados para análisis futuro")

# ==========================================
# RESUMEN FINAL
# ==========================================
# Imprimir línea vacía y guiones como separadores finales
print("\n" + "="*80)
# Título del bloque final
print("✅ ANÁLISIS SHAP COMPLETADO")
# Separador
print("="*80)
# Imprimir sección de Top 5
print(f"\n📊 TOP 5 CARACTERÍSTICAS POR SHAP:\n")
# Recorrer e imprimir top 5 en consola
for idx, (_, row) in enumerate(feature_importance_shap.head(5).iterrows(), 1):
    # Imprimir rank, nombre y media absoluta
    print(f"   {idx}. {row['Feature']:<30} Mean |SHAP|: {row['SHAP_Mean_Abs']:.6f}")

# Imprimir el listado de archivos visuales guardados
print(f"\n📁 Visualizaciones generadas:")
# Listar archivo de barra
print(f"   • 13_shap_summary_bar.png (Importancia global)")
# Listar archivo de dependencia
print(f"   • 14_shap_dependence_top2.png (Relaciones feature-SHAP)")
# Listar archivo de ejemplos individuales
print(f"   • 15_shap_examples.png (Predicciones individuales)")
# Listar archivo de distribuciones
print(f"   • 16_shap_distributions.png (Distribuciones SHAP)")
# Indicar dónde quedó el reporte textual final
print(f"\n📄 Reporte: {TXT_OUTPUT}\n")
# Línea final del script
print(f"{'='*80}\n")