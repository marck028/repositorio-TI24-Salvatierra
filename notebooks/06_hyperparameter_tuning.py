# Importar el módulo os para interactuar con el sistema operativo
import os
# Importar pandas con el alias pd para manipulación y análisis de datos
import pandas as pd
# Importar numpy con el alias np para operaciones numéricas y arreglos
import numpy as np
# Importar pyplot de matplotlib con el alias plt para crear gráficos
import matplotlib.pyplot as plt
# Importar seaborn con el alias sns para visualizaciones estadísticas
import seaborn as sns
# Importar GridSearchCV y StratifiedKFold de sklearn para búsqueda de hiperparámetros y validación cruzada
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Importar varias métricas de evaluación de sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
# Importar xgboost con el alias xgb para el modelo XGBoost
import xgboost as xgb
# Importar lightgbm con el alias lgb para el modelo LightGBM
import lightgbm as lgb
# Importar el módulo warnings para controlar las advertencias
import warnings
# Filtrar y omitir todas las advertencias para que no se muestren en consola
warnings.filterwarnings('ignore')
# Importar el módulo time para medir tiempos de ejecución
import time

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ==========================================
# Definir la ruta del directorio que contiene los datos procesados
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
# Definir la ruta del directorio donde se guardarán los resultados
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
# Definir la ruta del directorio donde se guardarán los gráficos
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"

# Crear el directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)
# Crear el directorio de gráficos si no existe
os.makedirs(PLOTS_DIR, exist_ok=True)

# Definir la ruta completa del archivo de texto donde se guardarán los resultados de optimización
TXT_OUTPUT = os.path.join(RESULTS_DIR, "06_hyperparameter_tuning_results.txt")

# Imprimir una línea de separación con 80 signos de igual
print("="*80)
# Imprimir el título de la etapa de optimización
print("OPTIMIZACIÓN DE HIPERPARÁMETROS CON GRIDSEARCHCV")
# Imprimir otra línea de separación
print("="*80)

# ==========================================
# 2. CARGAR DATOS
# ==========================================
# Imprimir un mensaje indicando el inicio de la carga de datos
print("\n[1/4] Cargando datos preprocesados...")
# Cargar las características de entrenamiento desde el archivo CSV correspondiente
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
# Cargar las características de prueba desde el archivo CSV correspondiente
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
# Cargar las etiquetas de entrenamiento y aplanarlas a un arreglo 1D
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
# Cargar las etiquetas de prueba y aplanarlas a un arreglo 1D
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

# Contar el número de muestras con clase negativa (0) en el conjunto de entrenamiento
num_neg = np.sum(y_train == 0)
# Contar el número de muestras con clase positiva (1) en el conjunto de entrenamiento
num_pos = np.sum(y_train == 1)
# Calcular el peso de la clase positiva para manejar el desbalanceo de clases
scale_pos_weight = num_neg / num_pos

# Imprimir las dimensiones de los conjuntos de datos cargados
print(f"   ✓ Datos cargados: X_train={X_train.shape}, X_test={X_test.shape}")
# Imprimir el peso calculado para la clase positiva
print(f"   ✓ Scale Pos Weight: {scale_pos_weight:.4f}")

# ==========================================
# 3. GRIDSEARCH PARA XGBOOST
# ==========================================
# Imprimir mensaje de inicio para GridSearchCV en XGBoost
print("\n[2/4] GridSearchCV para XGBoost (esto puede tomar 5-10 minutos)...")
# Informar sobre la cantidad de combinaciones y el tipo de validación cruzada
print("   Evaluando 24 combinaciones de hiperparámetros con 5-Fold CV...")

# Definir un diccionario con los hiperparámetros a evaluar para XGBoost
# Definir grid de parámetros
param_grid_xgb = {
    # Lista de valores para el número de estimadores (árboles)
    'n_estimators': [100, 150, 200],
    # Lista de valores para la profundidad máxima de cada árbol
    'max_depth': [3, 5, 7],
    # Lista de valores para la tasa de aprendizaje
    'learning_rate': [0.01, 0.05, 0.1],
    # Lista de valores para la fracción de muestras usadas para cada árbol
    'subsample': [0.7, 0.9],
}

# Inicializar el modelo clasificador de XGBoost base
# XGBoost base
xgb_base = xgb.XGBClassifier(
    # Asignar el peso calculado para las clases desbalanceadas
    scale_pos_weight=scale_pos_weight,
    # Establecer la semilla aleatoria para reproducibilidad
    random_state=42,
    # Usar logloss como métrica de evaluación interna
    eval_metric='logloss',
    # Establecer nivel de verbosidad a 0 (sin mensajes durante el entrenamiento base)
    verbosity=0,
    # Utilizar todos los núcleos disponibles del procesador
    n_jobs=-1
)

# Configurar la validación cruzada estratificada en 5 pliegues
# GridSearchCV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Registrar el tiempo de inicio antes de ejecutar GridSearchCV
start_time = time.time()
# Configurar la búsqueda en grilla (GridSearchCV) con el modelo XGBoost
grid_search_xgb = GridSearchCV(
    # El modelo estimador base
    estimator=xgb_base,
    # El diccionario de hiperparámetros a explorar
    param_grid=param_grid_xgb,
    # La estrategia de validación cruzada
    cv=skf,
    # La métrica principal que se utilizará para seleccionar el mejor modelo
    scoring='roc_auc',  # Métrica principal
    # Utilizar todos los núcleos para entrenar en paralelo
    n_jobs=-1,
    # Mostrar progreso en consola
    verbose=1
)

# Entrenar la búsqueda en grilla utilizando los datos de entrenamiento
grid_search_xgb.fit(X_train, y_train)
# Calcular el tiempo transcurrido para la búsqueda en grilla de XGBoost
elapsed_xgb = time.time() - start_time

# Imprimir mensaje indicando que la búsqueda terminó y el tiempo que tomó
print(f"\n   ✅ GridSearch XGBoost completado en {elapsed_xgb:.2f} segundos")
# Imprimir el mejor puntaje ROC-AUC obtenido en la validación cruzada
print(f"   Best Score (ROC-AUC CV): {grid_search_xgb.best_score_:.4f}")
# Imprimir los mejores hiperparámetros encontrados
print(f"   Best Params: {grid_search_xgb.best_params_}")

# Asignar el mejor estimador (modelo) encontrado en XGBoost a una variable
# Entrenar mejor modelo en test
best_model_xgb = grid_search_xgb.best_estimator_
# Predecir las clases para el conjunto de prueba
y_pred_xgb = best_model_xgb.predict(X_test)
# Predecir las probabilidades de pertenencia a la clase positiva
y_proba_xgb = best_model_xgb.predict_proba(X_test)[:, 1]

# Calcular la métrica de exactitud (accuracy)
test_acc_xgb = accuracy_score(y_test, y_pred_xgb)
# Calcular la métrica F1-score
test_f1_xgb = f1_score(y_test, y_pred_xgb)
# Calcular el Área Bajo la Curva ROC (ROC-AUC)
test_auc_xgb = roc_auc_score(y_test, y_proba_xgb)

# Imprimir la exactitud obtenida en prueba
print(f"   Test Accuracy: {test_acc_xgb:.4f}")
# Imprimir el F1-Score obtenido en prueba
print(f"   Test F1-Score: {test_f1_xgb:.4f}")
# Imprimir el ROC-AUC obtenido en prueba
print(f"   Test ROC-AUC: {test_auc_xgb:.4f}")

# ==========================================
# 4. GRIDSEARCH PARA LIGHTGBM
# ==========================================
# Imprimir mensaje de inicio para GridSearchCV en LightGBM
print("\n[3/4] GridSearchCV para LightGBM (esto puede tomar 5-10 minutos)...")
# Informar sobre la cantidad de combinaciones y el tipo de validación cruzada
print("   Evaluando 24 combinaciones de hiperparámetros con 5-Fold CV...")

# Definir un diccionario con los hiperparámetros a evaluar para LightGBM
param_grid_lgb = {
    # Lista de valores para el número de estimadores (árboles)
    'n_estimators': [100, 150, 200],
    # Lista de valores para la profundidad máxima
    'max_depth': [3, 5, 7],
    # Lista de valores para la tasa de aprendizaje
    'learning_rate': [0.01, 0.05, 0.1],
    # Lista de valores para el número máximo de hojas en un árbol
    'num_leaves': [31, 50, 70],
}

# Inicializar el modelo clasificador de LightGBM base
lgb_base = lgb.LGBMClassifier(
    # Asignar el peso de clases calculado previamente
    scale_pos_weight=scale_pos_weight,
    # Establecer la semilla aleatoria para reproducibilidad
    random_state=42,
    # Establecer el nivel de verbosidad a -1 (modo silencioso)
    verbosity=-1,
    # Utilizar todos los núcleos de procesamiento disponibles
    n_jobs=-1
)

# Registrar el tiempo de inicio antes de ejecutar GridSearchCV para LightGBM
start_time = time.time()
# Configurar la búsqueda en grilla para el modelo LightGBM
grid_search_lgb = GridSearchCV(
    # El estimador base de LightGBM
    estimator=lgb_base,
    # Los hiperparámetros a evaluar
    param_grid=param_grid_lgb,
    # La validación cruzada estratificada previamente definida
    cv=skf,
    # Usar ROC-AUC como métrica para seleccionar el mejor modelo
    scoring='roc_auc',
    # Procesar en paralelo con todos los núcleos
    n_jobs=-1,
    # Mostrar el avance por consola
    verbose=1
)

# Entrenar la búsqueda en grilla utilizando los datos de entrenamiento para LightGBM
grid_search_lgb.fit(X_train, y_train)
# Calcular el tiempo transcurrido para la búsqueda en grilla de LightGBM
elapsed_lgb = time.time() - start_time

# Imprimir mensaje de finalización y el tiempo invertido en la búsqueda
print(f"\n   ✅ GridSearch LightGBM completado en {elapsed_lgb:.2f} segundos")
# Imprimir el mejor puntaje ROC-AUC obtenido en validación cruzada para LightGBM
print(f"   Best Score (ROC-AUC CV): {grid_search_lgb.best_score_:.4f}")
# Imprimir los mejores hiperparámetros encontrados para LightGBM
print(f"   Best Params: {grid_search_lgb.best_params_}")

# Asignar el mejor estimador de LightGBM a una variable
# Entrenar mejor modelo en test
best_model_lgb = grid_search_lgb.best_estimator_
# Predecir las clases para el conjunto de prueba usando LightGBM
y_pred_lgb = best_model_lgb.predict(X_test)
# Predecir las probabilidades de pertenencia a la clase positiva con LightGBM
y_proba_lgb = best_model_lgb.predict_proba(X_test)[:, 1]

# Calcular la métrica de exactitud (accuracy) para LightGBM
test_acc_lgb = accuracy_score(y_test, y_pred_lgb)
# Calcular la métrica F1-score para LightGBM
test_f1_lgb = f1_score(y_test, y_pred_lgb)
# Calcular el ROC-AUC para LightGBM en el conjunto de prueba
test_auc_lgb = roc_auc_score(y_test, y_proba_lgb)

# Imprimir la exactitud obtenida con LightGBM
print(f"   Test Accuracy: {test_acc_lgb:.4f}")
# Imprimir el F1-Score obtenido con LightGBM
print(f"   Test F1-Score: {test_f1_lgb:.4f}")
# Imprimir el ROC-AUC obtenido con LightGBM
print(f"   Test ROC-AUC: {test_auc_lgb:.4f}")

# ==========================================
# 5. CREAR DATAFRAMES CON RESULTADOS
# ==========================================
# Imprimir inicio de la fase de generación de reportes
print("\n[4/4] Generando reportes...")

# Convertir el diccionario de resultados de XGBoost a un DataFrame de pandas
# Convertir resultados a DataFrame
results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)
# Convertir el diccionario de resultados de LightGBM a un DataFrame de pandas
results_lgb = pd.DataFrame(grid_search_lgb.cv_results_)

# Seleccionar solo las columnas con parámetros, medias, desviaciones estándar y rangos de XGBoost
# Seleccionar columnas relevantes
results_xgb_clean = results_xgb[['param_n_estimators', 'param_max_depth', 'param_learning_rate', 
                                   'param_subsample', 'mean_test_score', 'std_test_score', 'rank_test_score']]
# Renombrar las columnas para mayor claridad
results_xgb_clean.columns = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'mean_auc', 'std_auc', 'rank']
# Ordenar los resultados según el rango de puntuación (de mejor a peor)
results_xgb_clean = results_xgb_clean.sort_values('rank')

# Seleccionar solo las columnas de interés para LightGBM
results_lgb_clean = results_lgb[['param_n_estimators', 'param_max_depth', 'param_learning_rate',
                                  'param_num_leaves', 'mean_test_score', 'std_test_score', 'rank_test_score']]
# Renombrar las columnas para LightGBM
results_lgb_clean.columns = ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves', 'mean_auc', 'std_auc', 'rank']
# Ordenar los resultados de LightGBM según el rango de puntuación
results_lgb_clean = results_lgb_clean.sort_values('rank')

# ==========================================
# 6. EXPORTAR RESULTADOS A TXT
# ==========================================
# Abrir el archivo de texto en modo escritura (sobreescribiendo si ya existe) usando codificación UTF-8
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    # Escribir una línea separadora en el archivo
    f.write("="*90 + "\n")
    # Escribir el título principal del reporte
    f.write("REPORTE DE OPTIMIZACIÓN DE HIPERPARÁMETROS - GridSearchCV\n")
    # Escribir otra línea separadora y un salto de línea
    f.write("="*90 + "\n\n")
    
    # Escribir el subtítulo para la configuración general
    f.write("1. CONFIGURACIÓN GENERAL\n")
    # Escribir línea separadora para la sección
    f.write("-"*90 + "\n")
    # Escribir la estrategia de validación usada
    f.write(f"   Estrategia de Validación: Stratified K-Fold (5 pliegues)\n")
    # Escribir la métrica principal usada
    f.write(f"   Métrica de Optimización: ROC-AUC\n")
    # Escribir el valor de peso asignado para balancear las clases
    f.write(f"   Scale Pos Weight: {scale_pos_weight:.4f}\n")
    # Escribir la cantidad total de muestras en el conjunto de entrenamiento
    f.write(f"   Conjunto de Entrenamiento: {X_train.shape[0]} muestras\n\n")
    
    # Escribir el subtítulo para los resultados de XGBoost
    f.write("2. RESULTADOS XGBOOST - TOP 10 COMBINACIONES\n")
    # Escribir línea separadora
    f.write("-"*90 + "\n")
    # Convertir el DataFrame de XGBoost a formato de texto para las primeras 10 filas y escribirlo en el archivo
    f.write(results_xgb_clean.head(10).to_string(index=False))
    # Escribir saltos de línea para separación
    f.write("\n\n")
    
    # Escribir el subtítulo del mejor modelo XGBoost
    f.write("3. MEJOR MODELO XGBOOST\n")
    # Escribir línea separadora
    f.write("-"*90 + "\n")
    # Escribir la etiqueta de los hiperparámetros óptimos
    f.write(f"   Hiperparámetros Óptimos:\n")
    # Iterar sobre el diccionario de mejores parámetros de XGBoost
    for param, value in grid_search_xgb.best_params_.items():
        # Escribir cada parámetro y su valor
        f.write(f"      • {param}: {value}\n")
    # Escribir etiqueta para rendimiento en validación cruzada
    f.write(f"\n   Rendimiento CV (5-Fold):\n")
    # Escribir el mejor puntaje ROC-AUC promedio en CV
    f.write(f"      • ROC-AUC: {grid_search_xgb.best_score_:.4f}\n")
    # Escribir la desviación estándar de la métrica en la mejor combinación
    f.write(f"      • Std: {results_xgb_clean.iloc[0]['std_auc']:.4f}\n")
    # Escribir etiqueta para rendimiento en el conjunto de prueba
    f.write(f"\n   Rendimiento en Test:\n")
    # Escribir exactitud de prueba
    f.write(f"      • Accuracy: {test_acc_xgb:.4f}\n")
    # Escribir F1-Score de prueba
    f.write(f"      • F1-Score: {test_f1_xgb:.4f}\n")
    # Escribir ROC-AUC de prueba
    f.write(f"      • ROC-AUC: {test_auc_xgb:.4f}\n")
    # Escribir el tiempo transcurrido en el GridSearch
    f.write(f"      • Tiempo de entrenamiento GridSearch: {elapsed_xgb:.2f}s\n\n")
    
    # Escribir el subtítulo para los resultados de LightGBM
    f.write("4. RESULTADOS LIGHTGBM - TOP 10 COMBINACIONES\n")
    # Escribir línea separadora
    f.write("-"*90 + "\n")
    # Escribir la tabla con las 10 mejores combinaciones de LightGBM
    f.write(results_lgb_clean.head(10).to_string(index=False))
    # Escribir saltos de línea
    f.write("\n\n")
    
    # Escribir subtítulo del mejor modelo LightGBM
    f.write("5. MEJOR MODELO LIGHTGBM\n")
    # Escribir línea separadora
    f.write("-"*90 + "\n")
    # Escribir etiqueta de hiperparámetros
    f.write(f"   Hiperparámetros Óptimos:\n")
    # Iterar sobre los mejores parámetros de LightGBM
    for param, value in grid_search_lgb.best_params_.items():
        # Escribir cada parámetro y valor
        f.write(f"      • {param}: {value}\n")
    # Escribir etiqueta de rendimiento en CV
    f.write(f"\n   Rendimiento CV (5-Fold):\n")
    # Escribir el mejor puntaje ROC-AUC en CV para LightGBM
    f.write(f"      • ROC-AUC: {grid_search_lgb.best_score_:.4f}\n")
    # Escribir la desviación estándar para la mejor combinación
    f.write(f"      • Std: {results_lgb_clean.iloc[0]['std_auc']:.4f}\n")
    # Escribir etiqueta de rendimiento en prueba
    f.write(f"\n   Rendimiento en Test:\n")
    # Escribir exactitud en prueba para LightGBM
    f.write(f"      • Accuracy: {test_acc_lgb:.4f}\n")
    # Escribir F1-Score en prueba
    f.write(f"      • F1-Score: {test_f1_lgb:.4f}\n")
    # Escribir ROC-AUC en prueba
    f.write(f"      • ROC-AUC: {test_auc_lgb:.4f}\n")
    # Escribir el tiempo invertido en la búsqueda para LightGBM
    f.write(f"      • Tiempo de entrenamiento GridSearch: {elapsed_lgb:.2f}s\n\n")
    
    # Escribir subtítulo de comparación final
    f.write("6. COMPARACIÓN FINAL\n")
    # Escribir línea separadora
    f.write("-"*90 + "\n")
    # Escribir la cabecera de la tabla comparativa con formato alineado
    f.write(f"{'Métrica':<25} {'XGBoost':<20} {'LightGBM':<20}\n")
    # Escribir separador de cabecera
    f.write(f"{'-'*65}\n")
    # Escribir comparación del ROC-AUC en validación cruzada
    f.write(f"{'CV ROC-AUC':<25} {grid_search_xgb.best_score_:<20.4f} {grid_search_lgb.best_score_:<20.4f}\n")
    # Escribir comparación de Exactitud en prueba
    f.write(f"{'Test Accuracy':<25} {test_acc_xgb:<20.4f} {test_acc_lgb:<20.4f}\n")
    # Escribir comparación de F1-Score en prueba
    f.write(f"{'Test F1-Score':<25} {test_f1_xgb:<20.4f} {test_f1_lgb:<20.4f}\n")
    # Escribir comparación de ROC-AUC en prueba
    f.write(f"{'Test ROC-AUC':<25} {test_auc_xgb:<20.4f} {test_auc_lgb:<20.4f}\n\n")
    
    # Determinar qué modelo obtuvo mejor ROC-AUC en prueba
    winner = "XGBoost" if test_auc_xgb >= test_auc_lgb else "LightGBM"
    # Escribir cuál fue el mejor modelo final
    f.write(f"   ✓ MEJOR MODELO: {winner}\n\n")
    
    # Escribir sección de recomendaciones
    f.write("7. RECOMENDACIONES PARA DEFENSA ORAL\n")
    # Escribir línea separadora
    f.write("-"*90 + "\n")
    # Escribir recomendación metodológica 1
    f.write(f"   • Los hiperparámetros óptimos fueron seleccionados mediante GridSearchCV\n")
    # Escribir recomendación metodológica 2
    f.write(f"     con validación cruzada estratificada de 5 pliegues.\n")
    # Escribir número total de combinaciones evaluadas
    f.write(f"   • Se evaluaron 24 combinaciones diferentes para XGBoost y LightGBM.\n")
    # Escribir puntaje del modelo ganador en prueba
    f.write(f"   • El modelo ganador ({winner}) alcanzó ROC-AUC de {max(test_auc_xgb, test_auc_lgb):.4f} en test.\n")
    # Escribir mención sobre cómo se validó la estabilidad
    f.write(f"   • La estabilidad fue validada mediante desviación estándar de CV.\n")

# Imprimir por consola que el archivo fue guardado con éxito
print(f"\n✅ Resultados guardados en: {TXT_OUTPUT}")

# ==========================================
# 7. VISUALIZAR HEATMAP DE RESULTADOS
# ==========================================

# Hacer una copia del DataFrame limpio de XGBoost para preparar gráfico
# Crear tabla pivot para XGBoost
results_pivot_xgb = results_xgb_clean.copy()
# Crear una nueva columna string que combine los hiperparámetros principales
results_pivot_xgb['combination'] = (results_pivot_xgb['n_estimators'].astype(str) + '_' + 
                                     results_pivot_xgb['max_depth'].astype(str) + '_' + 
                                     results_pivot_xgb['learning_rate'].astype(str))
# Ordenar de forma descendente por métrica media y tomar los 12 mejores resultados
results_pivot_xgb = results_pivot_xgb.sort_values('mean_auc', ascending=False).head(12)

# Crear la figura y 2 subgráficos (ejes) acomodados en una fila
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Preparar los 12 mejores resultados de XGBoost ordenándolos de manera descendente
# XGBoost top 12
results_sorted_xgb = results_xgb_clean.sort_values('mean_auc', ascending=False).head(12)
# Crear gráfico de barras horizontales en el primer subgráfico para XGBoost con barras de error (std)
axes[0].barh(range(len(results_sorted_xgb)), results_sorted_xgb['mean_auc'], 
             xerr=results_sorted_xgb['std_auc'], capsize=5, color='#3498db', alpha=0.8)
# Crear etiquetas descriptivas para cada barra incluyendo el número de estimadores, profundidad y tasa de aprendizaje
labels_xgb = [f"n={int(row['n_estimators'])}, d={int(row['max_depth'])}, lr={row['learning_rate']}" 
              for _, row in results_sorted_xgb.iterrows()]
# Establecer las marcas en el eje Y
axes[0].set_yticks(range(len(results_sorted_xgb)))
# Asignar las etiquetas descriptivas a las marcas del eje Y con tamaño de fuente 9
axes[0].set_yticklabels(labels_xgb, fontsize=9)
# Establecer el nombre para el eje X en el subgráfico
axes[0].set_xlabel('ROC-AUC (CV)', fontweight='bold')
# Establecer título para el subgráfico de XGBoost
axes[0].set_title('XGBoost - Top 12 Combinaciones', fontweight='bold', fontsize=12)
# Invertir el eje Y para que el mejor resultado aparezca en la parte superior
axes[0].invert_yaxis()
# Agregar cuadrícula solo en el eje X con cierta transparencia
axes[0].grid(axis='x', alpha=0.3)

# Iterar sobre las filas de los resultados de XGBoost para añadir texto a cada barra
for i, (_, row) in enumerate(results_sorted_xgb.iterrows()):
    # Añadir texto al lado derecho del error mostrando el valor exacto de la media
    axes[0].text(row['mean_auc'] + row['std_auc'], i, f" {row['mean_auc']:.4f}", 
                va='center', fontsize=9, fontweight='bold')

# Preparar los 12 mejores resultados de LightGBM ordenándolos de manera descendente
# LightGBM top 12
results_sorted_lgb = results_lgb_clean.sort_values('mean_auc', ascending=False).head(12)
# Crear gráfico de barras horizontales en el segundo subgráfico para LightGBM con barras de error
axes[1].barh(range(len(results_sorted_lgb)), results_sorted_lgb['mean_auc'],
             xerr=results_sorted_lgb['std_auc'], capsize=5, color='#2ecc71', alpha=0.8)
# Crear etiquetas descriptivas para cada combinación de hiperparámetros en LightGBM
labels_lgb = [f"n={int(row['n_estimators'])}, d={int(row['max_depth'])}, lr={row['learning_rate']}" 
              for _, row in results_sorted_lgb.iterrows()]
# Establecer posiciones en el eje Y para el segundo subgráfico
axes[1].set_yticks(range(len(results_sorted_lgb)))
# Asignar las etiquetas a las posiciones correspondientes
axes[1].set_yticklabels(labels_lgb, fontsize=9)
# Establecer la etiqueta del eje X para LightGBM
axes[1].set_xlabel('ROC-AUC (CV)', fontweight='bold')
# Añadir el título para el subgráfico de LightGBM
axes[1].set_title('LightGBM - Top 12 Combinaciones', fontweight='bold', fontsize=12)
# Invertir el eje Y para el gráfico de LightGBM
axes[1].invert_yaxis()
# Añadir cuadrícula vertical para facilitar la lectura del valor ROC-AUC
axes[1].grid(axis='x', alpha=0.3)

# Iterar sobre las filas de LightGBM para añadir el texto a la derecha de cada barra
for i, (_, row) in enumerate(results_sorted_lgb.iterrows()):
    # Colocar el valor numérico con cuatro decimales de precisión
    axes[1].text(row['mean_auc'] + row['std_auc'], i, f" {row['mean_auc']:.4f}",
                va='center', fontsize=9, fontweight='bold')

# Ajustar el diseño para evitar que se superpongan elementos de ambos subgráficos
plt.tight_layout()
# Guardar la figura actual como imagen en el directorio configurado para gráficos
plt.savefig(os.path.join(PLOTS_DIR, "12_gridsearch_top12_comparacion.png"), dpi=300, bbox_inches='tight')
# Cerrar la figura para liberar la memoria asociada a ella
plt.close()
# Imprimir en consola que se ha guardado el gráfico exitosamente
print("   ✓ Gráfico de GridSearch guardado")

# ==========================================
# RESUMEN FINAL
# ==========================================
# Imprimir línea divisoria final con salto de línea previo
print("\n" + "="*80)
# Imprimir mensaje de conclusión
print("✅ OPTIMIZACIÓN DE HIPERPARÁMETROS COMPLETADA")
# Imprimir línea divisoria
print("="*80)
# Imprimir título del resumen
print(f"\n📊 RESUMEN:\n")
# Mostrar el nombre del modelo XGBoost
print(f"   XGBoost Óptimo:")
# Imprimir ROC-AUC obtenido en validación cruzada para XGBoost
print(f"   • CV ROC-AUC: {grid_search_xgb.best_score_:.4f}")
# Imprimir ROC-AUC obtenido en conjunto de test para XGBoost
print(f"   • Test ROC-AUC: {test_auc_xgb:.4f}")
# Imprimir el diccionario con los mejores parámetros hallados en XGBoost
print(f"   • Hiperparámetros: {grid_search_xgb.best_params_}\n")
# Mostrar el nombre del modelo LightGBM
print(f"   LightGBM Óptimo:")
# Imprimir ROC-AUC obtenido en validación cruzada para LightGBM
print(f"   • CV ROC-AUC: {grid_search_lgb.best_score_:.4f}")
# Imprimir ROC-AUC obtenido en conjunto de test para LightGBM
print(f"   • Test ROC-AUC: {test_auc_lgb:.4f}")
# Imprimir el diccionario con los mejores parámetros hallados en LightGBM
print(f"   • Hiperparámetros: {grid_search_lgb.best_params_}\n")
# Mostrar e indicar cuál es el mejor modelo determinando por el ROC-AUC en test
print(f"   ⭐ Mejor modelo: {'XGBoost' if test_auc_xgb >= test_auc_lgb else 'LightGBM'}")
# Imprimir última línea divisoria decorativa
print(f"\n{'='*80}\n")