# Importar el módulo os para interactuar con el sistema operativo
import os
# Importar la librería pandas con el alias pd para manipulación y análisis de datos
import pandas as pd
# Importar la librería numpy con el alias np para cálculos numéricos y arreglos
import numpy as np
# Importar pyplot de matplotlib con el alias plt para la creación de gráficos
import matplotlib.pyplot as plt
# Importar seaborn con el alias sns para visualización estadística basada en matplotlib
import seaborn as sns
# Importar múltiples funciones de métricas de evaluación del módulo sklearn.metrics
from sklearn.metrics import (
# Importar métricas específicas como accuracy_score, f1_score, confusion_matrix, etc.
    accuracy_score, f1_score, confusion_matrix, classification_report, 
# Importar métricas para curvas ROC y precisión-recall
    roc_curve, auc, roc_auc_score, precision_recall_curve
# Cerrar paréntesis de importación de métricas
)
# Importar clases para validación cruzada del módulo sklearn.model_selection
from sklearn.model_selection import StratifiedKFold, cross_validate
# Importar la librería xgboost con el alias xgb para modelos de gradient boosting
import xgboost as xgb
# Importar la librería lightgbm con el alias lgb para modelos rápidos de gradient boosting
import lightgbm as lgb
# Importar el módulo warnings para el manejo de advertencias en ejecución
import warnings
# Configurar el filtro de advertencias para ignorar (no mostrar) las advertencias
warnings.filterwarnings('ignore')

# Línea de comentario decorativo para separar secciones
# ==========================================
# Línea de comentario indicando la sección 1 (configuración de rutas)
# 1. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# Línea de comentario decorativo
# ==========================================
# Definir la variable PROCESSED_DIR con la ruta al directorio de datos procesados
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
# Definir la variable RESULTS_DIR con la ruta al directorio de resultados
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
# Definir la variable PLOTS_DIR con la ruta al directorio de gráficos
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"

# Comentario explicando que se crearán directorios si no existen
# Crear directorio si no existe
# Crear el directorio de resultados si no existe, sin lanzar error si ya está presente
os.makedirs(RESULTS_DIR, exist_ok=True)
# Crear el directorio de gráficos si no existe, sin lanzar error si ya está presente
os.makedirs(PLOTS_DIR, exist_ok=True)

# Unir la ruta RESULTS_DIR con el nombre del archivo de salida de texto
TXT_OUTPUT = os.path.join(RESULTS_DIR, "04_model_main_MEJORADO.txt")

# Imprimir una línea de 70 signos de igual como separador visual
print("="*70)
# Imprimir el título del script o proceso actual en consola
print("MODELO PRINCIPAL MEJORADO CON VALIDACIÓN CRUZADA")
# Imprimir otra línea de 70 signos de igual como separador
print("="*70)

# Línea de comentario decorativo
# ==========================================
# Línea de comentario indicando la sección 2 (cargar datos)
# 2. CARGAR DATOS PREPROCESADOS
# Línea de comentario decorativo
# ==========================================
# Imprimir mensaje indicando el paso 1 de 6: carga de datos
print("\n[1/6] Cargando datos preprocesados...")
# Iniciar bloque try para el manejo de excepciones en la lectura de archivos
try:
# Leer el archivo CSV de datos de entrenamiento X y guardarlo en X_train
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
# Leer el archivo CSV de datos de prueba X y guardarlo en X_test
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
# Leer el archivo CSV de etiquetas de entrenamiento y convertirlo a array unidimensional
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
# Leer el archivo CSV de etiquetas de prueba y convertirlo a array unidimensional
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()
# Imprimir la forma (dimensiones) del dataframe de entrenamiento X_train
    print(f"   ✓ X_train: {X_train.shape}")
# Imprimir la forma (dimensiones) del dataframe de prueba X_test
    print(f"   ✓ X_test: {X_test.shape}")
# Imprimir la forma del array y_train y contar las clases positivas (Sepsis) y negativas (No Sepsis)
    print(f"   ✓ y_train: {y_train.shape} (Sepsis: {y_train.sum()}, No Sepsis: {(1-y_train).sum()})")
# Imprimir la forma del array y_test y contar las clases positivas (Sepsis) y negativas (No Sepsis)
    print(f"   ✓ y_test: {y_test.shape} (Sepsis: {y_test.sum()}, No Sepsis: {(1-y_test).sum()})")
# Bloque except para atrapar cualquier excepción que ocurra en el bloque try
except Exception as e:
# Imprimir mensaje de error junto con la descripción de la excepción
    print(f"   ✗ Error cargando datos: {e}")
# Terminar la ejecución del script si ocurrió un error al cargar datos
    exit()

# Línea de comentario decorativo
# ==========================================
# Línea de comentario indicando la sección 3 (configuración del modelo)
# 3. CONFIGURACIÓN DEL MODELO Y PARÁMETROS
# Línea de comentario decorativo
# ==========================================
# Imprimir mensaje indicando el paso 2 de 6: configuración de XGBoost
print("\n[2/6] Configurando modelo XGBoost...")

# Comentario sobre el cálculo del factor de desbalanceo de clases
# Calcular factor de desbalanceo
# Contar el número de ejemplos negativos en y_train (clase 0)
num_neg = np.sum(y_train == 0)
# Contar el número de ejemplos positivos en y_train (clase 1)
num_pos = np.sum(y_train == 1)
# Calcular el peso para la clase positiva dividiendo el número de negativos entre los positivos
scale_pos_weight = num_neg / num_pos

# Imprimir en consola el ratio calculado entre clase negativa y positiva
print(f"   ✓ Ratio negativo/positivo: {scale_pos_weight:.4f}")
# Imprimir en consola el valor del parámetro scale_pos_weight que se utilizará
print(f"   ✓ Scale_pos_weight configurado: {scale_pos_weight:.4f}")

# Comentario indicando la definición de hiperparámetros
# Hiperparámetros base
# Crear un diccionario con los hiperparámetros del modelo XGBoost
hyperparams = {
# Definir el número de árboles (estimadores) en el modelo a 150
    'n_estimators': 150,
# Definir la profundidad máxima de cada árbol en 5
    'max_depth': 5,
# Definir la tasa de aprendizaje (learning rate) en 0.05
    'learning_rate': 0.05,
# Asignar el peso previamente calculado para balancear las clases
    'scale_pos_weight': scale_pos_weight,
# Establecer la semilla aleatoria en 42 para reproducibilidad
    'random_state': 42,
# Configurar la métrica de evaluación interna a entropía cruzada logarítmica (logloss)
    'eval_metric': 'logloss',
# Establecer el nivel de detalle de los mensajes durante el entrenamiento a 0 (silencioso)
    'verbosity': 0
# Cierre del diccionario de hiperparámetros
}

# Imprimir título para listar los hiperparámetros
print(f"   Hiperparámetros:")
# Iterar sobre las claves y valores del diccionario de hiperparámetros
for key, val in hyperparams.items():
# Imprimir en consola cada nombre de parámetro y su respectivo valor
    print(f"     - {key}: {val}")

# Línea de comentario decorativo
# ==========================================
# Línea de comentario indicando la sección 4 (validación cruzada)
# 4. VALIDACIÓN CRUZADA ESTRATIFICADA (K-FOLD)
# Línea de comentario decorativo
# ==========================================
# Imprimir mensaje indicando el paso 3 de 6: ejecución de validación cruzada
print("\n[3/6] Ejecutando Validación Cruzada Estratificada (5-Fold)...")

# Comentario sobre la definición del objeto StratifiedKFold
# Definir StratifiedKFold
# Inicializar StratifiedKFold con 5 particiones, mezclado aleatorio y semilla fija
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Comentario sobre la inicialización de listas para guardar resultados por partición (fold)
# Listas para almacenar métricas de cada fold
# Crear un diccionario para recolectar las métricas de evaluación de cada iteración del fold
fold_results = {
# Lista para guardar el número del fold
    'fold': [],
# Lista para guardar la exactitud (accuracy)
    'accuracy': [],
# Lista para guardar el puntaje F1 (f1_score)
    'f1_score': [],
# Lista para guardar el área bajo la curva ROC (roc_auc)
    'roc_auc': [],
# Lista para guardar la precisión
    'precision': [],
# Lista para guardar la sensibilidad o recall
    'recall': [],
# Lista para guardar el conteo de falsos negativos
    'fn_count': [],
# Lista para guardar el conteo de falsos positivos
    'fp_count': []
# Cierre del diccionario de resultados por fold
}

# Comentario sobre listas para almacenar predicciones y probabilidades acumuladas de todos los folds
# Almacenar predicciones y probabilidades de todos los folds
# Inicializar lista para guardar valores verdaderos de validación de todos los folds
all_y_true_cv = []
# Inicializar lista para guardar predicciones binarias de validación de todos los folds
all_y_pred_cv = []
# Inicializar lista para guardar probabilidades predichas de validación de todos los folds
all_y_proba_cv = []

# Inicializar un contador de fold comenzando en 1
fold_number = 1
# Bucle para iterar sobre cada partición de índices de entrenamiento y validación generada por skf
for train_idx, val_idx in skf.split(X_train, y_train):
# Comentario sobre la división de datos en cada fold
    # Dividir datos
# Extraer los datos de entrenamiento para este fold usando sus índices
    X_fold_train = X_train.iloc[train_idx]
# Extraer los datos de validación para este fold usando sus índices
    X_fold_val = X_train.iloc[val_idx]
# Extraer las etiquetas de entrenamiento para este fold
    y_fold_train = y_train[train_idx]
# Extraer las etiquetas de validación para este fold
    y_fold_val = y_train[val_idx]
    
# Comentario sobre el entrenamiento del modelo en este fold
    # Entrenar modelo
# Instanciar el clasificador XGBoost usando los hiperparámetros definidos previamente
    model_fold = xgb.XGBClassifier(**hyperparams)
# Entrenar el clasificador XGBoost usando los datos de entrenamiento de este fold
    model_fold.fit(X_fold_train, y_fold_train)
    
# Comentario sobre el proceso de inferencia/predicción
    # Predicciones
# Generar predicciones binarias sobre el conjunto de validación del fold
    y_pred_fold = model_fold.predict(X_fold_val)
# Obtener probabilidades continuas de pertenecer a la clase positiva (índice 1) en validación
    y_proba_fold = model_fold.predict_proba(X_fold_val)[:, 1]
    
# Comentario sobre el cálculo de métricas de clasificación
    # Calcular métricas
# Calcular la exactitud (accuracy) comparando las etiquetas reales con las predichas
    acc = accuracy_score(y_fold_val, y_pred_fold)
# Calcular el puntaje F1 comparando las etiquetas reales con las predichas
    f1 = f1_score(y_fold_val, y_pred_fold)
# Calcular el ROC-AUC utilizando las etiquetas reales y las probabilidades predichas
    roc_auc = roc_auc_score(y_fold_val, y_proba_fold)
    
# Comentario sobre el cálculo de la precisión y el recall
    # Precisión y recall
# Importar específicamente precision_score y recall_score desde sklearn
    from sklearn.metrics import precision_score, recall_score
# Calcular la precisión manejando el caso de divisiones por cero devolviendo cero
    prec = precision_score(y_fold_val, y_pred_fold, zero_division=0)
# Calcular el recall (sensibilidad) manejando divisiones por cero devolviendo cero
    rec = recall_score(y_fold_val, y_pred_fold, zero_division=0)
    
# Comentario indicando la extracción de componentes de la matriz de confusión
    # Matriz de confusión
# Calcular la matriz de confusión completa para el fold actual
    cm = confusion_matrix(y_fold_val, y_pred_fold)
# Extraer el número de Falsos Negativos (fila de reales positivos, columna de predicción negativos)
    fn = cm[1, 0]  # Falsos Negativos
# Extraer el número de Falsos Positivos (fila de reales negativos, columna de predicción positivos)
    fp = cm[0, 1]  # Falsos Positivos
    
# Comentario sobre guardar las métricas de esta iteración
    # Guardar resultados
# Agregar el número del fold actual a la lista en el diccionario
    fold_results['fold'].append(fold_number)
# Agregar la exactitud (accuracy) del fold a la lista en el diccionario
    fold_results['accuracy'].append(acc)
# Agregar el puntaje F1 del fold a la lista en el diccionario
    fold_results['f1_score'].append(f1)
# Agregar el ROC-AUC del fold a la lista en el diccionario
    fold_results['roc_auc'].append(roc_auc)
# Agregar la precisión del fold a la lista en el diccionario
    fold_results['precision'].append(prec)
# Agregar el recall del fold a la lista en el diccionario
    fold_results['recall'].append(rec)
# Agregar el conteo de Falsos Negativos del fold a la lista en el diccionario
    fold_results['fn_count'].append(fn)
# Agregar el conteo de Falsos Positivos del fold a la lista en el diccionario
    fold_results['fp_count'].append(fp)
    
# Comentario sobre guardar listas combinadas para analizar el desempeño general después
    # Almacenar para cálculo de curva ROC promedia
# Extender la lista global de etiquetas reales con las del fold actual
    all_y_true_cv.extend(y_fold_val)
# Extender la lista global de predicciones con las del fold actual
    all_y_pred_cv.extend(y_pred_fold)
# Extender la lista global de probabilidades predichas con las del fold actual
    all_y_proba_cv.extend(y_proba_fold)
    
# Imprimir resumen de métricas del fold actual
    print(f"   Fold {fold_number}/5: Acc={acc:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}, FN={fn}, FP={fp}")
# Incrementar el contador de fold para la siguiente iteración
    fold_number += 1

# Comentario sobre la conversión del diccionario de resultados en un DataFrame
# Crear DataFrame con resultados de folds
# Convertir el diccionario con los resultados individuales por fold en un DataFrame de pandas
df_folds = pd.DataFrame(fold_results)

# Comentario sobre imprimir resúmenes estadísticos de todas las iteraciones de la CV
# Calcular estadísticas de CV
# Imprimir un título de sección de estadísticas de la validación cruzada
print(f"\n   📊 ESTADÍSTICAS DE VALIDACIÓN CRUZADA:")
# Imprimir el promedio y la desviación estándar de la exactitud entre todos los folds
print(f"   ┌─ Accuracy:   {df_folds['accuracy'].mean():.4f} ± {df_folds['accuracy'].std():.4f}")
# Imprimir el promedio y la desviación estándar del puntaje F1 entre todos los folds
print(f"   ├─ F1-Score:   {df_folds['f1_score'].mean():.4f} ± {df_folds['f1_score'].std():.4f}")
# Imprimir el promedio y la desviación estándar del ROC-AUC entre todos los folds
print(f"   ├─ ROC-AUC:    {df_folds['roc_auc'].mean():.4f} ± {df_folds['roc_auc'].std():.4f}")
# Imprimir el promedio y la desviación estándar de la precisión entre todos los folds
print(f"   ├─ Precision:  {df_folds['precision'].mean():.4f} ± {df_folds['precision'].std():.4f}")
# Imprimir el promedio y la desviación estándar del recall entre todos los folds
print(f"   ├─ Recall:     {df_folds['recall'].mean():.4f} ± {df_folds['recall'].std():.4f}")
# Imprimir la suma total y el promedio por fold de Falsos Negativos
print(f"   ├─ FN Total:   {df_folds['fn_count'].sum()} (Media: {df_folds['fn_count'].mean():.1f})")
# Imprimir la suma total y el promedio por fold de Falsos Positivos
print(f"   └─ FP Total:   {df_folds['fp_count'].sum()} (Media: {df_folds['fp_count'].mean():.1f})")

# Línea decorativa
# ==========================================
# Título indicando el quinto paso del pipeline: entrenamiento final
# 5. ENTRENAR MODELO FINAL EN TODO TRAIN SET
# Línea decorativa
# ==========================================
# Imprimir en consola el paso 4/6 correspondiente al entrenamiento del modelo final
print("\n[4/6] Entrenando modelo final en conjunto de entrenamiento completo...")

# Instanciar el clasificador XGBoost definitivo con los mismos hiperparámetros definidos antes
model_final = xgb.XGBClassifier(**hyperparams)
# Entrenar el clasificador definitivo usando el conjunto completo de entrenamiento (X_train, y_train)
model_final.fit(X_train, y_train)

# Comentario indicando que se realizarán predicciones en el conjunto de pruebas
# Predicciones en test
# Utilizar el modelo definitivo entrenado para predecir sobre el conjunto de test
y_pred_test = model_final.predict(X_test)
# Obtener las probabilidades de pertenencia a la clase 1 para el conjunto de test
y_proba_test = model_final.predict_proba(X_test)[:, 1]

# Comentario indicando cálculo de métricas para test
# Métricas en test
# Calcular la exactitud final sobre el conjunto de test
acc_test = accuracy_score(y_test, y_pred_test)
# Calcular el puntaje F1 final sobre el conjunto de test
f1_test = f1_score(y_test, y_pred_test)
# Calcular el ROC-AUC final usando probabilidades de test
roc_auc_test = roc_auc_score(y_test, y_proba_test)

# Volver a importar precision y recall para asegurar su disponibilidad
from sklearn.metrics import precision_score, recall_score
# Calcular precisión de test, manejando divisiones por cero
prec_test = precision_score(y_test, y_pred_test, zero_division=0)
# Calcular recall de test, manejando divisiones por cero
rec_test = recall_score(y_test, y_pred_test, zero_division=0)

# Generar la matriz de confusión sobre los resultados de prueba
cm_test = confusion_matrix(y_test, y_pred_test)
# Generar un reporte detallado de clasificación con nombres de clases customizados
cls_report_test = classification_report(y_test, y_pred_test, target_names=['No Sepsis', 'Sepsis'])

# Calcular la tasa de falsos positivos y falsos verdaderos para la curva ROC de test
fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
# Calcular el área bajo la curva ROC de test usando los valores de FPR y TPR
roc_auc_test = auc(fpr_test, tpr_test)

# Imprimir la exactitud (accuracy) obtenida en el test set
print(f"   ✓ Accuracy: {acc_test:.4f}")
# Imprimir el puntaje F1 obtenido en el test set
print(f"   ✓ F1-Score: {f1_test:.4f}")
# Imprimir el ROC-AUC obtenido en el test set
print(f"   ✓ ROC-AUC: {roc_auc_test:.4f}")
# Imprimir la precisión obtenida en el test set
print(f"   ✓ Precision: {prec_test:.4f}")
# Imprimir el recall obtenido en el test set
print(f"   ✓ Recall: {rec_test:.4f}")

# Línea de separación
# ==========================================
# Título para la sección 6: análisis de importancia de características
# 6. FEATURE IMPORTANCE ANALYSIS
# Línea de separación
# ==========================================
# Imprimir el progreso 5 de 6 correspondiente a la importancia de variables
print("\n[5/6] Analizando importancia de características...")

# Comentario sobre la extracción de importancias
# Obtener importancia de características
# Crear un DataFrame con las características y su importancia derivada del modelo final entrenado
feature_importance = pd.DataFrame({
# Columna de nombres de características correspondientes a columnas de X_train
    'Feature': X_train.columns,
# Columna con los valores crudos de importancia del modelo XGBoost
    'Importance': model_final.feature_importances_,
# Columna con la importancia expresada como porcentaje normalizado del total
    'Importance_Percent': (model_final.feature_importances_ / model_final.feature_importances_.sum()) * 100
# Ordenar el DataFrame creado descendentemente basado en el valor de importancia
}).sort_values('Importance', ascending=False)

# Imprimir cabecera de reporte de top características
print(f"\n   📊 TOP 15 CARACTERÍSTICAS MÁS PREDICTIVAS:")
# Imprimir la cabecera tabulada de las métricas de importancia
print(f"   {'Rank':<5} {'Feature':<30} {'Importance':<12} {'Percent':<10}")
# Imprimir guiones a modo de separador
print(f"   {'-'*60}")
# Iterar sobre las 15 características más importantes e imprimir su rank, nombre, valor crudo y porcentaje
for idx, row in feature_importance.head(15).iterrows():
# Imprimir los datos tabulados para la característica en la iteración actual
    print(f"   {idx+1:<5} {row['Feature']:<30} {row['Importance']:<12.6f} {row['Importance_Percent']:<10.2f}%")

# Línea de separación
# ==========================================
# Título para la sección de escritura de reporte en un archivo de texto
# 7. EXPORTAR RESULTADOS A ARCHIVO TXT
# Línea de separación
# ==========================================
# Imprimir paso 6 de 6 correspondiente a la creación del reporte en archivo txt
print("\n[6/6] Generando reportes y visualizaciones...")

# Abrir el archivo de salida con la ruta TXT_OUTPUT en modo escritura ("w") con codificación utf-8
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
# Escribir línea separadora en el archivo TXT
    f.write("="*80 + "\n")
# Escribir título principal en el archivo TXT
    f.write("REPORTE COMPLETO: MODELO XGBOOST CON VALIDACIÓN CRUZADA\n")
# Escribir línea separadora en el archivo TXT
    f.write("="*80 + "\n\n")
    
# Escribir título de sección 1 (configuración hiperparámetros)
    f.write("1. CONFIGURACIÓN DE HIPERPARÁMETROS\n")
# Escribir línea separadora corta
    f.write("-"*80 + "\n")
# Iterar e imprimir cada hiperparámetro utilizado al archivo TXT
    for key, val in hyperparams.items():
# Escribir clave y valor del parámetro en el archivo TXT
        f.write(f"   {key}: {val}\n")
# Escribir el peso asignado para la clase positiva debido al desbalanceo
    f.write(f"\n   Scale Pos Weight (Desbalanceo): {scale_pos_weight:.4f}\n")
# Escribir la justificación del peso utilizando las cantidades exactas de positivos y negativos
    f.write(f"   Justificación: (Negativos={num_neg} / Positivos={num_pos})\n\n")
    
# Escribir título de sección 2 (resultados detallados de validación cruzada)
    f.write("2. RESULTADOS DE VALIDACIÓN CRUZADA (5-FOLD)\n")
# Escribir línea separadora
    f.write("-"*80 + "\n")
# Convertir el dataframe df_folds a texto e incluirlo en el reporte
    f.write(df_folds.to_string(index=False))
# Agregar retornos de carro adicionales para legibilidad
    f.write("\n\n")
    
# Escribir título de la sección 3 (estadísticas globales de CV)
    f.write("3. ESTADÍSTICAS AGREGADAS DE VALIDACIÓN CRUZADA\n")
# Escribir línea separadora
    f.write("-"*80 + "\n")
# Escribir la media y la desviación estándar de la exactitud (accuracy) de CV
    f.write(f"   Accuracy:   {df_folds['accuracy'].mean():.4f} ± {df_folds['accuracy'].std():.4f}\n")
# Escribir la media y la desviación estándar del puntaje F1 de CV
    f.write(f"   F1-Score:   {df_folds['f1_score'].mean():.4f} ± {df_folds['f1_score'].std():.4f}\n")
# Escribir la media y la desviación estándar del área ROC de CV
    f.write(f"   ROC-AUC:    {df_folds['roc_auc'].mean():.4f} ± {df_folds['roc_auc'].std():.4f}\n")
# Escribir la media y la desviación estándar de la precisión de CV
    f.write(f"   Precision:  {df_folds['precision'].mean():.4f} ± {df_folds['precision'].std():.4f}\n")
# Escribir la media y la desviación estándar del recall de CV
    f.write(f"   Recall:     {df_folds['recall'].mean():.4f} ± {df_folds['recall'].std():.4f}\n")
# Escribir conteos totales y media de falsos negativos de CV
    f.write(f"   Total FN:   {df_folds['fn_count'].sum()} (Media: {df_folds['fn_count'].mean():.2f} por fold)\n")
# Escribir conteos totales y media de falsos positivos de CV
    f.write(f"   Total FP:   {df_folds['fp_count'].sum()} (Media: {df_folds['fp_count'].mean():.2f} por fold)\n\n")
    
# Escribir título para la sección 4 (métricas obtenidas sobre conjunto Test final)
    f.write("4. EVALUACIÓN EN CONJUNTO DE TEST (1,250 pacientes)\n")
# Escribir línea separadora
    f.write("-"*80 + "\n")
# Escribir la exactitud en conjunto de test
    f.write(f"   Accuracy: {acc_test:.4f}\n")
# Escribir puntaje F1 en conjunto de test
    f.write(f"   F1-Score: {f1_test:.4f}\n")
# Escribir ROC-AUC en conjunto de test
    f.write(f"   ROC-AUC: {roc_auc_test:.4f}\n")
# Escribir precisión en conjunto de test
    f.write(f"   Precision: {prec_test:.4f}\n")
# Escribir recall en conjunto de test
    f.write(f"   Recall: {rec_test:.4f}\n\n")
    
# Escribir título para sección 5 (reporte extendido de clasificación del test)
    f.write("5. REPORTE DE CLASIFICACIÓN EN TEST\n")
# Escribir línea separadora
    f.write("-"*80 + "\n")
# Agregar el reporte de clasificación (precision, recall, f1 por clase) en formato de string
    f.write(cls_report_test)
# Escribir línea en blanco adicional
    f.write("\n")
    
# Escribir título de sección 6 (matriz de confusión del test set)
    f.write("6. MATRIZ DE CONFUSIÓN EN TEST\n")
# Escribir línea separadora
    f.write("-"*80 + "\n")
# Escribir el número de Verdaderos Negativos
    f.write(f"   Verdaderos Negativos (Correcto): {cm_test[0,0]}\n")
# Escribir el número de Falsos Positivos
    f.write(f"   Falsos Positivos (Alarma falsa): {cm_test[0,1]}\n")
# Escribir el número de Falsos Negativos
    f.write(f"   Falsos Negativos (Omisión): {cm_test[1,0]}\n")
# Escribir el número de Verdaderos Positivos
    f.write(f"   Verdaderos Positivos (Correcto): {cm_test[1,1]}\n\n")
    
# Escribir título de la sección 7 (Top variables predictoras)
    f.write("7. TOP 15 CARACTERÍSTICAS PREDICTIVAS\n")
# Escribir separador
    f.write("-"*80 + "\n")
# Escribir las 15 características más relevantes serializadas a string sin número de índice
    f.write(feature_importance.head(15).to_string(index=False))
# Escribir nuevas líneas en blanco
    f.write("\n\n")
    
# Escribir título para la sección 8 (Comentarios o interpretación de resultados del test y CV)
    f.write("8. INTERPRETACIÓN CLÍNICA\n")
# Escribir línea separadora
    f.write("-"*80 + "\n")
# Redactar interpretación sobre área ROC
    f.write(f"   El modelo alcanza ROC-AUC de {roc_auc_test:.4f} en el conjunto de test,\n")
# Redactar interpretación de Falsos Negativos y la cantidad real de sepsis
    f.write(f"   con una tasa de omisión (Falsos Negativos) de {cm_test[1,0]} de {y_test.sum()} casos de sepsis.\n")
# Redactar valor y significado del recall obtenido
    f.write(f"   El recall (sensibilidad) es de {rec_test:.4f}, indicando capacidad de detección\n")
# Continuar comentario de interpretación del recall en paciente
    f.write(f"   de pacientes con sepsis. La validación cruzada confirma estabilidad con\n")
# Señalar la desviación estándar en la validación cruzada como prueba de estabilidad
    f.write(f"   desviación estándar de ROC-AUC de ± {df_folds['roc_auc'].std():.4f}.\n\n")

# Imprimir en consola el path donde fue guardado el archivo txt con el resumen
print(f"✅ Reporte guardado en: {TXT_OUTPUT}")

# Línea de separación para iniciar parte gráfica
# ==========================================
# Título para la sección 8 de código (Creación de Visualizaciones)
# 8. VISUALIZACIONES
# Línea de separación
# ==========================================
# Mostrar un mensaje indicando el inicio del proceso de graficado
print("\nGenerando visualizaciones...")

# Comentario introductorio para el gráfico de barras por cada fold
# Gráfico 1: Métricas por Fold
# Crear figura de matplotlib que contenga una grilla de 2 filas y 3 columnas con un tamaño determinado
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
# Configurar título principal centralizado para toda la figura
fig.suptitle('Métricas por Fold - Validación Cruzada Estratificada', fontsize=14, fontweight='bold')

# Definir la lista de métricas que serán graficadas individualmente
metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision', 'recall']
# Configurar una lista de tuplas con las coordenadas (fila, columna) para cada métrica
positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]

# Iterar simultáneamente sobre las métricas y sus posiciones asignadas dentro de la figura
for metric, pos in zip(metrics, positions):
# Obtener el eje (subplot) correspondiente a la posición actual
    ax = axes[pos]
# Extraer todos los valores de la métrica en cuestión correspondientes a todos los folds
    values = df_folds[metric]
# Graficar un gráfico de barras para estos valores con colores predefinidos
    bars = ax.bar(range(1, 6), values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'], alpha=0.8)
# Dibujar una línea horizontal roja punteada indicando la media de los valores
    ax.axhline(y=values.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {values.mean():.4f}')
# Etiquetar el eje X como "Fold"
    ax.set_xlabel('Fold')
# Etiquetar el eje Y con el nombre de la métrica corregido en formato título
    ax.set_ylabel(metric.replace('_', ' ').title())
# Ajustar dinámicamente los límites del eje Y; si la media es > 0.95, arrancar en 0.95, sino usar [0, 1]
    ax.set_ylim([0.95, 1.0] if values.mean() > 0.95 else [0, 1])
# Mostrar la leyenda para que aparezca la etiqueta de la línea "Media"
    ax.legend()
# Activar grilla opaca en el eje Y para facilitar lectura
    ax.grid(axis='y', alpha=0.3)
# Iterar sobre las barras de este subplot para agregar etiquetas de texto
    for bar in bars:
# Obtener la altura (valor) de la barra actual
        height = bar.get_height()
# Imprimir texto con el valor de la barra, ubicándolo en el centro horizontal, ligeramente por encima
        ax.text(bar.get_x() + bar.get_width()/2., height,
# Texto de 4 decimales alineado y dimensionado
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# Comentario sobre eliminar subplots no utilizados
# Remover último subplot vacío
# Apagar (ocultar) el último eje disponible en la cuadrícula 2x3 dado que solo hay 5 métricas
axes[1, 2].axis('off')

# Ajustar el espaciado automático de matplotlib para evitar sobreposición
plt.tight_layout()
# Guardar la figura actual como imagen PNG con 300 DPI y recortar bordes vacíos
plt.savefig(os.path.join(PLOTS_DIR, "07_metricas_por_fold.png"), dpi=300, bbox_inches='tight')
# Cerrar la figura actual para liberar memoria del backend de matplotlib
plt.close()
# Imprimir éxito de la creación y grabado del primer gráfico de folds
print("   ✓ Gráfico de métricas por fold guardado")

# Comentario sobre el proceso del segundo gráfico (Matriz de confusión de test)
# Gráfico 2: Matriz de Confusión Test
# Inicializar nueva figura de matplotlib de 7x6 pulgadas
plt.figure(figsize=(7, 6))
# Dibujar mapa de calor para la matriz de confusión usando seaborn con anotaciones numéricas y sin notación científica
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
# Asignar etiquetas personalizadas para el eje X
            xticklabels=['No Sepsis', 'Sepsis'], 
# Asignar etiquetas personalizadas para el eje Y
            yticklabels=['No Sepsis', 'Sepsis'],
# Configurar propiedades para la barra lateral indicadora de color (colorbar)
            cbar_kws={'label': 'Cantidad'})
# Título para la matriz de confusión
plt.title('Matriz de Confusión - XGBoost (Test Set)', fontsize=12, fontweight='bold')
# Asignar etiqueta 'Clase Real' al eje Y
plt.ylabel('Clase Real')
# Asignar etiqueta 'Clase Predicha' al eje X
plt.xlabel('Clase Predicha')
# Ajustar el diseño para que todo el texto entre dentro de los márgenes
plt.tight_layout()
# Guardar la imagen de matriz de confusión en el disco
plt.savefig(os.path.join(PLOTS_DIR, "08_matriz_confusion_test_mejorada.png"), dpi=300, bbox_inches='tight')
# Cerrar figura actual
plt.close()
# Imprimir mensaje de éxito en generación de matriz de confusión
print("   ✓ Matriz de confusión guardada")

# Comentario inicial para generar la curva ROC comparativa
# Gráfico 3: Curva ROC (Test + CV promedio)
# Inicializar una nueva figura de proporciones 8x7 para Curvas ROC
plt.figure(figsize=(8, 7))

# Comentario sobre graficado ROC para test
# ROC en test
# Graficar la curva del conjunto Test en naranja utilizando las tasas de falsos/verdaderos positivos
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2.5, 
# Especificar nombre con el valor de AUC para la leyenda
         label=f'Test Set (AUC = {roc_auc_test:.4f})')

# Comentario introduciendo cálculo y graficado ROC para las probabilidades apiladas de CV
# ROC en CV
# Calcular tasa de falsos y verdaderos positivos consolidada para la validación cruzada
fpr_cv, tpr_cv, _ = roc_curve(all_y_true_cv, all_y_proba_cv)
# Calcular el área bajo la curva generalizada de validación cruzada
auc_cv = auc(fpr_cv, tpr_cv)
# Graficar esta curva CV en color verde, con estilo punteado, y un grosor específico
plt.plot(fpr_cv, tpr_cv, color='green', lw=2.5, linestyle='--',
# Especificar leyenda para la curva CV junto con el valor AUC consolidado
         label=f'CV Promedio (AUC = {auc_cv:.4f})')

# Comentario para línea de referencia neutra
# Línea diagonal
# Trazar línea de origen a origen (aleatoria) como benchmark de rendimiento de un modelo ingenuo (azar)
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Clasificador Aleatorio')

# Imponer un inicio exacto de 0 y fin de 1 para el eje X
plt.xlim([0.0, 1.0])
# Imponer límite de altura ligeramente por sobre 1 en eje Y para mejorar visibilidad visual de curva cerca del límite
plt.ylim([0.0, 1.05])
# Nombrar el eje X como tasa de falsos positivos
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
# Nombrar el eje Y como tasa de verdaderos positivos
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
# Configurar título descriptivo para la comparación ROC Test vs CV
plt.title('Curva ROC - XGBoost (Comparación Test vs CV)', fontsize=12, fontweight='bold')
# Mostrar leyenda abajo a la derecha con tamaño modificado
plt.legend(loc="lower right", fontsize=10)
# Mostrar grilla de fondo para facilitar la lectura de coordenadas en la curva
plt.grid(alpha=0.3)
# Ajustar formato de espacios para no perder elementos en la imagen
plt.tight_layout()
# Escribir a disco el gráfico de curvas ROC
plt.savefig(os.path.join(PLOTS_DIR, "09_curva_roc_mejorada.png"), dpi=300, bbox_inches='tight')
# Finalizar y limpiar la ventana del gráfico ROC
plt.close()
# Confirmar por consola la conclusión del renderizado ROC
print("   ✓ Curva ROC guardada")

# Comentario para la generación del plot de las mejores 15 variables predictivas
# Gráfico 4: Feature Importance Top 15
# Crear una nueva ventana para el gráfico de tamaños 10x8 pulgadas
plt.figure(figsize=(10, 8))
# Seleccionar los primeros (mejores) 15 registros de las variables predictivas del dataframe respectivo
top_15 = feature_importance.head(15)
# Generar un arreglo de colores basado en el mapeo "viridis" para pintar individualmente las barras del top 15
colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))
# Trazar barras horizontales para las variables, asignando color dinámico y el valor respectivo
bars = plt.barh(range(len(top_15)), top_15['Importance'], color=colors)
# Nombrar las marcas (ticks) del eje vertical izquierdo para que coincidan con el nombre de variables de Top 15
plt.yticks(range(len(top_15)), top_15['Feature'])
# Asignar rótulo al eje horizontal como Importancia de Ganancia
plt.xlabel('Importancia (Ganancia)', fontsize=11, fontweight='bold')
# Asignar título del gráfico de importancias
plt.title('Top 15 Características Más Predictivas - XGBoost', fontsize=12, fontweight='bold')
# Invertir el eje Y, forzando que las variables más altas queden visualmente arriba de todo
plt.gca().invert_yaxis()

# Comentario para la adición de etiquetas de datos a las barras horizontales
# Agregar valores en las barras
# Iterar obteniendo posición, e información completa sobre cada entrada del top 15
for i, (idx, row) in enumerate(top_15.iterrows()):
# Agregar una etiqueta de texto superpuesta sobre la punta derecha de cada barra con la medida de importancia y porcentaje
    plt.text(row['Importance'], i, f" {row['Importance']:.4f} ({row['Importance_Percent']:.1f}%)", 
# Configurar la posición vertical centrada y el tamaño del texto para las etiquetas de barra
             va='center', fontsize=9)

# Aplicar grilla vertical débil que ayude con las escalas horizontales
plt.grid(axis='x', alpha=0.3)
# Ajustar márgenes para asegurar que todos los textos, títulos y valores quepan
plt.tight_layout()
# Grabar la imagen en la carpeta PLOTS con el prefijo "10"
plt.savefig(os.path.join(PLOTS_DIR, "10_feature_importance_top15.png"), dpi=300, bbox_inches='tight')
# Limpiar entorno de render para este plot
plt.close()
# Comunicar por terminal la finalización correcta
print("   ✓ Feature Importance guardada")

# Comentario para iniciar la generación de la última gráfica de resumen comparando CV y Test
# Gráfico 5: Comparación Fold vs Test
# Crear grilla horizontal de 1 fila por 2 columnas con proporciones alargadas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Comentario sobre cómo proceder a extraer y graficar información para los Folds en primer subplot
# CV Metrics
# Seleccionar un subconjunto principal de tres métricas de CV
cv_metrics = ['accuracy', 'f1_score', 'roc_auc']
# Extraer sus medias
cv_means = [df_folds[m].mean() for m in cv_metrics]
# Extraer sus desviaciones estándar
cv_stds = [df_folds[m].std() for m in cv_metrics]

# Generar un gráfico de barras para estos valores mostrando una barra por métrica con línea de desviación
axes[0].bar(range(len(cv_metrics)), cv_means, yerr=cv_stds, capsize=5, 
# Colores diferenciadores (azul, verde, naranja) con una semi-transparencia
            color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
# Señalar que en X habrá tantos ticks como métricas
axes[0].set_xticks(range(len(cv_metrics)))
# Cambiar las etiquetas por nombres sin guión bajo y con formato Title
axes[0].set_xticklabels([m.replace('_', '\n').title() for m in cv_metrics])
# Identificar el eje Y del subplot como "Valor"
axes[0].set_ylabel('Valor', fontweight='bold')
# Título exclusivo para esta zona indicando "Métricas de Validación Cruzada"
axes[0].set_title('Métricas de Validación Cruzada (Media ± Std)', fontweight='bold')
# Establecer el piso del eje y en .95 para acercar mejor visualmente diferencias si todas están sobre 95%
axes[0].set_ylim([0.95, 1.0])
# Disponer grilla horizontal
axes[0].grid(axis='y', alpha=0.3)

# Realizar anotación numérica individual en barra agregando texto
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
# Colocar texto explícito con media encima del error (std + cap), centrada
    axes[0].text(i, mean + std + 0.002, f'{mean:.4f}', ha='center', fontweight='bold')

# Comentario sobre la sección equivalente para métricas de Test en el segundo subplot
# Test Metrics
# Seleccionar el mismo subconjunto de tres métricas a evaluar, pero en el test final
test_metrics = ['accuracy', 'f1_score', 'roc_auc']
# Obtener los valores individuales de variables creadas previamente para métricas test
test_values = [acc_test, f1_test, roc_auc_test]

# Construir barras con las mismas características de color que el subplot izquierdo, para su equivalente métrica
axes[1].bar(range(len(test_metrics)), test_values, 
# Mantener el mismo juego de tres colores y la opacidad
            color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
# Señalar tres posiciones correspondientes en X
axes[1].set_xticks(range(len(test_metrics)))
# Aplicar nombres de variables formateadas y amigables
axes[1].set_xticklabels([m.replace('_', '\n').title() for m in test_metrics])
# Etiquetar el eje Y de este segundo subplot de la derecha
axes[1].set_ylabel('Valor', fontweight='bold')
# Añadir el título para el panel de rendimiento en Test final
axes[1].set_title('Métricas en Test Set', fontweight='bold')
# Forzar visualmente la misma área que en el primer subplot con rango mínimo .95 y máx 1
axes[1].set_ylim([0.95, 1.0])
# Activar grillas horizontales discretas
axes[1].grid(axis='y', alpha=0.3)

# Repartir etiquetas de texto flotando sobre cada barra en Test
for i, val in enumerate(test_values):
# Escribir valores encima de cada respectiva barra
    axes[1].text(i, val + 0.003, f'{val:.4f}', ha='center', fontweight='bold')

# Configurar layout general de forma compactada y ajustada en el canvas global de 14x5
plt.tight_layout()
# Guardar plot comparativo dual bajo el prefijo "11"
plt.savefig(os.path.join(PLOTS_DIR, "11_comparacion_cv_vs_test.png"), dpi=300, bbox_inches='tight')
# Fin de la recolección visual de elementos de PyPlot
plt.close()
# Confirmación impresa de terminación de proceso gráfico de comparativa
print("   ✓ Gráfico comparativo CV vs Test guardado")

# División visual
# ==========================================
# Zona de fin de programa
# RESUMEN FINAL
# División visual
# ==========================================
# Impresión terminal decorativa en consola
print("\n" + "="*70)
# Mensaje visual principal de éxito al término del script
print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
# Línea divisora
print("="*70)
# Título para resumen en la salida de comandos estándar
print(f"\n📊 RESUMEN DE RESULTADOS:\n")
# Categoría indicativa sobre resultados por pliegues (validación interna)
print(f"   Validación Cruzada (5-Fold):")
# Impresión del resultado estadístico de área ROC para validación interna
print(f"   • ROC-AUC: {df_folds['roc_auc'].mean():.4f} ± {df_folds['roc_auc'].std():.4f}")
# Impresión resultado estadístico Accuracy para validación interna
print(f"   • Accuracy: {df_folds['accuracy'].mean():.4f} ± {df_folds['accuracy'].std():.4f}")
# Conteo absoluto general de fallos en el reconocimiento de sepsis durante CV
print(f"   • Falsos Negativos Totales: {df_folds['fn_count'].sum()}")
# Categoría indicativa sobre la sección de pruebas para el conjunto hold-out final de pacientes
print(f"\n   Test Set (1,250 muestras):")
# Área de desempeño y robustez ante el conjunto de prueba aislado
print(f"   • ROC-AUC: {roc_auc_test:.4f}")
# Exactitud lograda en conjunto reservado para medición
print(f"   • Accuracy: {acc_test:.4f}")
# Cantidad contada de Falsos Negativos incurridos al probar sobre las 1250 muestras
print(f"   • Falsos Negativos: {cm_test[1,0]}")
# Separador anunciando top variables al final
print(f"\n   Características Principales:")
# Extraer las tres características predominantes en general
top_3_features = feature_importance.head(3)
# Bucle enumerado sobre dichas características
for i, (idx, row) in enumerate(top_3_features.iterrows(), 1):
# Escribir la característica junto con el peso devuelto por su medida de impacto XGB
    print(f"   {i}. {row['Feature']}: {row['Importance']:.6f} ({row['Importance_Percent']:.2f}%)")

# Aviso confirmando salida de todos los elementos producidos a la consola local
print(f"\n📁 Archivos generados:")
# Mostrar de nuevo en pantalla cómo acceder al reporte en texto plano TXT
print(f"   • Reporte TXT: {TXT_OUTPUT}")
# Referencia de dónde buscar y encontrar los sub-elementos de los gráficos arrojados en formato PNG
print(f"   • Gráficos: {PLOTS_DIR}/07_* a 11_*")
# Fin y despedida formal del log estándar
print(f"\n{'='*70}\n")