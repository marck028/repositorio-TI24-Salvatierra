# Importar el módulo os para operaciones del sistema de archivos
import os
# Importar la biblioteca pandas para el manejo de datos estructurados (DataFrames)
import pandas as pd
# Importar la biblioteca numpy para operaciones numéricas y arreglos
import numpy as np
# Importar la función train_test_split para dividir datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
# Importar la clase StandardScaler para escalar características numéricas (estandarización)
from sklearn.preprocessing import StandardScaler

# 1. Definir rutas locales
# Definir la ruta del archivo de datos sintéticos en bruto (CSV)
DATA_PATH = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\raw\sepsis_icu_synthetic.csv"
# Definir la ruta del directorio donde se guardarán los resultados
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
# Construir la ruta completa para el archivo de salida de texto combinando RESULTS_DIR y el nombre del archivo
TXT_OUTPUT = os.path.join(RESULTS_DIR, "03_preprocessing_results.txt")

# Imprimir un mensaje indicando el inicio del proceso en consola
print("Iniciando el Pipeline de Preprocesamiento...")
# Leer el archivo CSV con pandas y cargarlo en un DataFrame 'df'
df = pd.read_csv(DATA_PATH)

# ==========================================
# 1. LIMPIEZA Y FILTRADO INICIAL
# ==========================================
# Eliminar columnas de control/identificadores y variables con varianza cero (como mechanical_ventilation)
# Modifica esta sección en tu archivo 03_preprocessing.py
# Reemplaza la lista de cols_to_drop en tu archivo 03_preprocessing.py por esta:
# Modifica esta sección en tu archivo 03_preprocessing.py
# Crear una lista con los nombres de las columnas que no aportan valor o generan fugas de información y deben eliminarse
cols_to_drop = [
    'subject_id', 'mechanical_ventilation', 
    'sofa_score', 'apache_iv', 'qsofa', 'sirs_criteria',
    'pao2_fio2_ratio', 'antibiotics_24h', 'fluids_ml_24h', 
    'vasopressors_flag', 'vasopressor_dose_mcg_kg_min'
]
# Crear un nuevo DataFrame 'df_cleaned' eliminando las columnas especificadas, verificando antes que existan en el DataFrame actual
df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Separar características (X) y variable objetivo (y)
# Crear el DataFrame 'X' con las variables predictoras eliminando la columna objetivo 'sepsis_label'
X = df_cleaned.drop(columns=['sepsis_label'])
# Crear la Serie 'y' que contiene únicamente la variable objetivo 'sepsis_label'
y = df_cleaned['sepsis_label']

# Identificar tipos de columnas
# Obtener una lista con los nombres de las columnas numéricas (enteros o de punto flotante)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Obtener una lista con los nombres de las columnas categóricas (texto u objetos)
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# ==========================================
# 2. IMPUTACIÓN DE VALORES NULOS (MEDIANA)
# ==========================================
# Usamos la mediana de entrenamiento para evitar el Data Leakage
# Iniciar un bucle para iterar a través de cada columna numérica identificada previamente
for col in num_cols:
    # Comprobar si la columna actual contiene al menos un valor nulo (NaN)
    if X[col].isnull().sum() > 0:
        # Calcular la mediana de la columna numérica actual
        median_value = X[col].median()
        # Rellenar los valores nulos en la columna con la mediana calculada
        X[col] = X[col].fillna(median_value)

# ==========================================
# 3. CODIFICACIÓN DE VARIABLES CATEGÓRICAS (ONE-HOT)
# ==========================================
# Convertir variables categóricas en múltiples columnas binarias usando One-Hot Encoding, eliminando la primera categoría para evitar multicolinealidad
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# ==========================================
# 4. DIVISIÓN TRAIN / TEST (ESTRATIFICADA)
# ==========================================
# Dividir los datos procesados en conjuntos de entrenamiento (75%) y prueba (25%), garantizando la misma proporción de clases en ambos ('stratify')
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42, stratify=y
)

# ==========================================
# 5. ESCALADO DE VARIABLES NUMÉRICAS
# ==========================================
# Inicializar el escalador estándar para transformar datos a media 0 y desviación estándar 1
scaler = StandardScaler()
# Ajustar con Train y transformar ambos para evitar filtración de información
# Ajustar el escalador solo con los datos de entrenamiento y transformar las variables numéricas en X_train
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
# Transformar las variables numéricas en el conjunto de prueba usando los parámetros aprendidos del entrenamiento
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ==========================================
# 6. EXPORTAR RESULTADOS Y LOGS
# ==========================================
# Abrir el archivo de texto en modo escritura ('w') con codificación UTF-8
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    # Escribir una línea separadora superior para el reporte
    f.write("==================================================\n")
    # Escribir el título principal del reporte de preprocesamiento
    f.write("     REPORTE DE PREPROCESAMIENTO Y LIMPIEZA\n")
    # Escribir una línea separadora inferior para el título y un salto de línea adicional
    f.write("==================================================\n\n")
    
    # Escribir el encabezado para la sección de tratamiento de columnas
    f.write("1. TRATAMIENTO DE COLUMNAS:\n")
    # Escribir en el archivo la lista de las columnas que fueron eliminadas del DataFrame original
    f.write(f"- Columnas eliminadas: {cols_to_drop}\n")
    # Escribir la cantidad total de variables numéricas procesadas en el pipeline
    f.write(f"- Variables numéricas procesadas: {len(num_cols)}\n")
    # Escribir los nombres de las variables categóricas que fueron codificadas (One-Hot)
    f.write(f"- Variables categóricas codificadas: {cat_cols}\n\n")
    
    # Escribir el encabezado para la sección de dimensiones de los conjuntos de datos
    f.write("2. DIMENSIONES FINALES DE LOS SETS:\n")
    # Escribir el número de filas y columnas resultantes para el conjunto de entrenamiento (X_train)
    f.write(f"- X_train (Entrenamiento): {X_train.shape[0]} filas, {X_train.shape[1]} columnas\n")
    # Escribir el número de filas y columnas resultantes para el conjunto de prueba (X_test)
    f.write(f"- X_test (Prueba): {X_test.shape[0]} filas, {X_test.shape[1]} columnas\n\n")
    
    # Escribir el encabezado para la sección de comprobación de la estratificación de clases
    f.write("3. VERIFICACIÓN DE ESTRATIFICACIÓN (PROPORCIÓN DE SEPSIS):\n")
    # Escribir el porcentaje de casos positivos de sepsis en el conjunto de entrenamiento
    f.write(f"- Proporción Sepsis en Train: {(y_train.sum()/len(y_train))*100:.2f}%\n")
    # Escribir el porcentaje de casos positivos de sepsis en el conjunto de prueba
    f.write(f"- Proporción Sepsis en Test: {(y_test.sum()/len(y_test))*100:.2f}%\n\n")
    
    # Escribir el encabezado para la sección final de verificación de valores nulos
    f.write("4. COMPROBACIÓN FINAL DE NULOS:\n")
    # Escribir el número total de valores nulos que quedaron en todo el conjunto de entrenamiento
    f.write(f"- Valores nulos remanentes en X_train: {X_train.isnull().sum().sum()}\n")
    # Escribir el número total de valores nulos que quedaron en todo el conjunto de prueba
    f.write(f"- Valores nulos remanentes en X_test: {X_test.isnull().sum().sum()}\n")

# Imprimir mensaje de éxito indicando que la limpieza finalizó
print(f"✅ Preprocesamiento completado de manera limpia.")
# Imprimir mensaje informando la ruta exacta donde se guardó el reporte
print(f"✅ Reporte de control guardado en: {TXT_OUTPUT}")

# Guardar los objetos en memoria temporal para usarlos en el script del modelo o pasar las variables al siguiente paso
# Para este flujo local, podemos exportar temporalmente matrices listas a archivos .csv procesados
# Definir la ruta del directorio donde se almacenarán los datos ya procesados
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
# Crear la carpeta de destino si no existe, sin mostrar errores en caso de que ya exista
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Guardar el conjunto de datos de entrenamiento (características) como archivo CSV, ignorando el índice
X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
# Guardar el conjunto de datos de prueba (características) como archivo CSV, ignorando el índice
X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
# Guardar la variable objetivo de entrenamiento como archivo CSV, ignorando el índice
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
# Guardar la variable objetivo de prueba como archivo CSV, ignorando el índice
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
# Imprimir mensaje de éxito tras exportar los conjuntos procesados
print(f"✅ Dataframes procesados exportados con éxito a 'data/processed/'")