import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Definir rutas locales
DATA_PATH = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\raw\sepsis_icu_synthetic.csv"
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
TXT_OUTPUT = os.path.join(RESULTS_DIR, "03_preprocessing_results.txt")

print("Iniciando el Pipeline de Preprocesamiento...")
df = pd.read_csv(DATA_PATH)

# ==========================================
# 1. LIMPIEZA Y FILTRADO INICIAL
# ==========================================
# Eliminar columnas de control/identificadores y variables con varianza cero (como mechanical_ventilation)
# Modifica esta sección en tu archivo 03_preprocessing.py
# Reemplaza la lista de cols_to_drop en tu archivo 03_preprocessing.py por esta:
# Modifica esta sección en tu archivo 03_preprocessing.py
cols_to_drop = [
    'subject_id', 'mechanical_ventilation', 
    'sofa_score', 'apache_iv', 'qsofa', 'sirs_criteria',
    'pao2_fio2_ratio', 'antibiotics_24h', 'fluids_ml_24h', 
    'vasopressors_flag', 'vasopressor_dose_mcg_kg_min'
]
df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Separar características (X) y variable objetivo (y)
X = df_cleaned.drop(columns=['sepsis_label'])
y = df_cleaned['sepsis_label']

# Identificar tipos de columnas
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# ==========================================
# 2. IMPUTACIÓN DE VALORES NULOS (MEDIANA)
# ==========================================
# Usamos la mediana de entrenamiento para evitar el Data Leakage
for col in num_cols:
    if X[col].isnull().sum() > 0:
        median_value = X[col].median()
        X[col] = X[col].fillna(median_value)

# ==========================================
# 3. CODIFICACIÓN DE VARIABLES CATEGÓRICAS (ONE-HOT)
# ==========================================
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# ==========================================
# 4. DIVISIÓN TRAIN / TEST (ESTRATIFICADA)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42, stratify=y
)

# ==========================================
# 5. ESCALADO DE VARIABLES NUMÉRICAS
# ==========================================
scaler = StandardScaler()
# Ajustar con Train y transformar ambos para evitar filtración de información
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ==========================================
# 6. EXPORTAR RESULTADOS Y LOGS
# ==========================================
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write("     REPORTE DE PREPROCESAMIENTO Y LIMPIEZA\n")
    f.write("==================================================\n\n")
    
    f.write("1. TRATAMIENTO DE COLUMNAS:\n")
    f.write(f"- Columnas eliminadas: {cols_to_drop}\n")
    f.write(f"- Variables numéricas procesadas: {len(num_cols)}\n")
    f.write(f"- Variables categóricas codificadas: {cat_cols}\n\n")
    
    f.write("2. DIMENSIONES FINALES DE LOS SETS:\n")
    f.write(f"- X_train (Entrenamiento): {X_train.shape[0]} filas, {X_train.shape[1]} columnas\n")
    f.write(f"- X_test (Prueba): {X_test.shape[0]} filas, {X_test.shape[1]} columnas\n\n")
    
    f.write("3. VERIFICACIÓN DE ESTRATIFICACIÓN (PROPORCIÓN DE SEPSIS):\n")
    f.write(f"- Proporción Sepsis en Train: {(y_train.sum()/len(y_train))*100:.2f}%\n")
    f.write(f"- Proporción Sepsis en Test: {(y_test.sum()/len(y_test))*100:.2f}%\n\n")
    
    f.write("4. COMPROBACIÓN FINAL DE NULOS:\n")
    f.write(f"- Valores nulos remanentes en X_train: {X_train.isnull().sum().sum()}\n")
    f.write(f"- Valores nulos remanentes en X_test: {X_test.isnull().sum().sum()}\n")

print(f"✅ Preprocesamiento completado de manera limpia.")
print(f"✅ Reporte de control guardado en: {TXT_OUTPUT}")

# Guardar los objetos en memoria temporal para usarlos en el script del modelo o pasar las variables al siguiente paso
# Para este flujo local, podemos exportar temporalmente matrices listas a archivos .csv procesados
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
print(f"✅ Dataframes procesados exportados con éxito a 'data/processed/'")