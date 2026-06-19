# Importar el módulo os para trabajar con rutas y el sistema operativo
import os
# Importar la librería pandas con el alias pd para el manejo de datos
import pandas as pd
# Importar la librería xgboost con el alias xgb para el modelado predictivo
import xgboost as xgb

# Definir la ruta del directorio que contiene los datos procesados
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
# Cargar el archivo CSV de las características de entrenamiento en un DataFrame de pandas
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
# Cargar la variable objetivo de entrenamiento y aplanarla a un array unidimensional
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()

# Entrenar un modelo rápido para extraer importancia de características
# Inicializar un clasificador de XGBoost con 10 estimadores, profundidad máxima de 3 y una semilla fija
model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
# Entrenar el modelo utilizando las características y la variable objetivo
model.fit(X_train, y_train)

# Crear un DataFrame con las importancias
# Construir un nuevo DataFrame que relacione cada variable con su nivel de importancia
importancias = pd.DataFrame({
    # Asignar los nombres de las columnas del conjunto de entrenamiento
    'Variable': X_train.columns,
    # Extraer las importancias de las características desde el modelo entrenado
    'Importancia': model.feature_importances_
# Ordenar el DataFrame por la columna de importancia de forma descendente (de mayor a menor)
}).sort_values(by='Importancia', ascending=False)

# Mostrar un mensaje indicando que se imprimirán las variables más importantes
print("Variables con mayor poder predictivo (posibles filtraciones):")
# Mostrar las 10 primeras filas del DataFrame ordenado de importancias
print(importancias.head(10))