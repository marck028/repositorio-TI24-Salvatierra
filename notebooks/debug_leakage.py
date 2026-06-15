import os
import pandas as pd
import xgboost as xgb

PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()

# Entrenar un modelo rápido para extraer importancia de características
model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Crear un DataFrame con las importancias
importancias = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': model.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("Variables con mayor poder predictivo (posibles filtraciones):")
print(importancias.head(10))