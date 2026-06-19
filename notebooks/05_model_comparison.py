# Importar el módulo os para interactuar con el sistema operativo
import os
# Importar la biblioteca pandas con el alias pd para manipulación y análisis de datos
import pandas as pd
# Importar la biblioteca numpy con el alias np para operaciones numéricas y arreglos
import numpy as np
# Importar la interfaz pyplot de matplotlib con el alias plt para crear gráficos
import matplotlib.pyplot as plt
# Importar varias métricas de evaluación desde el módulo metrics de scikit-learn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
# Importar la biblioteca xgboost con el alias xgb para usar el algoritmo XGBoost
import xgboost as xgb
# Importar la biblioteca lightgbm con el alias lgb para usar el algoritmo LightGBM
import lightgbm as lgb

# 1. Definir rutas locales
# Definir la ruta del directorio donde se encuentran los datos procesados
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
# Definir la ruta del directorio donde se guardarán los resultados
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
# Definir la ruta del directorio donde se guardarán los gráficos
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"
# Unir el directorio de resultados con el nombre del archivo para obtener la ruta de salida del texto
TXT_OUTPUT = os.path.join(RESULTS_DIR, "05_model_comparison_results.txt")

# Imprimir un mensaje indicando que se están cargando los datos
print("Cargando datos preprocesados para la comparación...")
# Leer el archivo CSV con las características de entrenamiento usando pandas
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
# Leer el archivo CSV con las características de prueba usando pandas
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
# Leer las etiquetas de entrenamiento, obtener sus valores y aplanarlos a 1D
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
# Leer las etiquetas de prueba, obtener sus valores y aplanarlos a 1D
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

# Calcular factor de desbalanceo
# Calcular la proporción entre la clase mayoritaria (0) y la minoritaria (1) para balancear
scale_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# ==========================================
# 2. ENTRENAR MODELO 1: XGBOOST (CORREGIDO)
# ==========================================
# Imprimir un mensaje indicando que XGBoost se está entrenando
print("Entrenando XGBoost...")
# Inicializar el clasificador XGBoost con hiperparámetros específicos
xgb_model = xgb.XGBClassifier(
    # Establecer 150 árboles, profundidad máxima de 5, y tasa de aprendizaje de 0.05
    n_estimators=150, max_depth=5, learning_rate=0.05,
    # Balancear el peso de las clases positivas, semilla aleatoria 42, y métrica logloss
    scale_pos_weight=scale_weight, random_state=42, eval_metric='logloss'
# Cerrar el paréntesis de la inicialización del clasificador
)
# Ajustar el modelo XGBoost a los datos de entrenamiento
xgb_model.fit(X_train, y_train)
# Predecir las probabilidades de la clase positiva (1) en los datos de prueba
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
# Predecir las etiquetas de clase en los datos de prueba
xgb_pred = xgb_model.predict(X_test)

# ==========================================
# 3. ENTRENAR MODELO 2: LIGHTGBM (COMPARATIVO)
# ==========================================
# Imprimir un mensaje indicando que LightGBM se está entrenando
print("Entrenando LightGBM...")
# Inicializar el clasificador LightGBM con hiperparámetros específicos
lgb_model = lgb.LGBMClassifier(
    # Establecer 150 árboles, profundidad máxima de 5, y tasa de aprendizaje de 0.05
    n_estimators=150, max_depth=5, learning_rate=0.05,
    # Balancear el peso de clases, semilla 42 y nivel de verbosidad -1 (silencioso)
    scale_pos_weight=scale_weight, random_state=42, verbosity=-1
# Cerrar el paréntesis de la inicialización del clasificador LightGBM
)
# Ajustar el modelo LightGBM a los datos de entrenamiento
lgb_model.fit(X_train, y_train)
# Predecir las probabilidades de la clase positiva (1) en los datos de prueba
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
# Predecir las etiquetas de clase en los datos de prueba
lgb_pred = lgb_model.predict(X_test)

# ==========================================
# 4. EVALUACIÓN Y COMPARACIÓN DE MÉTRICAS
# ==========================================
# Imprimir un mensaje indicando que se está generando el reporte
print("Generando reporte comparativo...")

# Métricas XGBoost
# Calcular la precisión (accuracy) del modelo XGBoost
xgb_acc = accuracy_score(y_test, xgb_pred)
# Calcular la puntuación F1 del modelo XGBoost
xgb_f1 = f1_score(y_test, xgb_pred)
# Calcular la curva ROC (tasa de falsos y verdaderos positivos) para XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
# Calcular el Área Bajo la Curva (AUC) para XGBoost
auc_xgb = auc(fpr_xgb, tpr_xgb)

# Métricas LightGBM
# Calcular la precisión (accuracy) del modelo LightGBM
lgb_acc = accuracy_score(y_test, lgb_pred)
# Calcular la puntuación F1 del modelo LightGBM
lgb_f1 = f1_score(y_test, lgb_pred)
# Calcular la curva ROC (tasa de falsos y verdaderos positivos) para LightGBM
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_proba)
# Calcular el Área Bajo la Curva (AUC) para LightGBM
auc_lgb = auc(fpr_lgb, tpr_lgb)

# Guardar reporte en TXT
# Abrir el archivo de texto en modo escritura ("w") con codificación utf-8
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    # Escribir una línea separadora superior en el archivo
    f.write("==================================================\n")
    # Escribir el título del reporte en el archivo
    f.write("       REPORTE COMPARATIVO DE MODELOS DE BOOSTING\n")
    # Escribir una línea separadora inferior y un salto de línea en el archivo
    f.write("==================================================\n\n")
    
    # Escribir el subtítulo para la tabla de métricas
    f.write("TABLA COMPARATIVA DE MÉTRICAS:\n")
    # Escribir una línea separadora debajo del subtítulo
    f.write("--------------------------------------------------\n")
    # Escribir los encabezados de las columnas para la tabla alineados
    f.write(f"{'Métrica':<15}{'XGBoost':<15}{'LightGBM':<15}\n")
    # Escribir una línea de guiones como separador de encabezados
    f.write(f"{'-'*45}\n")
    # Escribir la fila correspondiente a la métrica de Accuracy formateada
    f.write(f"{'Accuracy':<15}{f'{xgb_acc:.4f}':<15}{f'{lgb_acc:.4f}':<15}\n")
    # Escribir la fila correspondiente a la métrica F1-Score formateada
    f.write(f"{'F1-Score':<15}{f'{xgb_f1:.4f}':<15}{f'{lgb_f1:.4f}':<15}\n")
    # Escribir la fila correspondiente a la métrica ROC AUC formateada
    f.write(f"{'ROC AUC':<15}{f'{auc_xgb:.4f}':<15}{f'{auc_lgb:.4f}':<15}\n\n")
    
    # Escribir el subtítulo para el análisis de confusión de LightGBM
    f.write("ANÁLISIS DE CONFUSIÓN DE LIGHTGBM:\n")
    # Escribir una línea separadora para la sección
    f.write("--------------------------------------------------\n")
    # Calcular la matriz de confusión para LightGBM
    cm_lgb = confusion_matrix(y_test, lgb_pred)
    # Escribir el número de Verdaderos Negativos extraído de la matriz
    f.write(f"- Verdaderos Negativos: {cm_lgb[0,0]}\n")
    # Escribir el número de Falsos Positivos extraído de la matriz
    f.write(f"- Falsos Positivos: {cm_lgb[0,1]}\n")
    # Escribir el número de Falsos Negativos extraído de la matriz
    f.write(f"- Falsos Negativos: {cm_lgb[1,0]}\n")
    # Escribir el número de Verdaderos Positivos extraído de la matriz
    f.write(f"- Verdaderos Positivos: {cm_lgb[1,1]}\n")

# ==========================================
# 5. GRÁFICO: CURVAS ROC SUPERPUESTAS
# ==========================================
# Imprimir un mensaje indicando que se está exportando el gráfico
print("Exportando gráfico comparativo...")
# Crear una nueva figura para el gráfico con tamaño de 7x6 pulgadas
plt.figure(figsize=(7, 6))
# Dibujar la curva ROC de XGBoost en color naranja oscuro
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'XGBoost (AUC = {auc_xgb:.4f})')
# Dibujar la curva ROC de LightGBM en color verde
plt.plot(fpr_lgb, tpr_lgb, color='green', lw=2, label=f'LightGBM (AUC = {auc_lgb:.4f})')
# Dibujar la línea diagonal punteada que representa un clasificador aleatorio
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# Limitar el eje X de 0.0 a 1.0
plt.xlim([0.0, 1.0])
# Limitar el eje Y de 0.0 a 1.05 para dar un poco de espacio en la parte superior
plt.ylim([0.0, 1.05])
# Establecer la etiqueta del eje X
plt.xlabel('Tasa de Falsos Positivos (FPR)')
# Establecer la etiqueta del eje Y
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
# Establecer el título del gráfico
plt.title('Comparación de Curvas ROC (XGBoost vs LightGBM)', fontsize=12, fontweight='bold')
# Mostrar la leyenda del gráfico en la esquina inferior derecha
plt.legend(loc="lower right")
# Ajustar automáticamente los márgenes del gráfico
plt.tight_layout()
# Guardar el gráfico generado en la ruta de gráficos en formato PNG con 300 dpi
plt.savefig(os.path.join(PLOTS_DIR, "06_comparacion_curvas_roc.png"), dpi=300)
# Cerrar la figura actual para liberar memoria
plt.close()

# Imprimir un mensaje de éxito con la ruta del reporte en TXT
print(f"✅ Reporte comparativo guardado en: {TXT_OUTPUT}")
# Imprimir un mensaje de éxito indicando dónde se guardó la imagen ROC
print(f"✅ Curva ROC comparativa exportada en 'plots/06_comparacion_curvas_roc.png'")