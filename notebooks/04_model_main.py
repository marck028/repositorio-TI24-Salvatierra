import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb

# 1. Definir rutas locales
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"
TXT_OUTPUT = os.path.join(RESULTS_DIR, "04_model_main_results.txt")

print("Cargando datos preprocesados...")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

# ==========================================
# 2. CONFIGURACIÓN Y ENTRENAMIENTO DE XGBOOST
# ==========================================
print("Entrenando el modelo principal (XGBoost)...")

# Calcular factor de desbalanceo (Clase negativa / Clase positiva)
num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_weight = num_neg / num_pos

# Definir hiperparámetros clave solicitados por la guía
model = xgb.XGBClassifier(
    n_estimators=150,          # Número de árboles de decisión a construir
    max_depth=5,               # Profundidad máxima de cada árbol (evita sobreajuste)
    learning_rate=0.05,        # Tasa de aprendizaje (shrinkage)
    scale_pos_weight=scale_weight, # Ponderación matemática para el desbalanceo de clases
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ==========================================
# 3. EVALUACIÓN DE RENDIMIENTO
# ==========================================
print("Evaluando el modelo...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cls_report = classification_report(y_test, y_pred, target_names=['No Sepsis', 'Sepsis'])

# Calcular Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# ==========================================
# 4. EXPORTAR REPORTE DE RESULTADOS (.TXT)
# ==========================================
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write("    REPORTE DE RENDIMIENTO: MODELO PRINCIPAL (XGBOOST)\n")
    f.write("==================================================\n\n")
    
    f.write("1. HIPERPARÁMETROS ELEGIDOS:\n")
    f.write(f"- n_estimators: 150 (Árboles secuenciales)\n")
    f.write(f"- max_depth: 5 (Control de complejidad estructural)\n")
    f.write(f"- learning_rate: 0.05 (Ajuste fino de pesos)\n")
    f.write(f"- scale_pos_weight: {scale_weight:.2f} (Factor de balanceo de clases)\n\n")
    
    f.write("2. MÉTRICAS GLOBALES DE EVALUACIÓN:\n")
    f.write(f"- Accuracy: {acc:.4f}\n")
    f.write(f"- F1-Score: {f1:.4f}\n")
    f.write(f"- ROC AUC: {roc_auc:.4f}\n\n")
    
    f.write("3. REPORTE DE CLASIFICACIÓN DETALLADO:\n")
    f.write("--------------------------------------------------\n")
    f.write(cls_report)
    f.write("\n")
    
    f.write("4. MATRIZ DE CONFUSIÓN (VALORES):\n")
    f.write(f"Verdaderos Negativos (No Sepsis correcto): {cm[0,0]}\n")
    f.write(f"Falsos Positivos (Alarma falsa): {cm[0,1]}\n")
    f.write(f"Falsos Negativos (Sepsis no detectada): {cm[1,0]}\n")
    f.write(f"Verdaderos Positivos (Sepsis detectada correctamente): {cm[1,1]}\n")

# ==========================================
# 5. GENERACIÓN DE GRÁFICOS DE RENDIMIENTO
# ==========================================
print("Exportando gráficos de evaluación...")

# Gráfico 1: Matriz de Confusión Visual
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Sepsis', 'Sepsis'], yticklabels=['No Sepsis', 'Sepsis'])
plt.title('Matriz de Confusión - XGBoost', fontsize=12, fontweight='bold')
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_matriz_confusion_xgb.png"), dpi=300)
plt.close()

# Gráfico 2: Curva ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Predicción de Sepsis', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_curva_roc_xgb.png"), dpi=300)
plt.close()

print(f"✅ Entrenamiento de XGBoost completado con éxito.")
print(f"✅ Métricas guardadas en: {TXT_OUTPUT}")
print(f"✅ Gráficos guardados en la carpeta 'plots/'")