import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import lightgbm as lgb

# 1. Definir rutas locales
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"
TXT_OUTPUT = os.path.join(RESULTS_DIR, "05_model_comparison_results.txt")

print("Cargando datos preprocesados para la comparación...")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

# Calcular factor de desbalanceo
scale_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# ==========================================
# 2. ENTRENAR MODELO 1: XGBOOST (CORREGIDO)
# ==========================================
print("Entrenando XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_weight, random_state=42, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict(X_test)

# ==========================================
# 3. ENTRENAR MODELO 2: LIGHTGBM (COMPARATIVO)
# ==========================================
print("Entrenando LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_weight, random_state=42, verbosity=-1
)
lgb_model.fit(X_train, y_train)
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_pred = lgb_model.predict(X_test)

# ==========================================
# 4. EVALUACIÓN Y COMPARACIÓN DE MÉTRICAS
# ==========================================
print("Generando reporte comparativo...")

# Métricas XGBoost
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# Métricas LightGBM
lgb_acc = accuracy_score(y_test, lgb_pred)
lgb_f1 = f1_score(y_test, lgb_pred)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_proba)
auc_lgb = auc(fpr_lgb, tpr_lgb)

# Guardar reporte en TXT
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write("       REPORTE COMPARATIVO DE MODELOS DE BOOSTING\n")
    f.write("==================================================\n\n")
    
    f.write("TABLA COMPARATIVA DE MÉTRICAS:\n")
    f.write("--------------------------------------------------\n")
    f.write(f"{'Métrica':<15}{'XGBoost':<15}{'LightGBM':<15}\n")
    f.write(f"{'-'*45}\n")
    f.write(f"{'Accuracy':<15}{f'{xgb_acc:.4f}':<15}{f'{lgb_acc:.4f}':<15}\n")
    f.write(f"{'F1-Score':<15}{f'{xgb_f1:.4f}':<15}{f'{lgb_f1:.4f}':<15}\n")
    f.write(f"{'ROC AUC':<15}{f'{auc_xgb:.4f}':<15}{f'{auc_lgb:.4f}':<15}\n\n")
    
    f.write("ANÁLISIS DE CONFUSIÓN DE LIGHTGBM:\n")
    f.write("--------------------------------------------------\n")
    cm_lgb = confusion_matrix(y_test, lgb_pred)
    f.write(f"- Verdaderos Negativos: {cm_lgb[0,0]}\n")
    f.write(f"- Falsos Positivos: {cm_lgb[0,1]}\n")
    f.write(f"- Falsos Negativos: {cm_lgb[1,0]}\n")
    f.write(f"- Verdaderos Positivos: {cm_lgb[1,1]}\n")

# ==========================================
# 5. GRÁFICO: CURVAS ROC SUPERPUESTAS
# ==========================================
print("Exportando gráfico comparativo...")
plt.figure(figsize=(7, 6))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'XGBoost (AUC = {auc_xgb:.4f})')
plt.plot(fpr_lgb, tpr_lgb, color='green', lw=2, label=f'LightGBM (AUC = {auc_lgb:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Comparación de Curvas ROC (XGBoost vs LightGBM)', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_comparacion_curvas_roc.png"), dpi=300)
plt.close()

print(f"✅ Reporte comparativo guardado en: {TXT_OUTPUT}")
print(f"✅ Curva ROC comparativa exportada en 'plots/06_comparacion_curvas_roc.png'")