import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ==========================================
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"

# Crear directorio si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TXT_OUTPUT = os.path.join(RESULTS_DIR, "04_model_main_MEJORADO.txt")

print("="*70)
print("MODELO PRINCIPAL MEJORADO CON VALIDACIÓN CRUZADA")
print("="*70)

# ==========================================
# 2. CARGAR DATOS PREPROCESADOS
# ==========================================
print("\n[1/6] Cargando datos preprocesados...")
try:
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()
    print(f"   ✓ X_train: {X_train.shape}")
    print(f"   ✓ X_test: {X_test.shape}")
    print(f"   ✓ y_train: {y_train.shape} (Sepsis: {y_train.sum()}, No Sepsis: {(1-y_train).sum()})")
    print(f"   ✓ y_test: {y_test.shape} (Sepsis: {y_test.sum()}, No Sepsis: {(1-y_test).sum()})")
except Exception as e:
    print(f"   ✗ Error cargando datos: {e}")
    exit()

# ==========================================
# 3. CONFIGURACIÓN DEL MODELO Y PARÁMETROS
# ==========================================
print("\n[2/6] Configurando modelo XGBoost...")

# Calcular factor de desbalanceo
num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_pos_weight = num_neg / num_pos

print(f"   ✓ Ratio negativo/positivo: {scale_pos_weight:.4f}")
print(f"   ✓ Scale_pos_weight configurado: {scale_pos_weight:.4f}")

# Hiperparámetros base
hyperparams = {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.05,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0
}

print(f"   Hiperparámetros:")
for key, val in hyperparams.items():
    print(f"     - {key}: {val}")

# ==========================================
# 4. VALIDACIÓN CRUZADA ESTRATIFICADA (K-FOLD)
# ==========================================
print("\n[3/6] Ejecutando Validación Cruzada Estratificada (5-Fold)...")

# Definir StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Listas para almacenar métricas de cada fold
fold_results = {
    'fold': [],
    'accuracy': [],
    'f1_score': [],
    'roc_auc': [],
    'precision': [],
    'recall': [],
    'fn_count': [],
    'fp_count': []
}

# Almacenar predicciones y probabilidades de todos los folds
all_y_true_cv = []
all_y_pred_cv = []
all_y_proba_cv = []

fold_number = 1
for train_idx, val_idx in skf.split(X_train, y_train):
    # Dividir datos
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]
    
    # Entrenar modelo
    model_fold = xgb.XGBClassifier(**hyperparams)
    model_fold.fit(X_fold_train, y_fold_train)
    
    # Predicciones
    y_pred_fold = model_fold.predict(X_fold_val)
    y_proba_fold = model_fold.predict_proba(X_fold_val)[:, 1]
    
    # Calcular métricas
    acc = accuracy_score(y_fold_val, y_pred_fold)
    f1 = f1_score(y_fold_val, y_pred_fold)
    roc_auc = roc_auc_score(y_fold_val, y_proba_fold)
    
    # Precisión y recall
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_fold_val, y_pred_fold, zero_division=0)
    rec = recall_score(y_fold_val, y_pred_fold, zero_division=0)
    
    # Matriz de confusión
    cm = confusion_matrix(y_fold_val, y_pred_fold)
    fn = cm[1, 0]  # Falsos Negativos
    fp = cm[0, 1]  # Falsos Positivos
    
    # Guardar resultados
    fold_results['fold'].append(fold_number)
    fold_results['accuracy'].append(acc)
    fold_results['f1_score'].append(f1)
    fold_results['roc_auc'].append(roc_auc)
    fold_results['precision'].append(prec)
    fold_results['recall'].append(rec)
    fold_results['fn_count'].append(fn)
    fold_results['fp_count'].append(fp)
    
    # Almacenar para cálculo de curva ROC promedia
    all_y_true_cv.extend(y_fold_val)
    all_y_pred_cv.extend(y_pred_fold)
    all_y_proba_cv.extend(y_proba_fold)
    
    print(f"   Fold {fold_number}/5: Acc={acc:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}, FN={fn}, FP={fp}")
    fold_number += 1

# Crear DataFrame con resultados de folds
df_folds = pd.DataFrame(fold_results)

# Calcular estadísticas de CV
print(f"\n   📊 ESTADÍSTICAS DE VALIDACIÓN CRUZADA:")
print(f"   ┌─ Accuracy:   {df_folds['accuracy'].mean():.4f} ± {df_folds['accuracy'].std():.4f}")
print(f"   ├─ F1-Score:   {df_folds['f1_score'].mean():.4f} ± {df_folds['f1_score'].std():.4f}")
print(f"   ├─ ROC-AUC:    {df_folds['roc_auc'].mean():.4f} ± {df_folds['roc_auc'].std():.4f}")
print(f"   ├─ Precision:  {df_folds['precision'].mean():.4f} ± {df_folds['precision'].std():.4f}")
print(f"   ├─ Recall:     {df_folds['recall'].mean():.4f} ± {df_folds['recall'].std():.4f}")
print(f"   ├─ FN Total:   {df_folds['fn_count'].sum()} (Media: {df_folds['fn_count'].mean():.1f})")
print(f"   └─ FP Total:   {df_folds['fp_count'].sum()} (Media: {df_folds['fp_count'].mean():.1f})")

# ==========================================
# 5. ENTRENAR MODELO FINAL EN TODO TRAIN SET
# ==========================================
print("\n[4/6] Entrenando modelo final en conjunto de entrenamiento completo...")

model_final = xgb.XGBClassifier(**hyperparams)
model_final.fit(X_train, y_train)

# Predicciones en test
y_pred_test = model_final.predict(X_test)
y_proba_test = model_final.predict_proba(X_test)[:, 1]

# Métricas en test
acc_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
roc_auc_test = roc_auc_score(y_test, y_proba_test)

from sklearn.metrics import precision_score, recall_score
prec_test = precision_score(y_test, y_pred_test, zero_division=0)
rec_test = recall_score(y_test, y_pred_test, zero_division=0)

cm_test = confusion_matrix(y_test, y_pred_test)
cls_report_test = classification_report(y_test, y_pred_test, target_names=['No Sepsis', 'Sepsis'])

fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
roc_auc_test = auc(fpr_test, tpr_test)

print(f"   ✓ Accuracy: {acc_test:.4f}")
print(f"   ✓ F1-Score: {f1_test:.4f}")
print(f"   ✓ ROC-AUC: {roc_auc_test:.4f}")
print(f"   ✓ Precision: {prec_test:.4f}")
print(f"   ✓ Recall: {rec_test:.4f}")

# ==========================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ==========================================
print("\n[5/6] Analizando importancia de características...")

# Obtener importancia de características
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model_final.feature_importances_,
    'Importance_Percent': (model_final.feature_importances_ / model_final.feature_importances_.sum()) * 100
}).sort_values('Importance', ascending=False)

print(f"\n   📊 TOP 15 CARACTERÍSTICAS MÁS PREDICTIVAS:")
print(f"   {'Rank':<5} {'Feature':<30} {'Importance':<12} {'Percent':<10}")
print(f"   {'-'*60}")
for idx, row in feature_importance.head(15).iterrows():
    print(f"   {idx+1:<5} {row['Feature']:<30} {row['Importance']:<12.6f} {row['Importance_Percent']:<10.2f}%")

# ==========================================
# 7. EXPORTAR RESULTADOS A ARCHIVO TXT
# ==========================================
print("\n[6/6] Generando reportes y visualizaciones...")

with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("REPORTE COMPLETO: MODELO XGBOOST CON VALIDACIÓN CRUZADA\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. CONFIGURACIÓN DE HIPERPARÁMETROS\n")
    f.write("-"*80 + "\n")
    for key, val in hyperparams.items():
        f.write(f"   {key}: {val}\n")
    f.write(f"\n   Scale Pos Weight (Desbalanceo): {scale_pos_weight:.4f}\n")
    f.write(f"   Justificación: (Negativos={num_neg} / Positivos={num_pos})\n\n")
    
    f.write("2. RESULTADOS DE VALIDACIÓN CRUZADA (5-FOLD)\n")
    f.write("-"*80 + "\n")
    f.write(df_folds.to_string(index=False))
    f.write("\n\n")
    
    f.write("3. ESTADÍSTICAS AGREGADAS DE VALIDACIÓN CRUZADA\n")
    f.write("-"*80 + "\n")
    f.write(f"   Accuracy:   {df_folds['accuracy'].mean():.4f} ± {df_folds['accuracy'].std():.4f}\n")
    f.write(f"   F1-Score:   {df_folds['f1_score'].mean():.4f} ± {df_folds['f1_score'].std():.4f}\n")
    f.write(f"   ROC-AUC:    {df_folds['roc_auc'].mean():.4f} ± {df_folds['roc_auc'].std():.4f}\n")
    f.write(f"   Precision:  {df_folds['precision'].mean():.4f} ± {df_folds['precision'].std():.4f}\n")
    f.write(f"   Recall:     {df_folds['recall'].mean():.4f} ± {df_folds['recall'].std():.4f}\n")
    f.write(f"   Total FN:   {df_folds['fn_count'].sum()} (Media: {df_folds['fn_count'].mean():.2f} por fold)\n")
    f.write(f"   Total FP:   {df_folds['fp_count'].sum()} (Media: {df_folds['fp_count'].mean():.2f} por fold)\n\n")
    
    f.write("4. EVALUACIÓN EN CONJUNTO DE TEST (1,250 pacientes)\n")
    f.write("-"*80 + "\n")
    f.write(f"   Accuracy: {acc_test:.4f}\n")
    f.write(f"   F1-Score: {f1_test:.4f}\n")
    f.write(f"   ROC-AUC: {roc_auc_test:.4f}\n")
    f.write(f"   Precision: {prec_test:.4f}\n")
    f.write(f"   Recall: {rec_test:.4f}\n\n")
    
    f.write("5. REPORTE DE CLASIFICACIÓN EN TEST\n")
    f.write("-"*80 + "\n")
    f.write(cls_report_test)
    f.write("\n")
    
    f.write("6. MATRIZ DE CONFUSIÓN EN TEST\n")
    f.write("-"*80 + "\n")
    f.write(f"   Verdaderos Negativos (Correcto): {cm_test[0,0]}\n")
    f.write(f"   Falsos Positivos (Alarma falsa): {cm_test[0,1]}\n")
    f.write(f"   Falsos Negativos (Omisión): {cm_test[1,0]}\n")
    f.write(f"   Verdaderos Positivos (Correcto): {cm_test[1,1]}\n\n")
    
    f.write("7. TOP 15 CARACTERÍSTICAS PREDICTIVAS\n")
    f.write("-"*80 + "\n")
    f.write(feature_importance.head(15).to_string(index=False))
    f.write("\n\n")
    
    f.write("8. INTERPRETACIÓN CLÍNICA\n")
    f.write("-"*80 + "\n")
    f.write(f"   El modelo alcanza ROC-AUC de {roc_auc_test:.4f} en el conjunto de test,\n")
    f.write(f"   con una tasa de omisión (Falsos Negativos) de {cm_test[1,0]} de {y_test.sum()} casos de sepsis.\n")
    f.write(f"   El recall (sensibilidad) es de {rec_test:.4f}, indicando capacidad de detección\n")
    f.write(f"   de pacientes con sepsis. La validación cruzada confirma estabilidad con\n")
    f.write(f"   desviación estándar de ROC-AUC de ± {df_folds['roc_auc'].std():.4f}.\n\n")

print(f"✅ Reporte guardado en: {TXT_OUTPUT}")

# ==========================================
# 8. VISUALIZACIONES
# ==========================================
print("\nGenerando visualizaciones...")

# Gráfico 1: Métricas por Fold
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Métricas por Fold - Validación Cruzada Estratificada', fontsize=14, fontweight='bold')

metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision', 'recall']
positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]

for metric, pos in zip(metrics, positions):
    ax = axes[pos]
    values = df_folds[metric]
    bars = ax.bar(range(1, 6), values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'], alpha=0.8)
    ax.axhline(y=values.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {values.mean():.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_ylim([0.95, 1.0] if values.mean() > 0.95 else [0, 1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# Remover último subplot vacío
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_metricas_por_fold.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Gráfico de métricas por fold guardado")

# Gráfico 2: Matriz de Confusión Test
plt.figure(figsize=(7, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Sepsis', 'Sepsis'], 
            yticklabels=['No Sepsis', 'Sepsis'],
            cbar_kws={'label': 'Cantidad'})
plt.title('Matriz de Confusión - XGBoost (Test Set)', fontsize=12, fontweight='bold')
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_matriz_confusion_test_mejorada.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Matriz de confusión guardada")

# Gráfico 3: Curva ROC (Test + CV promedio)
plt.figure(figsize=(8, 7))

# ROC en test
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2.5, 
         label=f'Test Set (AUC = {roc_auc_test:.4f})')

# ROC en CV
fpr_cv, tpr_cv, _ = roc_curve(all_y_true_cv, all_y_proba_cv)
auc_cv = auc(fpr_cv, tpr_cv)
plt.plot(fpr_cv, tpr_cv, color='green', lw=2.5, linestyle='--',
         label=f'CV Promedio (AUC = {auc_cv:.4f})')

# Línea diagonal
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Clasificador Aleatorio')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
plt.title('Curva ROC - XGBoost (Comparación Test vs CV)', fontsize=12, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "09_curva_roc_mejorada.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Curva ROC guardada")

# Gráfico 4: Feature Importance Top 15
plt.figure(figsize=(10, 8))
top_15 = feature_importance.head(15)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))
bars = plt.barh(range(len(top_15)), top_15['Importance'], color=colors)
plt.yticks(range(len(top_15)), top_15['Feature'])
plt.xlabel('Importancia (Ganancia)', fontsize=11, fontweight='bold')
plt.title('Top 15 Características Más Predictivas - XGBoost', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()

# Agregar valores en las barras
for i, (idx, row) in enumerate(top_15.iterrows()):
    plt.text(row['Importance'], i, f" {row['Importance']:.4f} ({row['Importance_Percent']:.1f}%)", 
             va='center', fontsize=9)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "10_feature_importance_top15.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Feature Importance guardada")

# Gráfico 5: Comparación Fold vs Test
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CV Metrics
cv_metrics = ['accuracy', 'f1_score', 'roc_auc']
cv_means = [df_folds[m].mean() for m in cv_metrics]
cv_stds = [df_folds[m].std() for m in cv_metrics]

axes[0].bar(range(len(cv_metrics)), cv_means, yerr=cv_stds, capsize=5, 
            color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
axes[0].set_xticks(range(len(cv_metrics)))
axes[0].set_xticklabels([m.replace('_', '\n').title() for m in cv_metrics])
axes[0].set_ylabel('Valor', fontweight='bold')
axes[0].set_title('Métricas de Validación Cruzada (Media ± Std)', fontweight='bold')
axes[0].set_ylim([0.95, 1.0])
axes[0].grid(axis='y', alpha=0.3)

for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    axes[0].text(i, mean + std + 0.002, f'{mean:.4f}', ha='center', fontweight='bold')

# Test Metrics
test_metrics = ['accuracy', 'f1_score', 'roc_auc']
test_values = [acc_test, f1_test, roc_auc_test]

axes[1].bar(range(len(test_metrics)), test_values, 
            color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
axes[1].set_xticks(range(len(test_metrics)))
axes[1].set_xticklabels([m.replace('_', '\n').title() for m in test_metrics])
axes[1].set_ylabel('Valor', fontweight='bold')
axes[1].set_title('Métricas en Test Set', fontweight='bold')
axes[1].set_ylim([0.95, 1.0])
axes[1].grid(axis='y', alpha=0.3)

for i, val in enumerate(test_values):
    axes[1].text(i, val + 0.003, f'{val:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_comparacion_cv_vs_test.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Gráfico comparativo CV vs Test guardado")

# ==========================================
# RESUMEN FINAL
# ==========================================
print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*70)
print(f"\n📊 RESUMEN DE RESULTADOS:\n")
print(f"   Validación Cruzada (5-Fold):")
print(f"   • ROC-AUC: {df_folds['roc_auc'].mean():.4f} ± {df_folds['roc_auc'].std():.4f}")
print(f"   • Accuracy: {df_folds['accuracy'].mean():.4f} ± {df_folds['accuracy'].std():.4f}")
print(f"   • Falsos Negativos Totales: {df_folds['fn_count'].sum()}")
print(f"\n   Test Set (1,250 muestras):")
print(f"   • ROC-AUC: {roc_auc_test:.4f}")
print(f"   • Accuracy: {acc_test:.4f}")
print(f"   • Falsos Negativos: {cm_test[1,0]}")
print(f"\n   Características Principales:")
top_3_features = feature_importance.head(3)
for i, (idx, row) in enumerate(top_3_features.iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Importance']:.6f} ({row['Importance_Percent']:.2f}%)")

print(f"\n📁 Archivos generados:")
print(f"   • Reporte TXT: {TXT_OUTPUT}")
print(f"   • Gráficos: {PLOTS_DIR}/07_* a 11_*")
print(f"\n{'='*70}\n")