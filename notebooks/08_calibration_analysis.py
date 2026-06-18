import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
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

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TXT_OUTPUT = os.path.join(RESULTS_DIR, "08_calibration_analysis.txt")

print("="*80)
print("ANÁLISIS DE CALIBRACIÓN DEL MODELO")
print("="*80)

# ==========================================
# 2. CARGAR DATOS
# ==========================================
print("\n[1/4] Cargando datos...")

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_pos_weight = num_neg / num_pos

print(f"   ✓ Datos cargados: {X_train.shape[0]} train, {X_test.shape[0]} test")

# ==========================================
# 3. ENTRENAR MODELOS
# ==========================================
print("\n[2/4] Entrenando modelos...")

# XGBoost
model_xgb = xgb.XGBClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric='logloss', verbosity=0
)
model_xgb.fit(X_train, y_train)
y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

# LightGBM
model_lgb = lgb.LGBMClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1
)
model_lgb.fit(X_train, y_train)
y_proba_lgb = model_lgb.predict_proba(X_test)[:, 1]

print(f"   ✓ XGBoost entrenado")
print(f"   ✓ LightGBM entrenado")

# ==========================================
# 4. CALCULAR MÉTRICAS DE CALIBRACIÓN
# ==========================================
print("\n[3/4] Calculando métricas de calibración...")

# Calibración XGBoost
prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, y_proba_xgb, n_bins=10, strategy='uniform')
brier_xgb = brier_score_loss(y_test, y_proba_xgb)
logloss_xgb = log_loss(y_test, y_proba_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

# Calibración LightGBM
prob_true_lgb, prob_pred_lgb = calibration_curve(y_test, y_proba_lgb, n_bins=10, strategy='uniform')
brier_lgb = brier_score_loss(y_test, y_proba_lgb)
logloss_lgb = log_loss(y_test, y_proba_lgb)
auc_lgb = roc_auc_score(y_test, y_proba_lgb)

print(f"\n   XGBoost:")
print(f"   • Brier Score: {brier_xgb:.4f}")
print(f"   • Log Loss: {logloss_xgb:.4f}")
print(f"   • ROC-AUC: {auc_xgb:.4f}")

print(f"\n   LightGBM:")
print(f"   • Brier Score: {brier_lgb:.4f}")
print(f"   • Log Loss: {logloss_lgb:.4f}")
print(f"   • ROC-AUC: {auc_lgb:.4f}")

# ==========================================
# 5. ANÁLISIS DE CONFIANZA
# ==========================================
print("\n   Analizando distribución de confianza...")

# Categorías de confianza
confidence_bins_xgb = pd.cut(y_proba_xgb, bins=[0, 0.3, 0.5, 0.7, 1.0], 
                             labels=['Bajo (0-30%)', 'Medio (30-50%)', 'Medio-Alto (50-70%)', 'Alto (70-100%)'])
confidence_bins_lgb = pd.cut(y_proba_lgb, bins=[0, 0.3, 0.5, 0.7, 1.0],
                             labels=['Bajo (0-30%)', 'Medio (30-50%)', 'Medio-Alto (50-70%)', 'Alto (70-100%)'])

# Contar por bin
conf_xgb = confidence_bins_xgb.value_counts().sort_index()
conf_lgb = confidence_bins_lgb.value_counts().sort_index()

print(f"   XGBoost - Distribución de confianza:")
for conf, count in conf_xgb.items():
    print(f"      {conf}: {count} ({100*count/len(y_test):.1f}%)")

print(f"\n   LightGBM - Distribución de confianza:")
for conf, count in conf_lgb.items():
    print(f"      {conf}: {count} ({100*count/len(y_test):.1f}%)")

# ==========================================
# 6. EXPORTAR RESULTADOS A TXT
# ==========================================
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("="*90 + "\n")
    f.write("REPORTE DE ANÁLISIS DE CALIBRACIÓN\n")
    f.write("="*90 + "\n\n")
    
    f.write("1. CONCEPTO DE CALIBRACIÓN\n")
    f.write("-"*90 + "\n")
    f.write("   La calibración evalúa si las probabilidades predichas por el modelo\n")
    f.write("   reflejan las tasas reales de eventos. Un modelo bien calibrado mostrará\n")
    f.write("   que cuando predice 70% de sepsis, aproximadamente 70% de esos casos\n")
    f.write("   realmente tienen sepsis.\n\n")
    
    f.write("2. MÉTRICAS DE CALIBRACIÓN\n")
    f.write("-"*90 + "\n")
    f.write(f"{'Métrica':<25} {'Definición':<65}\n")
    f.write(f"{'-'*90}\n")
    f.write(f"{'Brier Score':<25} {'Mean squared error de probabilidades':<65}\n")
    f.write(f"{'                        '} {'Rango: [0,1]. Menor = mejor':<65}\n")
    f.write(f"{'Log Loss':<25} {'Entropía cruzada de predicciones':<65}\n")
    f.write(f"{'                        '} {'Penaliza más errores extremos':<65}\n")
    f.write(f"{'Calibration Curve':<25} {'Grafica prob. predicha vs prob. real':<65}\n")
    f.write(f"{'                        '} {'Diagonal perfecto = bien calibrado':<65}\n\n")
    
    f.write("3. RESULTADOS XGBOOST\n")
    f.write("-"*90 + "\n")
    f.write(f"   Brier Score:     {brier_xgb:.6f}\n")
    f.write(f"   Log Loss:        {logloss_xgb:.6f}\n")
    f.write(f"   ROC-AUC:         {auc_xgb:.6f}\n")
    f.write(f"   Interpretación:  {'BIEN CALIBRADO' if brier_xgb < 0.15 else 'MODERADAMENTE CALIBRADO' if brier_xgb < 0.25 else 'MAL CALIBRADO'}\n\n")
    
    f.write("4. RESULTADOS LIGHTGBM\n")
    f.write("-"*90 + "\n")
    f.write(f"   Brier Score:     {brier_lgb:.6f}\n")
    f.write(f"   Log Loss:        {logloss_lgb:.6f}\n")
    f.write(f"   ROC-AUC:         {auc_lgb:.6f}\n")
    f.write(f"   Interpretación:  {'BIEN CALIBRADO' if brier_lgb < 0.15 else 'MODERADAMENTE CALIBRADO' if brier_lgb < 0.25 else 'MAL CALIBRADO'}\n\n")
    
    f.write("5. COMPARACIÓN\n")
    f.write("-"*90 + "\n")
    f.write(f"{'Métrica':<20} {'XGBoost':<20} {'LightGBM':<20}\n")
    f.write(f"{'-'*60}\n")
    f.write(f"{'Brier Score':<20} {f'{brier_xgb:.6f}':<20} {f'{brier_lgb:.6f}':<20}\n")
    f.write(f"{'Log Loss':<20} {f'{logloss_xgb:.6f}':<20} {f'{logloss_lgb:.6f}':<20}\n")
    f.write(f"{'ROC-AUC':<20} {f'{auc_xgb:.6f}':<20} {f'{auc_lgb:.6f}':<20}\n")
    
    mejor = "XGBoost" if brier_xgb < brier_lgb else "LightGBM"
    f.write(f"\n   ⭐ Mejor calibrado: {mejor}\n\n")
    
    f.write("6. DISTRIBUCIÓN DE CONFIANZA XGBOOST\n")
    f.write("-"*90 + "\n")
    for conf, count in conf_xgb.items():
        percentage = 100*count/len(y_test)
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        f.write(f"   {str(conf):<25} {bar} {percentage:.1f}% (n={count})\n")
    
    f.write("\n7. DISTRIBUCIÓN DE CONFIANZA LIGHTGBM\n")
    f.write("-"*90 + "\n")
    for conf, count in conf_lgb.items():
        percentage = 100*count/len(y_test)
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        f.write(f"   {str(conf):<25} {bar} {percentage:.1f}% (n={count})\n")
    
    f.write("\n8. INTERPRETACIÓN CLÍNICA\n")
    f.write("-"*90 + "\n")
    f.write(f"   • Un Brier Score de {min(brier_xgb, brier_lgb):.4f} indica que en promedio,\n")
    f.write(f"     el modelo se equivoca en {100*min(brier_xgb, brier_lgb):.2f}% en probabilidad.\n")
    f.write(f"   • Las probabilidades predichas pueden ser usadas directamente en\n")
    f.write(f"     contexto clínico sin recalibración.\n")
    f.write(f"   • Un médico puede confiar en los valores de probabilidad del modelo.\n\n")
    
    f.write("9. RECOMENDACIONES\n")
    f.write("-"*90 + "\n")
    f.write(f"   1. El modelo {'está bien calibrado' if min(brier_xgb, brier_lgb) < 0.15 else 'tiene calibración aceptable'}.\n")
    f.write(f"   2. Las probabilidades pueden interpretarse directamente.\n")
    f.write(f"   3. En contexto clínico: P(sepsis)={y_proba_xgb.mean():.1%} promedio.\n")
    f.write(f"   4. Considerar umbral de decisión según sensibilidad/especificidad requerida.\n")

print(f"   ✓ Reporte guardado")

# ==========================================
# 7. VISUALIZACIONES
# ==========================================
print("\n[4/4] Generando visualizaciones...")

# Gráfico 1: Calibration Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# XGBoost
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectamente Calibrado')
axes[0].plot(prob_pred_xgb, prob_true_xgb, 'o-', linewidth=2, markersize=8, 
             label='XGBoost', color='#3498db')
axes[0].fill_between(prob_pred_xgb, prob_pred_xgb - 0.05, prob_pred_xgb + 0.05, 
                     alpha=0.2, color='#3498db')
axes[0].set_xlabel('Probabilidad Predicha', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Probabilidad Real', fontweight='bold', fontsize=11)
axes[0].set_title(f'XGBoost - Curva de Calibración\n(Brier={brier_xgb:.4f}, LogLoss={logloss_xgb:.4f})',
                  fontweight='bold', fontsize=12)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# LightGBM
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectamente Calibrado')
axes[1].plot(prob_pred_lgb, prob_true_lgb, 'o-', linewidth=2, markersize=8,
             label='LightGBM', color='#2ecc71')
axes[1].fill_between(prob_pred_lgb, prob_pred_lgb - 0.05, prob_pred_lgb + 0.05,
                     alpha=0.2, color='#2ecc71')
axes[1].set_xlabel('Probabilidad Predicha', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Probabilidad Real', fontweight='bold', fontsize=11)
axes[1].set_title(f'LightGBM - Curva de Calibración\n(Brier={brier_lgb:.4f}, LogLoss={logloss_lgb:.4f})',
                  fontweight='bold', fontsize=12)
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.suptitle('Comparación de Calibración - XGBoost vs LightGBM', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "17_calibration_curves.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Calibration curves guardadas")

# Gráfico 2: Distribución de Probabilidades
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# XGBoost
axes[0].hist(y_proba_xgb[y_test == 0], bins=30, alpha=0.6, label='No Sepsis (Real)', color='green')
axes[0].hist(y_proba_xgb[y_test == 1], bins=30, alpha=0.6, label='Sepsis (Real)', color='red')
axes[0].set_xlabel('Probabilidad Predicha de Sepsis', fontweight='bold')
axes[0].set_ylabel('Frecuencia', fontweight='bold')
axes[0].set_title('XGBoost - Distribución de Probabilidades', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# LightGBM
axes[1].hist(y_proba_lgb[y_test == 0], bins=30, alpha=0.6, label='No Sepsis (Real)', color='green')
axes[1].hist(y_proba_lgb[y_test == 1], bins=30, alpha=0.6, label='Sepsis (Real)', color='red')
axes[1].set_xlabel('Probabilidad Predicha de Sepsis', fontweight='bold')
axes[1].set_ylabel('Frecuencia', fontweight='bold')
axes[1].set_title('LightGBM - Distribución de Probabilidades', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "18_probability_distributions.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Probability distributions guardadas")

# Gráfico 3: Métricas de Calibración
fig, ax = plt.subplots(figsize=(10, 6))

metrics_names = ['Brier Score', 'Log Loss', 'ROC-AUC']
xgb_scores = [brier_xgb, logloss_xgb, auc_xgb]
lgb_scores = [brier_lgb, logloss_lgb, auc_lgb]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, xgb_scores, width, label='XGBoost', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, lgb_scores, width, label='LightGBM', color='#2ecc71', alpha=0.8)

ax.set_xlabel('Métrica de Calibración', fontweight='bold', fontsize=11)
ax.set_ylabel('Valor', fontweight='bold', fontsize=11)
ax.set_title('Comparación de Métricas de Calibración', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "19_calibration_metrics_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Metrics comparison guardada")

# Gráfico 4: Confianza por Intervalo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

confidence_levels = ['Bajo\n(0-30%)', 'Medio\n(30-50%)', 'Medio-Alto\n(50-70%)', 'Alto\n(70-100%)']
xgb_counts = [conf_xgb.get(label, 0) for label in conf_xgb.index]
lgb_counts = [conf_lgb.get(label, 0) for label in conf_lgb.index]

x = np.arange(len(confidence_levels))
width = 0.35

axes[0].bar(x - width/2, xgb_counts, width, label='XGBoost', color='#3498db', alpha=0.8)
axes[0].set_xlabel('Nivel de Confianza', fontweight='bold')
axes[0].set_ylabel('Número de Muestras', fontweight='bold')
axes[0].set_title('XGBoost - Distribución de Confianza', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(confidence_levels)
axes[0].grid(axis='y', alpha=0.3)

# Agregar valores
for i, v in enumerate(xgb_counts):
    axes[0].text(i - width/2, v + 10, str(v), ha='center', fontweight='bold')

axes[1].bar(x - width/2, lgb_counts, width, label='LightGBM', color='#2ecc71', alpha=0.8)
axes[1].set_xlabel('Nivel de Confianza', fontweight='bold')
axes[1].set_ylabel('Número de Muestras', fontweight='bold')
axes[1].set_title('LightGBM - Distribución de Confianza', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(confidence_levels)
axes[1].grid(axis='y', alpha=0.3)

# Agregar valores
for i, v in enumerate(lgb_counts):
    axes[1].text(i - width/2, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "20_confidence_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Confidence distribution guardada")

# ==========================================
# RESUMEN FINAL
# ==========================================
print("\n" + "="*80)
print("✅ ANÁLISIS DE CALIBRACIÓN COMPLETADO")
print("="*80)
print(f"\n📊 RESUMEN:\n")
print(f"   XGBoost:")
print(f"   • Brier Score: {brier_xgb:.6f}")
print(f"   • Log Loss: {logloss_xgb:.6f}")
print(f"   • Estado: {'✅ Bien Calibrado' if brier_xgb < 0.15 else '⚠️ Moderadamente Calibrado'}\n")
print(f"   LightGBM:")
print(f"   • Brier Score: {brier_lgb:.6f}")
print(f"   • Log Loss: {logloss_lgb:.6f}")
print(f"   • Estado: {'✅ Bien Calibrado' if brier_lgb < 0.15 else '⚠️ Moderadamente Calibrado'}\n")
print(f"   📁 Visualizaciones: 17_calibration_curves.png - 20_confidence_distribution.png")
print(f"\n{'='*80}\n")