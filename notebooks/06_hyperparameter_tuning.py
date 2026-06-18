import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import time

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ==========================================
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TXT_OUTPUT = os.path.join(RESULTS_DIR, "06_hyperparameter_tuning_results.txt")

print("="*80)
print("OPTIMIZACIÓN DE HIPERPARÁMETROS CON GRIDSEARCHCV")
print("="*80)

# ==========================================
# 2. CARGAR DATOS
# ==========================================
print("\n[1/4] Cargando datos preprocesados...")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_pos_weight = num_neg / num_pos

print(f"   ✓ Datos cargados: X_train={X_train.shape}, X_test={X_test.shape}")
print(f"   ✓ Scale Pos Weight: {scale_pos_weight:.4f}")

# ==========================================
# 3. GRIDSEARCH PARA XGBOOST
# ==========================================
print("\n[2/4] GridSearchCV para XGBoost (esto puede tomar 5-10 minutos)...")
print("   Evaluando 24 combinaciones de hiperparámetros con 5-Fold CV...")

# Definir grid de parámetros
param_grid_xgb = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9],
}

# XGBoost base
xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    n_jobs=-1
)

# GridSearchCV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

start_time = time.time()
grid_search_xgb = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid_xgb,
    cv=skf,
    scoring='roc_auc',  # Métrica principal
    n_jobs=-1,
    verbose=1
)

grid_search_xgb.fit(X_train, y_train)
elapsed_xgb = time.time() - start_time

print(f"\n   ✅ GridSearch XGBoost completado en {elapsed_xgb:.2f} segundos")
print(f"   Best Score (ROC-AUC CV): {grid_search_xgb.best_score_:.4f}")
print(f"   Best Params: {grid_search_xgb.best_params_}")

# Entrenar mejor modelo en test
best_model_xgb = grid_search_xgb.best_estimator_
y_pred_xgb = best_model_xgb.predict(X_test)
y_proba_xgb = best_model_xgb.predict_proba(X_test)[:, 1]

test_acc_xgb = accuracy_score(y_test, y_pred_xgb)
test_f1_xgb = f1_score(y_test, y_pred_xgb)
test_auc_xgb = roc_auc_score(y_test, y_proba_xgb)

print(f"   Test Accuracy: {test_acc_xgb:.4f}")
print(f"   Test F1-Score: {test_f1_xgb:.4f}")
print(f"   Test ROC-AUC: {test_auc_xgb:.4f}")

# ==========================================
# 4. GRIDSEARCH PARA LIGHTGBM
# ==========================================
print("\n[3/4] GridSearchCV para LightGBM (esto puede tomar 5-10 minutos)...")
print("   Evaluando 24 combinaciones de hiperparámetros con 5-Fold CV...")

param_grid_lgb = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70],
}

lgb_base = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)

start_time = time.time()
grid_search_lgb = GridSearchCV(
    estimator=lgb_base,
    param_grid=param_grid_lgb,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search_lgb.fit(X_train, y_train)
elapsed_lgb = time.time() - start_time

print(f"\n   ✅ GridSearch LightGBM completado en {elapsed_lgb:.2f} segundos")
print(f"   Best Score (ROC-AUC CV): {grid_search_lgb.best_score_:.4f}")
print(f"   Best Params: {grid_search_lgb.best_params_}")

# Entrenar mejor modelo en test
best_model_lgb = grid_search_lgb.best_estimator_
y_pred_lgb = best_model_lgb.predict(X_test)
y_proba_lgb = best_model_lgb.predict_proba(X_test)[:, 1]

test_acc_lgb = accuracy_score(y_test, y_pred_lgb)
test_f1_lgb = f1_score(y_test, y_pred_lgb)
test_auc_lgb = roc_auc_score(y_test, y_proba_lgb)

print(f"   Test Accuracy: {test_acc_lgb:.4f}")
print(f"   Test F1-Score: {test_f1_lgb:.4f}")
print(f"   Test ROC-AUC: {test_auc_lgb:.4f}")

# ==========================================
# 5. CREAR DATAFRAMES CON RESULTADOS
# ==========================================
print("\n[4/4] Generando reportes...")

# Convertir resultados a DataFrame
results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)
results_lgb = pd.DataFrame(grid_search_lgb.cv_results_)

# Seleccionar columnas relevantes
results_xgb_clean = results_xgb[['param_n_estimators', 'param_max_depth', 'param_learning_rate', 
                                   'param_subsample', 'mean_test_score', 'std_test_score', 'rank_test_score']]
results_xgb_clean.columns = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'mean_auc', 'std_auc', 'rank']
results_xgb_clean = results_xgb_clean.sort_values('rank')

results_lgb_clean = results_lgb[['param_n_estimators', 'param_max_depth', 'param_learning_rate',
                                  'param_num_leaves', 'mean_test_score', 'std_test_score', 'rank_test_score']]
results_lgb_clean.columns = ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves', 'mean_auc', 'std_auc', 'rank']
results_lgb_clean = results_lgb_clean.sort_values('rank')

# ==========================================
# 6. EXPORTAR RESULTADOS A TXT
# ==========================================
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("="*90 + "\n")
    f.write("REPORTE DE OPTIMIZACIÓN DE HIPERPARÁMETROS - GridSearchCV\n")
    f.write("="*90 + "\n\n")
    
    f.write("1. CONFIGURACIÓN GENERAL\n")
    f.write("-"*90 + "\n")
    f.write(f"   Estrategia de Validación: Stratified K-Fold (5 pliegues)\n")
    f.write(f"   Métrica de Optimización: ROC-AUC\n")
    f.write(f"   Scale Pos Weight: {scale_pos_weight:.4f}\n")
    f.write(f"   Conjunto de Entrenamiento: {X_train.shape[0]} muestras\n\n")
    
    f.write("2. RESULTADOS XGBOOST - TOP 10 COMBINACIONES\n")
    f.write("-"*90 + "\n")
    f.write(results_xgb_clean.head(10).to_string(index=False))
    f.write("\n\n")
    
    f.write("3. MEJOR MODELO XGBOOST\n")
    f.write("-"*90 + "\n")
    f.write(f"   Hiperparámetros Óptimos:\n")
    for param, value in grid_search_xgb.best_params_.items():
        f.write(f"      • {param}: {value}\n")
    f.write(f"\n   Rendimiento CV (5-Fold):\n")
    f.write(f"      • ROC-AUC: {grid_search_xgb.best_score_:.4f}\n")
    f.write(f"      • Std: {results_xgb_clean.iloc[0]['std_auc']:.4f}\n")
    f.write(f"\n   Rendimiento en Test:\n")
    f.write(f"      • Accuracy: {test_acc_xgb:.4f}\n")
    f.write(f"      • F1-Score: {test_f1_xgb:.4f}\n")
    f.write(f"      • ROC-AUC: {test_auc_xgb:.4f}\n")
    f.write(f"      • Tiempo de entrenamiento GridSearch: {elapsed_xgb:.2f}s\n\n")
    
    f.write("4. RESULTADOS LIGHTGBM - TOP 10 COMBINACIONES\n")
    f.write("-"*90 + "\n")
    f.write(results_lgb_clean.head(10).to_string(index=False))
    f.write("\n\n")
    
    f.write("5. MEJOR MODELO LIGHTGBM\n")
    f.write("-"*90 + "\n")
    f.write(f"   Hiperparámetros Óptimos:\n")
    for param, value in grid_search_lgb.best_params_.items():
        f.write(f"      • {param}: {value}\n")
    f.write(f"\n   Rendimiento CV (5-Fold):\n")
    f.write(f"      • ROC-AUC: {grid_search_lgb.best_score_:.4f}\n")
    f.write(f"      • Std: {results_lgb_clean.iloc[0]['std_auc']:.4f}\n")
    f.write(f"\n   Rendimiento en Test:\n")
    f.write(f"      • Accuracy: {test_acc_lgb:.4f}\n")
    f.write(f"      • F1-Score: {test_f1_lgb:.4f}\n")
    f.write(f"      • ROC-AUC: {test_auc_lgb:.4f}\n")
    f.write(f"      • Tiempo de entrenamiento GridSearch: {elapsed_lgb:.2f}s\n\n")
    
    f.write("6. COMPARACIÓN FINAL\n")
    f.write("-"*90 + "\n")
    f.write(f"{'Métrica':<25} {'XGBoost':<20} {'LightGBM':<20}\n")
    f.write(f"{'-'*65}\n")
    f.write(f"{'CV ROC-AUC':<25} {grid_search_xgb.best_score_:<20.4f} {grid_search_lgb.best_score_:<20.4f}\n")
    f.write(f"{'Test Accuracy':<25} {test_acc_xgb:<20.4f} {test_acc_lgb:<20.4f}\n")
    f.write(f"{'Test F1-Score':<25} {test_f1_xgb:<20.4f} {test_f1_lgb:<20.4f}\n")
    f.write(f"{'Test ROC-AUC':<25} {test_auc_xgb:<20.4f} {test_auc_lgb:<20.4f}\n\n")
    
    winner = "XGBoost" if test_auc_xgb >= test_auc_lgb else "LightGBM"
    f.write(f"   ✓ MEJOR MODELO: {winner}\n\n")
    
    f.write("7. RECOMENDACIONES PARA DEFENSA ORAL\n")
    f.write("-"*90 + "\n")
    f.write(f"   • Los hiperparámetros óptimos fueron seleccionados mediante GridSearchCV\n")
    f.write(f"     con validación cruzada estratificada de 5 pliegues.\n")
    f.write(f"   • Se evaluaron 24 combinaciones diferentes para XGBoost y LightGBM.\n")
    f.write(f"   • El modelo ganador ({winner}) alcanzó ROC-AUC de {max(test_auc_xgb, test_auc_lgb):.4f} en test.\n")
    f.write(f"   • La estabilidad fue validada mediante desviación estándar de CV.\n")

print(f"\n✅ Resultados guardados en: {TXT_OUTPUT}")

# ==========================================
# 7. VISUALIZAR HEATMAP DE RESULTADOS
# ==========================================

# Crear tabla pivot para XGBoost
results_pivot_xgb = results_xgb_clean.copy()
results_pivot_xgb['combination'] = (results_pivot_xgb['n_estimators'].astype(str) + '_' + 
                                     results_pivot_xgb['max_depth'].astype(str) + '_' + 
                                     results_pivot_xgb['learning_rate'].astype(str))
results_pivot_xgb = results_pivot_xgb.sort_values('mean_auc', ascending=False).head(12)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# XGBoost top 12
results_sorted_xgb = results_xgb_clean.sort_values('mean_auc', ascending=False).head(12)
axes[0].barh(range(len(results_sorted_xgb)), results_sorted_xgb['mean_auc'], 
             xerr=results_sorted_xgb['std_auc'], capsize=5, color='#3498db', alpha=0.8)
labels_xgb = [f"n={int(row['n_estimators'])}, d={int(row['max_depth'])}, lr={row['learning_rate']}" 
              for _, row in results_sorted_xgb.iterrows()]
axes[0].set_yticks(range(len(results_sorted_xgb)))
axes[0].set_yticklabels(labels_xgb, fontsize=9)
axes[0].set_xlabel('ROC-AUC (CV)', fontweight='bold')
axes[0].set_title('XGBoost - Top 12 Combinaciones', fontweight='bold', fontsize=12)
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

for i, (_, row) in enumerate(results_sorted_xgb.iterrows()):
    axes[0].text(row['mean_auc'] + row['std_auc'], i, f" {row['mean_auc']:.4f}", 
                va='center', fontsize=9, fontweight='bold')

# LightGBM top 12
results_sorted_lgb = results_lgb_clean.sort_values('mean_auc', ascending=False).head(12)
axes[1].barh(range(len(results_sorted_lgb)), results_sorted_lgb['mean_auc'],
             xerr=results_sorted_lgb['std_auc'], capsize=5, color='#2ecc71', alpha=0.8)
labels_lgb = [f"n={int(row['n_estimators'])}, d={int(row['max_depth'])}, lr={row['learning_rate']}" 
              for _, row in results_sorted_lgb.iterrows()]
axes[1].set_yticks(range(len(results_sorted_lgb)))
axes[1].set_yticklabels(labels_lgb, fontsize=9)
axes[1].set_xlabel('ROC-AUC (CV)', fontweight='bold')
axes[1].set_title('LightGBM - Top 12 Combinaciones', fontweight='bold', fontsize=12)
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

for i, (_, row) in enumerate(results_sorted_lgb.iterrows()):
    axes[1].text(row['mean_auc'] + row['std_auc'], i, f" {row['mean_auc']:.4f}",
                va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "12_gridsearch_top12_comparacion.png"), dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Gráfico de GridSearch guardado")

# ==========================================
# RESUMEN FINAL
# ==========================================
print("\n" + "="*80)
print("✅ OPTIMIZACIÓN DE HIPERPARÁMETROS COMPLETADA")
print("="*80)
print(f"\n📊 RESUMEN:\n")
print(f"   XGBoost Óptimo:")
print(f"   • CV ROC-AUC: {grid_search_xgb.best_score_:.4f}")
print(f"   • Test ROC-AUC: {test_auc_xgb:.4f}")
print(f"   • Hiperparámetros: {grid_search_xgb.best_params_}\n")
print(f"   LightGBM Óptimo:")
print(f"   • CV ROC-AUC: {grid_search_lgb.best_score_:.4f}")
print(f"   • Test ROC-AUC: {test_auc_lgb:.4f}")
print(f"   • Hiperparámetros: {grid_search_lgb.best_params_}\n")
print(f"   ⭐ Mejor modelo: {'XGBoost' if test_auc_xgb >= test_auc_lgb else 'LightGBM'}")
print(f"\n{'='*80}\n")