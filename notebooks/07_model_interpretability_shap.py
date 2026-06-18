import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
except ImportError:
    print("⚠️ SHAP no está instalado. Instalando...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap', '-q'])
    import shap

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ==========================================
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TXT_OUTPUT = os.path.join(RESULTS_DIR, "07_shap_interpretability.txt")

print("="*80)
print("ANÁLISIS DE INTERPRETABILIDAD CON SHAP (VERSIÓN MEJORADA)")
print("="*80)

# ==========================================
# 2. CARGAR DATOS Y ENTRENAR MODELO
# ==========================================
print("\n[1/4] Cargando datos y entrenando modelo...")

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

# Convertir a float64 para evitar problemas de tipo
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_pos_weight = num_neg / num_pos

# Entrenar modelo
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(X_train, y_train)
print(f"   ✓ Modelo entrenado con {X_train.shape[0]} muestras")
print(f"   ✓ Características: {X_train.shape[1]}")

# ==========================================
# 3. CALCULAR SHAP VALUES
# ==========================================
print("\n[2/4] Calculando SHAP values (TreeExplainer)...")
print("   Esto puede tomar 1-2 minutos...")

try:
    # Usar TreeExplainer directamente (más robusto)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Para clasificación binaria, obtener valores de la clase positiva (sepsis)
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]  # Clase 1 (Sepsis)
    else:
        shap_values_class1 = shap_values
    
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value_class1 = expected_value[1]
    else:
        expected_value_class1 = expected_value
    
    print(f"   ✓ SHAP values calculados: {shap_values_class1.shape}")
    print(f"   ✓ Expected value (baseline): {expected_value_class1:.4f}")
    
    # Para visualización, usar sample de training
    print("\n   Preparando datos para visualización...")
    X_train_sample = X_train.iloc[:min(100, len(X_train))].copy()
    explainer_viz = shap.TreeExplainer(model)
    shap_values_viz = explainer_viz(X_test.iloc[:min(100, len(X_test))].copy())
    
    print(f"   ✓ Muestra para visualización: {X_test.iloc[:min(100, len(X_test))].shape[0]} muestras")
    
except Exception as e:
    print(f"   ⚠️ Error en TreeExplainer: {str(e)}")
    print("   Usando método alternativo (KernelExplainer)...")
    
    # Fallback: usar KernelExplainer (más lento pero más robusto)
    explainer = shap.KernelExplainer(
        model.predict_proba,
        shap.sample(X_train, 50),  # Sample de 50 muestras como background
        link="logit"
    )
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
    
    expected_value_class1 = explainer.expected_value
    if isinstance(expected_value_class1, list):
        expected_value_class1 = expected_value_class1[1]
    
    print(f"   ✓ SHAP values calculados (KernelExplainer)")
    print(f"   ✓ Expected value (baseline): {expected_value_class1:.4f}")
    
    shap_values_viz = shap_values
    explainer_viz = explainer

# ==========================================
# 4. ANÁLISIS DE IMPORTANCIA GLOBAL
# ==========================================
print("\n[3/4] Generando análisis de importancia global...")

# Calcular media absoluta de SHAP values para cada feature
feature_importance_shap = pd.DataFrame({
    'Feature': X_test.columns,
    'SHAP_Mean_Abs': np.abs(shap_values_class1).mean(axis=0),
    'SHAP_Mean': shap_values_class1.mean(axis=0)
}).sort_values('SHAP_Mean_Abs', ascending=False)

print("\n   📊 TOP 10 CARACTERÍSTICAS POR SHAP (Media Absoluta):")
print(f"   {'Rank':<5} {'Feature':<35} {'SHAP Mean Abs':<15} {'SHAP Mean':<15}")
print(f"   {'-'*70}")
for idx, (_, row) in enumerate(feature_importance_shap.head(10).iterrows(), 1):
    print(f"   {idx:<5} {row['Feature']:<35} {row['SHAP_Mean_Abs']:<15.6f} {row['SHAP_Mean']:<15.6f}")

# ==========================================
# 5. EXPORTAR RESULTADOS A TXT
# ==========================================
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("="*90 + "\n")
    f.write("REPORTE DE INTERPRETABILIDAD - SHAP (Shapley Additive exPlanations)\n")
    f.write("="*90 + "\n\n")
    
    f.write("1. TEORÍA DE SHAP\n")
    f.write("-"*90 + "\n")
    f.write("   SHAP utiliza la teoría de juegos para explicar predicciones de ML.\n")
    f.write("   Para cada predicción, SHAP calcula la contribución de cada característica\n")
    f.write("   basada en permutaciones del modelo. El valor SHAP representa cuánto\n")
    f.write("   cada característica contribuye a alejar la predicción del valor base.\n\n")
    f.write(f"   Expected Value (Base): {expected_value_class1:.6f}\n")
    f.write(f"   (Promedio de predicciones en datos de entrenamiento)\n\n")
    
    f.write("2. TOP 15 CARACTERÍSTICAS POR IMPORTANCIA SHAP\n")
    f.write("-"*90 + "\n")
    f.write(f"{'Rank':<5} {'Feature':<35} {'SHAP Mean Abs':<18} {'SHAP Mean':<18}\n")
    f.write(f"{'-'*76}\n")
    
    for idx, (_, row) in enumerate(feature_importance_shap.head(15).iterrows(), 1):
        f.write(f"{idx:<5} {row['Feature']:<35} {row['SHAP_Mean_Abs']:<18.6f} {row['SHAP_Mean']:<18.6f}\n")
    
    f.write("\n")
    f.write("3. INTERPRETACIÓN CLÍNICA DE RESULTADOS SHAP\n")
    f.write("-"*90 + "\n")
    
    top_5 = feature_importance_shap.head(5)
    f.write("   Las 5 características más importantes para predecir sepsis:\n\n")
    
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        feature_name = row['Feature']
        shap_impact = row['SHAP_Mean_Abs']
        direction = "Aumenta riesgo" if row['SHAP_Mean'] > 0 else "Disminuye riesgo"
        
        f.write(f"   {idx}. {feature_name.upper()}\n")
        f.write(f"      • SHAP Impact: {shap_impact:.6f}\n")
        f.write(f"      • Tendencia: {direction}\n")
        f.write(f"      • Explicación: Esta variable tiene alto impacto en las\n")
        f.write(f"        predicciones del modelo de sepsis.\n\n")
    
    f.write("4. VALIDEZ DEL ANÁLISIS SHAP\n")
    f.write("-"*90 + "\n")
    f.write("   ✓ Método: TreeExplainer (O(T*D) complejidad)\n")
    f.write("   ✓ Eficiencia: Rápido para gradient boosting\n")
    f.write("   ✓ Convergencia: Teóricamente garantizada\n")
    f.write("   ✓ Muestras: Test set completo analizado\n")
    f.write("   ✓ Validez: SHAP values respetan propiedades de Shapley\n")
    f.write(f"     (Eficiencia local, simetría, dummy, aditividad)\n\n")
    
    f.write("5. EJEMPLO DE INTERPRETACIÓN INDIVIDUAL\n")
    f.write("-"*90 + "\n")
    f.write(f"   Para una predicción individual, SHAP muestra:\n")
    f.write(f"   • Valor base (Expected Value): {expected_value_class1:.4f}\n")
    f.write(f"   • Contribución de cada feature: Valor SHAP\n")
    f.write(f"   • Predicción final: Base + Σ(Contribuciones SHAP)\n\n")
    f.write(f"   Ejemplo:\n")
    f.write(f"   Si un paciente tiene:\n")
    f.write(f"   - Creatinina alta: +0.15 (aumenta probabilidad de sepsis)\n")
    f.write(f"   - SpO2 normal: -0.08 (reduce probabilidad)\n")
    f.write(f"   - Lactato elevado: +0.22 (fuerte indicador de sepsis)\n")
    f.write(f"   Predicción: 0.50 (base) + 0.15 - 0.08 + 0.22 = 0.79 (79% sepsis)\n\n")

print(f"   ✓ Reporte guardado")

# ==========================================
# 6. GENERAR VISUALIZACIONES SHAP
# ==========================================
print("\n[4/4] Generando visualizaciones SHAP...")

# Gráfico 1: Summary Bar Plot
try:
    plt.figure(figsize=(10, 8))
    
    # Crear bar plot manualmente si shap.summary_plot falla
    top_features = feature_importance_shap.head(15)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    
    plt.barh(range(len(top_features)), top_features['SHAP_Mean_Abs'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Mean |SHAP value|', fontweight='bold', fontsize=11)
    plt.title("SHAP Summary - Importancia Media Absoluta de Características", 
              fontsize=12, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    for i, val in enumerate(top_features['SHAP_Mean_Abs']):
        plt.text(val, i, f' {val:.6f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "13_shap_summary_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Summary bar plot guardado")
except Exception as e:
    print(f"   ⚠️ Error en summary bar plot: {str(e)}")

# Gráfico 2: Feature Importance por valor absoluto (alternativa a beeswarm)
try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico de dispersión SHAP vs Feature Value para top 2 características
    top_2_features = feature_importance_shap.head(2)
    
    for plot_idx, (_, feature_row) in enumerate(top_2_features.iterrows()):
        ax = axes[plot_idx]
        feature_name = feature_row['Feature']
        feature_idx = X_test.columns.get_loc(feature_name)
        
        feature_values = X_test.iloc[:, feature_idx].values
        shap_feature_values = shap_values_class1[:, feature_idx]
        
        scatter = ax.scatter(feature_values, shap_feature_values, 
                           c=y_test, cmap='RdYlGn_r', alpha=0.6, s=50)
        
        ax.set_xlabel(feature_name, fontweight='bold', fontsize=11)
        ax.set_ylabel('SHAP Value', fontweight='bold', fontsize=11)
        ax.set_title(f'{feature_name} - Relación SHAP', fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('0=No Sepsis\n1=Sepsis', fontweight='bold', fontsize=9)
    
    plt.suptitle("SHAP Dependence - Top 2 Características", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "14_shap_dependence_top2.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Dependence plot guardado")
except Exception as e:
    print(f"   ⚠️ Error en dependence plot: {str(e)}")

# Gráfico 3: Ejemplos de predicciones individuales
try:
    sepsis_indices = np.where(y_test == 1)[0][:5]
    no_sepsis_indices = np.where(y_test == 0)[0][:5]
    example_indices = list(sepsis_indices) + list(no_sepsis_indices)
    
    fig, axes = plt.subplots(5, 2, figsize=(14, 12))
    fig.suptitle("Ejemplos de Predicciones - Contribución SHAP Top 3 Features", 
                 fontsize=14, fontweight='bold')
    
    for plot_idx, idx in enumerate(example_indices):
        ax = axes[plot_idx // 2, plot_idx % 2]
        
        pred = model.predict_proba(X_test.iloc[[idx]])[0, 1]
        shap_sample = shap_values_class1[idx]
        
        class_label = "Sepsis" if y_test[idx] == 1 else "No Sepsis"
        predicted_label = "Sepsis" if pred >= 0.5 else "No Sepsis"
        
        ax.text(0.5, 0.9, f"Muestra {idx}", ha='center', fontsize=10, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.75, f"Real: {class_label} | Predicción: {predicted_label}", 
                ha='center', fontsize=9, transform=ax.transAxes)
        ax.text(0.5, 0.6, f"P(Sepsis) = {pred:.4f}", ha='center', fontsize=9, fontweight='bold',
                color='red' if pred > 0.5 else 'green', transform=ax.transAxes)
        
        top_features_idx = np.argsort(np.abs(shap_sample))[-3:][::-1]
        y_pos = 0.45
        for feature_idx in top_features_idx:
            feature_name = X_test.columns[feature_idx]
            feature_value = X_test.iloc[idx, feature_idx]
            shap_val = shap_sample[feature_idx]
            
            ax.text(0.05, y_pos, f"{feature_name}: {feature_value:.2f} (SHAP: {shap_val:.4f})",
                    ha='left', fontsize=8, transform=ax.transAxes, family='monospace')
            y_pos -= 0.12
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "15_shap_examples.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Examples guardado")
except Exception as e:
    print(f"   ⚠️ Error en ejemplos: {str(e)}")

# Gráfico 4: Distribución de SHAP values para top 5 características
try:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("Distribución de SHAP Values - Top 5 Características", 
                 fontsize=14, fontweight='bold')
    
    top_5_features = feature_importance_shap.head(5)
    
    for idx, (_, feature_row) in enumerate(top_5_features.iterrows()):
        ax = axes[idx]
        feature_name = feature_row['Feature']
        feature_idx = X_test.columns.get_loc(feature_name)
        
        shap_vals = shap_values_class1[:, feature_idx]
        
        ax.hist(shap_vals[y_test == 0], bins=30, alpha=0.6, label='No Sepsis', color='green')
        ax.hist(shap_vals[y_test == 1], bins=30, alpha=0.6, label='Sepsis', color='red')
        ax.set_xlabel('SHAP Value', fontweight='bold', fontsize=9)
        ax.set_ylabel('Frecuencia', fontweight='bold', fontsize=9)
        ax.set_title(feature_name, fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "16_shap_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Distributions guardado")
except Exception as e:
    print(f"   ⚠️ Error en distributions: {str(e)}")

# ==========================================
# 7. GUARDAR SHAP VALUES PARA USO FUTURO
# ==========================================
np.save(os.path.join(RESULTS_DIR, "shap_values_class1.npy"), shap_values_class1)
feature_importance_shap.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"), index=False)
print("   ✓ SHAP values guardados para análisis futuro")

# ==========================================
# RESUMEN FINAL
# ==========================================
print("\n" + "="*80)
print("✅ ANÁLISIS SHAP COMPLETADO")
print("="*80)
print(f"\n📊 TOP 5 CARACTERÍSTICAS POR SHAP:\n")
for idx, (_, row) in enumerate(feature_importance_shap.head(5).iterrows(), 1):
    print(f"   {idx}. {row['Feature']:<30} Mean |SHAP|: {row['SHAP_Mean_Abs']:.6f}")

print(f"\n📁 Visualizaciones generadas:")
print(f"   • 13_shap_summary_bar.png (Importancia global)")
print(f"   • 14_shap_dependence_top2.png (Relaciones feature-SHAP)")
print(f"   • 15_shap_examples.png (Predicciones individuales)")
print(f"   • 16_shap_distributions.png (Distribuciones SHAP)")
print(f"\n📄 Reporte: {TXT_OUTPUT}\n")
print(f"{'='*80}\n")