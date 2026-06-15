import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Definir rutas locales
DATA_PATH = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\raw\sepsis_icu_synthetic.csv"
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"
TXT_OUTPUT = os.path.join(RESULTS_DIR, "02_eda_results.txt")

# Asegurar que las carpetas existan
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Cargando datos para el Análisis Exploratorio (EDA)...")
df = pd.read_csv(DATA_PATH)

# ==========================================
# A. CÁLCULOS MATEMÁTICOS Y ESTADÍSTICOS
# ==========================================
print("Generando reportes estadísticos...")
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write("        REPORTE DE ANÁLISIS EXPLORATORIO (EDA)\n")
    f.write("==================================================\n\n")
    
    # 1. Balance de la variable objetivo (Sepsis vs No Sepsis)
    f.write("1. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (sepsis_label)\n")
    f.write("--------------------------------------------------\n")
    sepsis_counts = df['sepsis_label'].value_counts()
    sepsis_pct = df['sepsis_label'].value_counts(normalize=True) * 100
    for val, count in sepsis_counts.items():
        label = "Sepsis" if val == 1 else "No Sepsis"
        f.write(f"- {label}: {count} pacientes ({sepsis_pct[val]:.2f}%)\n")
    f.write("\n")

    # 2. Estadísticas descriptivas de variables numéricas clave
    f.write("2. ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES CLAVE\n")
    f.write("--------------------------------------------------\n")
    features_clave = ['age', 'bmi', 'hr_mean', 'sbp_mean', 'temp_celsius_mean', 'spo2_mean', 'wbc', 'lactate_mmol', 'sofa_score']
    f.write(df[features_clave].describe().to_string())
    f.write("\n\n")

    # 3. Correlación lineal de Pearson con la Sepsis
    f.write("3. TOP CORRELACIONES CON SEPSIS_LABEL\n")
    f.write("--------------------------------------------------\n")
    # Seleccionar solo columnas numéricas para la correlación
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    correlations = numeric_df.corr()['sepsis_label'].sort_values(ascending=False)
    f.write("Variables más correlacionadas positivamente:\n")
    f.write(correlations.head(6).to_string())
    f.write("\n\nVariables más correlacionadas negativamente:\n")
    f.write(correlations.tail(5).to_string())
    f.write("\n")

# ==========================================
# B. GENERACIÓN DE GRÁFICOS PROFESIONALES
# ==========================================
print("Generando gráficos de alta calidad...")
sns.set_theme(style="whitegrid")

# Gráfico 1: Distribución de la variable objetivo
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='sepsis_label', data=df, hue='sepsis_label', palette="Set2", legend=False)
plt.title("Distribución de Casos de Sepsis en UCI", fontsize=12, fontweight='bold')
plt.xlabel("Estado (0 = No Sepsis, 1 = Sepsis)")
plt.ylabel("Número de Pacientes")
plt.xticks([0, 1], ['No Sepsis', 'Sepsis'])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_distribucion_sepsis.png"), dpi=300)
plt.close()

# Gráfico 2: Edad vs Sepsis (Boxplot)
plt.figure(figsize=(7, 4))
sns.boxplot(x='sepsis_label', y='age', data=df, hue='sepsis_label', palette="Pastel1", legend=False)
plt.title("Distribución de Edad según Presencia de Sepsis", fontsize=12, fontweight='bold')
plt.xlabel("Sepsis Label")
plt.ylabel("Edad (Años)")
plt.xticks([0, 1], ['No Sepsis', 'Sepsis'])
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_edad_vs_sepsis.png"), dpi=300)
plt.close()

# Gráfico 3: Matriz de Correlación reducida (Heatmap)
plt.figure(figsize=(8, 6))
corr_matrix = df[features_clave + ['sepsis_label']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación - Variables Clínicas Clave", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_matriz_correlacion.png"), dpi=300)
plt.close()

print(f"✅ Reporte guardado en: {TXT_OUTPUT}")
print(f"✅ Gráficos exportados en: {PLOTS_DIR}")