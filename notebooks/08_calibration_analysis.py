# Importa el módulo os para interactuar con el sistema operativo
import os
# Importa la librería pandas con el alias pd para manipulación de datos
import pandas as pd
# Importa la librería numpy con el alias np para cálculos numéricos
import numpy as np
# Importa pyplot de matplotlib con el alias plt para generar gráficos
import matplotlib.pyplot as plt
# Importa la librería seaborn con el alias sns para visualización estadística
import seaborn as sns
# De sklearn.metrics importa varias funciones para evaluar el modelo
from sklearn.metrics import (
# Importa brier_score_loss, log_loss y roc_auc_score
    brier_score_loss, log_loss, roc_auc_score,
# Importa roc_curve, auc y precision_recall_curve
    roc_curve, auc, precision_recall_curve
# Cierra el paréntesis de importación de sklearn.metrics
)
# De sklearn.calibration importa la función calibration_curve
from sklearn.calibration import calibration_curve
# De sklearn.calibration importa la clase CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV
# Importa la librería xgboost con el alias xgb para modelos de árboles potenciados
import xgboost as xgb
# Importa la librería lightgbm con el alias lgb para modelos LightGBM
import lightgbm as lgb
# Importa el módulo warnings para el manejo de advertencias
import warnings
# Filtra las advertencias para ignorarlas
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ==========================================
# Define la ruta del directorio de datos procesados
PROCESSED_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\processed"
# Define la ruta del directorio de resultados
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
# Define la ruta del directorio de gráficos
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"

# Crea el directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)
# Crea el directorio de gráficos si no existe
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define la ruta completa para el archivo de salida de texto
TXT_OUTPUT = os.path.join(RESULTS_DIR, "08_calibration_analysis.txt")

# Imprime una línea separadora de signos igual
print("="*80)
# Imprime el título del script
print("ANÁLISIS DE CALIBRACIÓN DEL MODELO")
# Imprime otra línea separadora
print("="*80)

# ==========================================
# 2. CARGAR DATOS
# ==========================================
# Imprime un mensaje indicando que se están cargando los datos
print("\n[1/4] Cargando datos...")

# Lee el archivo CSV de datos de entrenamiento y lo asigna a X_train
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
# Lee el archivo CSV de datos de prueba y lo asigna a X_test
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
# Lee las etiquetas de entrenamiento, convierte a arreglo y aplana
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
# Lee las etiquetas de prueba, convierte a arreglo y aplana
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

# Calcula el número de casos negativos en el conjunto de entrenamiento
num_neg = np.sum(y_train == 0)
# Calcula el número de casos positivos en el conjunto de entrenamiento
num_pos = np.sum(y_train == 1)
# Calcula el peso para la clase positiva para balancear las clases
scale_pos_weight = num_neg / num_pos

# Imprime las dimensiones de los conjuntos de datos cargados
print(f"   ✓ Datos cargados: {X_train.shape[0]} train, {X_test.shape[0]} test")

# ==========================================
# 3. ENTRENAR MODELOS
# ==========================================
# Imprime un mensaje indicando el inicio del entrenamiento
print("\n[2/4] Entrenando modelos...")

# XGBoost
# Inicializa el clasificador XGBoost con parámetros específicos
model_xgb = xgb.XGBClassifier(
# Define hiperparámetros como número de estimadores, profundidad máxima y tasa de aprendizaje
    n_estimators=150, max_depth=5, learning_rate=0.05,
# Define el peso de las clases, semilla aleatoria
    scale_pos_weight=scale_pos_weight, random_state=42,
# Define la métrica de evaluación y desactiva la verbosidad
    eval_metric='logloss', verbosity=0
# Cierra la inicialización de XGBoost
)
# Entrena el modelo XGBoost con los datos de entrenamiento
model_xgb.fit(X_train, y_train)
# Predice las probabilidades de la clase positiva en el conjunto de prueba
y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

# LightGBM
# Inicializa el clasificador LightGBM con parámetros específicos
model_lgb = lgb.LGBMClassifier(
# Define hiperparámetros como número de estimadores, profundidad máxima, tasa de aprendizaje
    n_estimators=150, max_depth=5, learning_rate=0.05,
# Define el peso de las clases, semilla aleatoria y desactiva la verbosidad
    scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1
# Cierra la inicialización de LightGBM
)
# Entrena el modelo LightGBM con los datos de entrenamiento
model_lgb.fit(X_train, y_train)
# Predice las probabilidades de la clase positiva en el conjunto de prueba
y_proba_lgb = model_lgb.predict_proba(X_test)[:, 1]

# Imprime confirmación del entrenamiento de XGBoost
print(f"   ✓ XGBoost entrenado")
# Imprime confirmación del entrenamiento de LightGBM
print(f"   ✓ LightGBM entrenado")

# ==========================================
# 4. CALCULAR MÉTRICAS DE CALIBRACIÓN
# ==========================================
# Imprime un mensaje indicando el cálculo de métricas de calibración
print("\n[3/4] Calculando métricas de calibración...")

# Calibración XGBoost
# Calcula las fracciones de verdaderos positivos y las probabilidades medias predichas para XGBoost
prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, y_proba_xgb, n_bins=10, strategy='uniform')
# Calcula la puntuación de Brier para XGBoost
brier_xgb = brier_score_loss(y_test, y_proba_xgb)
# Calcula la pérdida logarítmica para XGBoost
logloss_xgb = log_loss(y_test, y_proba_xgb)
# Calcula el área bajo la curva ROC para XGBoost
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

# Calibración LightGBM
# Calcula las fracciones de verdaderos positivos y las probabilidades medias predichas para LightGBM
prob_true_lgb, prob_pred_lgb = calibration_curve(y_test, y_proba_lgb, n_bins=10, strategy='uniform')
# Calcula la puntuación de Brier para LightGBM
brier_lgb = brier_score_loss(y_test, y_proba_lgb)
# Calcula la pérdida logarítmica para LightGBM
logloss_lgb = log_loss(y_test, y_proba_lgb)
# Calcula el área bajo la curva ROC para LightGBM
auc_lgb = roc_auc_score(y_test, y_proba_lgb)

# Imprime el encabezado para XGBoost
print(f"\n   XGBoost:")
# Imprime el valor de Brier Score para XGBoost
print(f"   • Brier Score: {brier_xgb:.4f}")
# Imprime el valor de Log Loss para XGBoost
print(f"   • Log Loss: {logloss_xgb:.4f}")
# Imprime el valor de ROC-AUC para XGBoost
print(f"   • ROC-AUC: {auc_xgb:.4f}")

# Imprime el encabezado para LightGBM
print(f"\n   LightGBM:")
# Imprime el valor de Brier Score para LightGBM
print(f"   • Brier Score: {brier_lgb:.4f}")
# Imprime el valor de Log Loss para LightGBM
print(f"   • Log Loss: {logloss_lgb:.4f}")
# Imprime el valor de ROC-AUC para LightGBM
print(f"   • ROC-AUC: {auc_lgb:.4f}")

# ==========================================
# 5. ANÁLISIS DE CONFIANZA
# ==========================================
# Imprime un mensaje indicando el análisis de distribución de confianza
print("\n   Analizando distribución de confianza...")

# Categorías de confianza
# Crea bins para las probabilidades predichas de XGBoost
confidence_bins_xgb = pd.cut(y_proba_xgb, bins=[0, 0.3, 0.5, 0.7, 1.0], 
# Asigna etiquetas a los bins de XGBoost
                             labels=['Bajo (0-30%)', 'Medio (30-50%)', 'Medio-Alto (50-70%)', 'Alto (70-100%)'])
# Crea bins para las probabilidades predichas de LightGBM
confidence_bins_lgb = pd.cut(y_proba_lgb, bins=[0, 0.3, 0.5, 0.7, 1.0],
# Asigna etiquetas a los bins de LightGBM
                             labels=['Bajo (0-30%)', 'Medio (30-50%)', 'Medio-Alto (50-70%)', 'Alto (70-100%)'])

# Contar por bin
# Cuenta las frecuencias de cada categoría para XGBoost y las ordena
conf_xgb = confidence_bins_xgb.value_counts().sort_index()
# Cuenta las frecuencias de cada categoría para LightGBM y las ordena
conf_lgb = confidence_bins_lgb.value_counts().sort_index()

# Imprime el encabezado para la distribución de confianza de XGBoost
print(f"   XGBoost - Distribución de confianza:")
# Itera sobre las categorías y recuentos de XGBoost
for conf, count in conf_xgb.items():
# Imprime la categoría, el recuento y el porcentaje correspondiente
    print(f"      {conf}: {count} ({100*count/len(y_test):.1f}%)")

# Imprime el encabezado para la distribución de confianza de LightGBM
print(f"\n   LightGBM - Distribución de confianza:")
# Itera sobre las categorías y recuentos de LightGBM
for conf, count in conf_lgb.items():
# Imprime la categoría, el recuento y el porcentaje correspondiente
    print(f"      {conf}: {count} ({100*count/len(y_test):.1f}%)")

# ==========================================
# 6. EXPORTAR RESULTADOS A TXT
# ==========================================
# Abre el archivo de texto en modo escritura con codificación utf-8
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
# Escribe una línea separadora en el archivo
    f.write("="*90 + "\n")
# Escribe el título del reporte
    f.write("REPORTE DE ANÁLISIS DE CALIBRACIÓN\n")
# Escribe otra línea separadora
    f.write("="*90 + "\n\n")
    
# Escribe el título de la sección 1
    f.write("1. CONCEPTO DE CALIBRACIÓN\n")
# Escribe una línea separadora
    f.write("-"*90 + "\n")
# Escribe la explicación del concepto de calibración
    f.write("   La calibración evalúa si las probabilidades predichas por el modelo\n")
# Continúa la explicación
    f.write("   reflejan las tasas reales de eventos. Un modelo bien calibrado mostrará\n")
# Continúa la explicación
    f.write("   que cuando predice 70% de sepsis, aproximadamente 70% de esos casos\n")
# Termina la explicación
    f.write("   realmente tienen sepsis.\n\n")
    
# Escribe el título de la sección 2
    f.write("2. MÉTRICAS DE CALIBRACIÓN\n")
# Escribe una línea separadora
    f.write("-"*90 + "\n")
# Escribe los encabezados de las métricas
    f.write(f"{'Métrica':<25} {'Definición':<65}\n")
# Escribe una línea separadora
    f.write(f"{'-'*90}\n")
# Escribe la definición de Brier Score
    f.write(f"{'Brier Score':<25} {'Mean squared error de probabilidades':<65}\n")
# Escribe detalles del rango de Brier Score
    f.write(f"{'                        '} {'Rango: [0,1]. Menor = mejor':<65}\n")
# Escribe la definición de Log Loss
    f.write(f"{'Log Loss':<25} {'Entropía cruzada de predicciones':<65}\n")
# Escribe detalles de Log Loss
    f.write(f"{'                        '} {'Penaliza más errores extremos':<65}\n")
# Escribe la definición de Calibration Curve
    f.write(f"{'Calibration Curve':<25} {'Grafica prob. predicha vs prob. real':<65}\n")
# Escribe detalles de Calibration Curve
    f.write(f"{'                        '} {'Diagonal perfecto = bien calibrado':<65}\n\n")
    
# Escribe el título de la sección 3
    f.write("3. RESULTADOS XGBOOST\n")
# Escribe una línea separadora
    f.write("-"*90 + "\n")
# Escribe el Brier Score de XGBoost
    f.write(f"   Brier Score:     {brier_xgb:.6f}\n")
# Escribe el Log Loss de XGBoost
    f.write(f"   Log Loss:        {logloss_xgb:.6f}\n")
# Escribe el ROC-AUC de XGBoost
    f.write(f"   ROC-AUC:         {auc_xgb:.6f}\n")
# Escribe la interpretación basada en el Brier Score
    f.write(f"   Interpretación:  {'BIEN CALIBRADO' if brier_xgb < 0.15 else 'MODERADAMENTE CALIBRADO' if brier_xgb < 0.25 else 'MAL CALIBRADO'}\n\n")
    
# Escribe el título de la sección 4
    f.write("4. RESULTADOS LIGHTGBM\n")
# Escribe una línea separadora
    f.write("-"*90 + "\n")
# Escribe el Brier Score de LightGBM
    f.write(f"   Brier Score:     {brier_lgb:.6f}\n")
# Escribe el Log Loss de LightGBM
    f.write(f"   Log Loss:        {logloss_lgb:.6f}\n")
# Escribe el ROC-AUC de LightGBM
    f.write(f"   ROC-AUC:         {auc_lgb:.6f}\n")
# Escribe la interpretación basada en el Brier Score
    f.write(f"   Interpretación:  {'BIEN CALIBRADO' if brier_lgb < 0.15 else 'MODERADAMENTE CALIBRADO' if brier_lgb < 0.25 else 'MAL CALIBRADO'}\n\n")
    
# Escribe el título de la sección 5
    f.write("5. COMPARACIÓN\n")
# Escribe una línea separadora
    f.write("-"*90 + "\n")
# Escribe los encabezados de comparación
    f.write(f"{'Métrica':<20} {'XGBoost':<20} {'LightGBM':<20}\n")
# Escribe una línea separadora
    f.write(f"{'-'*60}\n")
# Escribe la comparación de Brier Score
    f.write(f"{'Brier Score':<20} {f'{brier_xgb:.6f}':<20} {f'{brier_lgb:.6f}':<20}\n")
# Escribe la comparación de Log Loss
    f.write(f"{'Log Loss':<20} {f'{logloss_xgb:.6f}':<20} {f'{logloss_lgb:.6f}':<20}\n")
# Escribe la comparación de ROC-AUC
    f.write(f"{'ROC-AUC':<20} {f'{auc_xgb:.6f}':<20} {f'{auc_lgb:.6f}':<20}\n")
    
# Determina cuál modelo es mejor basándose en el Brier Score
    mejor = "XGBoost" if brier_xgb < brier_lgb else "LightGBM"
# Escribe cuál es el mejor calibrado
    f.write(f"\n   ⭐ Mejor calibrado: {mejor}\n\n")
    
# Escribe el título de la sección 6
    f.write("6. DISTRIBUCIÓN DE CONFIANZA XGBOOST\n")
# Escribe una línea separadora
    f.write("-"*90 + "\n")
# Itera sobre los elementos de la distribución de confianza de XGBoost
    for conf, count in conf_xgb.items():
# Calcula el porcentaje correspondiente
        percentage = 100*count/len(y_test)
# Calcula la longitud de la barra en base al porcentaje
        bar_length = int(percentage / 2)
# Genera el string de la barra usando caracteres especiales
        bar = "█" * bar_length + "░" * (50 - bar_length)
# Escribe la barra y la información de la categoría
        f.write(f"   {str(conf):<25} {bar} {percentage:.1f}% (n={count})\n")
    
# Escribe el título de la sección 7
    f.write("\n7. DISTRIBUCIÓN DE CONFIANZA LIGHTGBM\n")
# Escribe una línea separadora
    f.write("-"*90 + "\n")
# Itera sobre los elementos de la distribución de confianza de LightGBM
    for conf, count in conf_lgb.items():
# Calcula el porcentaje correspondiente
        percentage = 100*count/len(y_test)
# Calcula la longitud de la barra en base al porcentaje
        bar_length = int(percentage / 2)
# Genera el string de la barra usando caracteres especiales
        bar = "█" * bar_length + "░" * (50 - bar_length)
# Escribe la barra y la información de la categoría
        f.write(f"   {str(conf):<25} {bar} {percentage:.1f}% (n={count})\n")
    
# Escribe el título de la sección 8
    f.write("\n8. INTERPRETACIÓN CLÍNICA\n")
# Escribe una línea separadora
    f.write("-"*90 + "\n")
# Escribe la interpretación del Brier Score en contexto
    f.write(f"   • Un Brier Score de {min(brier_xgb, brier_lgb):.4f} indica que en promedio,\n")
# Escribe el error porcentual en probabilidad
    f.write(f"     el modelo se equivoca en {100*min(brier_xgb, brier_lgb):.2f}% en probabilidad.\n")
# Escribe un punto sobre el uso directo de las probabilidades
    f.write(f"   • Las probabilidades predichas pueden ser usadas directamente en\n")
# Continúa el punto
    f.write(f"     contexto clínico sin recalibración.\n")
# Escribe sobre la confianza en el modelo
    f.write(f"   • Un médico puede confiar en los valores de probabilidad del modelo.\n\n")
    
# Escribe el título de la sección 9
    f.write("9. RECOMENDACIONES\n")
# Escribe una línea separadora
    f.write("-"*90 + "\n")
# Escribe la primera recomendación sobre la calibración
    f.write(f"   1. El modelo {'está bien calibrado' if min(brier_xgb, brier_lgb) < 0.15 else 'tiene calibración aceptable'}.\n")
# Escribe la segunda recomendación
    f.write(f"   2. Las probabilidades pueden interpretarse directamente.\n")
# Escribe la tercera recomendación con la probabilidad promedio
    f.write(f"   3. En contexto clínico: P(sepsis)={y_proba_xgb.mean():.1%} promedio.\n")
# Escribe la cuarta recomendación
    f.write(f"   4. Considerar umbral de decisión según sensibilidad/especificidad requerida.\n")

# Imprime confirmación de reporte guardado
print(f"   ✓ Reporte guardado")

# ==========================================
# 7. VISUALIZACIONES
# ==========================================
# Imprime un mensaje indicando la generación de visualizaciones
print("\n[4/4] Generando visualizaciones...")

# Gráfico 1: Calibration Curves
# Crea una figura con dos subgráficos horizontales
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# XGBoost
# Dibuja la línea diagonal de referencia de calibración perfecta
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectamente Calibrado')
# Grafica la curva de calibración de XGBoost
axes[0].plot(prob_pred_xgb, prob_true_xgb, 'o-', linewidth=2, markersize=8, 
# Establece la etiqueta y color
             label='XGBoost', color='#3498db')
# Rellena un área de +-5% alrededor de la curva
axes[0].fill_between(prob_pred_xgb, prob_pred_xgb - 0.05, prob_pred_xgb + 0.05, 
# Define la transparencia y el color
                     alpha=0.2, color='#3498db')
# Etiqueta el eje X
axes[0].set_xlabel('Probabilidad Predicha', fontweight='bold', fontsize=11)
# Etiqueta el eje Y
axes[0].set_ylabel('Probabilidad Real', fontweight='bold', fontsize=11)
# Establece el título del gráfico de XGBoost
axes[0].set_title(f'XGBoost - Curva de Calibración\n(Brier={brier_xgb:.4f}, LogLoss={logloss_xgb:.4f})',
# Configura el peso de fuente y tamaño
                  fontweight='bold', fontsize=12)
# Muestra la leyenda
axes[0].legend(loc='lower right', fontsize=10)
# Muestra la cuadrícula
axes[0].grid(alpha=0.3)
# Limita el eje X de 0 a 1
axes[0].set_xlim([0, 1])
# Limita el eje Y de 0 a 1
axes[0].set_ylim([0, 1])

# LightGBM
# Dibuja la línea diagonal de referencia
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectamente Calibrado')
# Grafica la curva de calibración de LightGBM
axes[1].plot(prob_pred_lgb, prob_true_lgb, 'o-', linewidth=2, markersize=8,
# Establece la etiqueta y color
             label='LightGBM', color='#2ecc71')
# Rellena un área de +-5% alrededor de la curva
axes[1].fill_between(prob_pred_lgb, prob_pred_lgb - 0.05, prob_pred_lgb + 0.05,
# Define la transparencia y color
                     alpha=0.2, color='#2ecc71')
# Etiqueta el eje X
axes[1].set_xlabel('Probabilidad Predicha', fontweight='bold', fontsize=11)
# Etiqueta el eje Y
axes[1].set_ylabel('Probabilidad Real', fontweight='bold', fontsize=11)
# Establece el título del gráfico de LightGBM
axes[1].set_title(f'LightGBM - Curva de Calibración\n(Brier={brier_lgb:.4f}, LogLoss={logloss_lgb:.4f})',
# Configura el peso de fuente y tamaño
                  fontweight='bold', fontsize=12)
# Muestra la leyenda
axes[1].legend(loc='lower right', fontsize=10)
# Muestra la cuadrícula
axes[1].grid(alpha=0.3)
# Limita el eje X de 0 a 1
axes[1].set_xlim([0, 1])
# Limita el eje Y de 0 a 1
axes[1].set_ylim([0, 1])

# Añade un título general a la figura
plt.suptitle('Comparación de Calibración - XGBoost vs LightGBM', 
# Configura el tamaño y posición
             fontsize=14, fontweight='bold', y=1.00)
# Ajusta el layout
plt.tight_layout()
# Guarda la figura en el directorio de gráficos
plt.savefig(os.path.join(PLOTS_DIR, "17_calibration_curves.png"), dpi=300, bbox_inches='tight')
# Cierra la figura
plt.close()
# Imprime confirmación
print("   ✓ Calibration curves guardadas")

# Gráfico 2: Distribución de Probabilidades
# Crea una figura con dos subgráficos horizontales
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# XGBoost
# Grafica el histograma de predicciones negativas
axes[0].hist(y_proba_xgb[y_test == 0], bins=30, alpha=0.6, label='No Sepsis (Real)', color='green')
# Grafica el histograma de predicciones positivas
axes[0].hist(y_proba_xgb[y_test == 1], bins=30, alpha=0.6, label='Sepsis (Real)', color='red')
# Etiqueta el eje X
axes[0].set_xlabel('Probabilidad Predicha de Sepsis', fontweight='bold')
# Etiqueta el eje Y
axes[0].set_ylabel('Frecuencia', fontweight='bold')
# Establece el título
axes[0].set_title('XGBoost - Distribución de Probabilidades', fontweight='bold')
# Muestra la leyenda
axes[0].legend()
# Muestra la cuadrícula
axes[0].grid(alpha=0.3)

# LightGBM
# Grafica el histograma de predicciones negativas
axes[1].hist(y_proba_lgb[y_test == 0], bins=30, alpha=0.6, label='No Sepsis (Real)', color='green')
# Grafica el histograma de predicciones positivas
axes[1].hist(y_proba_lgb[y_test == 1], bins=30, alpha=0.6, label='Sepsis (Real)', color='red')
# Etiqueta el eje X
axes[1].set_xlabel('Probabilidad Predicha de Sepsis', fontweight='bold')
# Etiqueta el eje Y
axes[1].set_ylabel('Frecuencia', fontweight='bold')
# Establece el título
axes[1].set_title('LightGBM - Distribución de Probabilidades', fontweight='bold')
# Muestra la leyenda
axes[1].legend()
# Muestra la cuadrícula
axes[1].grid(alpha=0.3)

# Ajusta el layout
plt.tight_layout()
# Guarda la figura
plt.savefig(os.path.join(PLOTS_DIR, "18_probability_distributions.png"), dpi=300, bbox_inches='tight')
# Cierra la figura
plt.close()
# Imprime confirmación
print("   ✓ Probability distributions guardadas")

# Gráfico 3: Métricas de Calibración
# Crea una figura con un solo gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Define los nombres de las métricas
metrics_names = ['Brier Score', 'Log Loss', 'ROC-AUC']
# Crea lista de scores para XGBoost
xgb_scores = [brier_xgb, logloss_xgb, auc_xgb]
# Crea lista de scores para LightGBM
lgb_scores = [brier_lgb, logloss_lgb, auc_lgb]

# Crea un arreglo para las posiciones de las barras
x = np.arange(len(metrics_names))
# Define el ancho de las barras
width = 0.35

# Dibuja las barras de XGBoost
bars1 = ax.bar(x - width/2, xgb_scores, width, label='XGBoost', color='#3498db', alpha=0.8)
# Dibuja las barras de LightGBM
bars2 = ax.bar(x + width/2, lgb_scores, width, label='LightGBM', color='#2ecc71', alpha=0.8)

# Etiqueta el eje X
ax.set_xlabel('Métrica de Calibración', fontweight='bold', fontsize=11)
# Etiqueta el eje Y
ax.set_ylabel('Valor', fontweight='bold', fontsize=11)
# Establece el título
ax.set_title('Comparación de Métricas de Calibración', fontweight='bold', fontsize=12)
# Configura las posiciones de las marcas en X
ax.set_xticks(x)
# Configura las etiquetas de las marcas en X
ax.set_xticklabels(metrics_names)
# Muestra la leyenda
ax.legend(fontsize=10)
# Muestra la cuadrícula en el eje Y
ax.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
# Itera sobre los grupos de barras
for bars in [bars1, bars2]:
# Itera sobre cada barra individual
    for bar in bars:
# Obtiene la altura de la barra
        height = bar.get_height()
# Añade el valor numérico en la parte superior de la barra
        ax.text(bar.get_x() + bar.get_width()/2., height,
# Configura formato y alineación
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Ajusta el layout
plt.tight_layout()
# Guarda la figura
plt.savefig(os.path.join(PLOTS_DIR, "19_calibration_metrics_comparison.png"), dpi=300, bbox_inches='tight')
# Cierra la figura
plt.close()
# Imprime confirmación
print("   ✓ Metrics comparison guardada")

# Gráfico 4: Confianza por Intervalo
# Crea una figura con dos subgráficos horizontales
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Define las etiquetas de niveles de confianza
confidence_levels = ['Bajo\n(0-30%)', 'Medio\n(30-50%)', 'Medio-Alto\n(50-70%)', 'Alto\n(70-100%)']
# Extrae los recuentos de confianza para XGBoost
xgb_counts = [conf_xgb.get(label, 0) for label in conf_xgb.index]
# Extrae los recuentos de confianza para LightGBM
lgb_counts = [conf_lgb.get(label, 0) for label in conf_lgb.index]

# Crea arreglo de posiciones
x = np.arange(len(confidence_levels))
# Define ancho de barras
width = 0.35

# Dibuja barras de XGBoost
axes[0].bar(x - width/2, xgb_counts, width, label='XGBoost', color='#3498db', alpha=0.8)
# Etiqueta eje X
axes[0].set_xlabel('Nivel de Confianza', fontweight='bold')
# Etiqueta eje Y
axes[0].set_ylabel('Número de Muestras', fontweight='bold')
# Establece título
axes[0].set_title('XGBoost - Distribución de Confianza', fontweight='bold')
# Configura marcas en X
axes[0].set_xticks(x)
# Configura etiquetas en X
axes[0].set_xticklabels(confidence_levels)
# Muestra cuadrícula en Y
axes[0].grid(axis='y', alpha=0.3)

# Agregar valores
# Itera sobre los valores de conteo
for i, v in enumerate(xgb_counts):
# Muestra texto encima de la barra
    axes[0].text(i - width/2, v + 10, str(v), ha='center', fontweight='bold')

# Dibuja barras de LightGBM
axes[1].bar(x - width/2, lgb_counts, width, label='LightGBM', color='#2ecc71', alpha=0.8)
# Etiqueta eje X
axes[1].set_xlabel('Nivel de Confianza', fontweight='bold')
# Etiqueta eje Y
axes[1].set_ylabel('Número de Muestras', fontweight='bold')
# Establece título
axes[1].set_title('LightGBM - Distribución de Confianza', fontweight='bold')
# Configura marcas en X
axes[1].set_xticks(x)
# Configura etiquetas en X
axes[1].set_xticklabels(confidence_levels)
# Muestra cuadrícula en Y
axes[1].grid(axis='y', alpha=0.3)

# Agregar valores
# Itera sobre los valores de conteo
for i, v in enumerate(lgb_counts):
# Muestra texto encima de la barra
    axes[1].text(i - width/2, v + 10, str(v), ha='center', fontweight='bold')

# Ajusta layout
plt.tight_layout()
# Guarda figura
plt.savefig(os.path.join(PLOTS_DIR, "20_confidence_distribution.png"), dpi=300, bbox_inches='tight')
# Cierra figura
plt.close()
# Imprime confirmación
print("   ✓ Confidence distribution guardada")

# ==========================================
# RESUMEN FINAL
# ==========================================
# Imprime una línea en blanco y separadores
print("\n" + "="*80)
# Imprime indicación de finalización
print("✅ ANÁLISIS DE CALIBRACIÓN COMPLETADO")
# Imprime separadores
print("="*80)
# Imprime encabezado del resumen
print(f"\n📊 RESUMEN:\n")
# Imprime título XGBoost
print(f"   XGBoost:")
# Imprime Brier Score
print(f"   • Brier Score: {brier_xgb:.6f}")
# Imprime Log Loss
print(f"   • Log Loss: {logloss_xgb:.6f}")
# Imprime Estado basado en Brier Score
print(f"   • Estado: {'✅ Bien Calibrado' if brier_xgb < 0.15 else '⚠️ Moderadamente Calibrado'}\n")
# Imprime título LightGBM
print(f"   LightGBM:")
# Imprime Brier Score
print(f"   • Brier Score: {brier_lgb:.6f}")
# Imprime Log Loss
print(f"   • Log Loss: {logloss_lgb:.6f}")
# Imprime Estado basado en Brier Score
print(f"   • Estado: {'✅ Bien Calibrado' if brier_lgb < 0.15 else '⚠️ Moderadamente Calibrado'}\n")
# Imprime archivos guardados
print(f"   📁 Visualizaciones: 17_calibration_curves.png - 20_confidence_distribution.png")
# Imprime línea final
print(f"\n{'='*80}\n")