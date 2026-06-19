# Importar el módulo os para interactuar con el sistema operativo
import os
# Importar la librería pandas para manipulación y análisis de datos
import pandas as pd
# Importar el módulo pyplot de matplotlib para crear gráficos
import matplotlib.pyplot as plt
# Importar la librería seaborn para visualización estadística de datos
import seaborn as sns

# 1. Definir rutas locales
# Definir la ruta del archivo CSV que contiene los datos sintéticos de sepsis
DATA_PATH = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\raw\sepsis_icu_synthetic.csv"
# Definir la ruta del directorio donde se guardarán los resultados
RESULTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
# Definir la ruta del directorio donde se guardarán los gráficos
PLOTS_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\plots"
# Crear la ruta completa para el archivo de texto de salida uniendo el directorio y el nombre del archivo
TXT_OUTPUT = os.path.join(RESULTS_DIR, "02_eda_results.txt")

# Asegurar que las carpetas existan
# Crear el directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)
# Crear el directorio de gráficos si no existe
os.makedirs(PLOTS_DIR, exist_ok=True)

# Imprimir un mensaje indicando que se están cargando los datos
print("Cargando datos para el Análisis Exploratorio (EDA)...")
# Cargar los datos del archivo CSV en un DataFrame de pandas
df = pd.read_csv(DATA_PATH)

# ==========================================
# A. CÁLCULOS MATEMÁTICOS Y ESTADÍSTICOS
# ==========================================
# Imprimir un mensaje indicando que se están generando los reportes
print("Generando reportes estadísticos...")
# Abrir el archivo de texto en modo escritura ("w") con codificación UTF-8
with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
    # Escribir una línea separadora en el archivo
    f.write("==================================================\n")
    # Escribir el título del reporte en el archivo
    f.write("        REPORTE DE ANÁLISIS EXPLORATORIO (EDA)\n")
    # Escribir otra línea separadora y saltos de línea
    f.write("==================================================\n\n")
    
    # 1. Balance de la variable objetivo (Sepsis vs No Sepsis)
    # Escribir el subtítulo para la distribución de la variable objetivo
    f.write("1. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (sepsis_label)\n")
    # Escribir una línea separadora
    f.write("--------------------------------------------------\n")
    # Contar la cantidad de ocurrencias para cada clase en la variable 'sepsis_label'
    sepsis_counts = df['sepsis_label'].value_counts()
    # Calcular el porcentaje de cada clase y multiplicarlo por 100
    sepsis_pct = df['sepsis_label'].value_counts(normalize=True) * 100
    # Iterar sobre los valores y conteos obtenidos
    for val, count in sepsis_counts.items():
        # Asignar la etiqueta "Sepsis" si el valor es 1, de lo contrario "No Sepsis"
        label = "Sepsis" if val == 1 else "No Sepsis"
        # Escribir en el archivo la cantidad y el porcentaje formateado a dos decimales
        f.write(f"- {label}: {count} pacientes ({sepsis_pct[val]:.2f}%)\n")
    # Escribir un salto de línea en el archivo
    f.write("\n")

    # 2. Estadísticas descriptivas de variables numéricas clave
    # Escribir el subtítulo para las estadísticas descriptivas
    f.write("2. ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES CLAVE\n")
    # Escribir una línea separadora
    f.write("--------------------------------------------------\n")
    # Definir una lista con las variables numéricas clave a analizar
    features_clave = ['age', 'bmi', 'hr_mean', 'sbp_mean', 'temp_celsius_mean', 'spo2_mean', 'wbc', 'lactate_mmol', 'sofa_score']
    # Calcular las estadísticas descriptivas para las variables clave, convertirlas a cadena y escribirlas
    f.write(df[features_clave].describe().to_string())
    # Escribir dos saltos de línea
    f.write("\n\n")

    # 3. Correlación lineal de Pearson con la Sepsis
    # Escribir el subtítulo para las correlaciones
    f.write("3. TOP CORRELACIONES CON SEPSIS_LABEL\n")
    # Escribir una línea separadora
    f.write("--------------------------------------------------\n")
    # Seleccionar solo columnas numéricas para la correlación
    # Filtrar el DataFrame original seleccionando únicamente columnas de tipo entero y flotante
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    # Calcular la matriz de correlación, seleccionar la columna 'sepsis_label' y ordenar los valores descendentemente
    correlations = numeric_df.corr()['sepsis_label'].sort_values(ascending=False)
    # Escribir el subtítulo para las variables más correlacionadas positivamente
    f.write("Variables más correlacionadas positivamente:\n")
    # Convertir las primeras 6 correlaciones a cadena y escribirlas en el archivo
    f.write(correlations.head(6).to_string())
    # Escribir el subtítulo para las variables más correlacionadas negativamente
    f.write("\n\nVariables más correlacionadas negativamente:\n")
    # Convertir las últimas 5 correlaciones a cadena y escribirlas en el archivo
    f.write(correlations.tail(5).to_string())
    # Escribir un salto de línea en el archivo
    f.write("\n")

# ==========================================
# B. GENERACIÓN DE GRÁFICOS PROFESIONALES
# ==========================================
# Imprimir un mensaje indicando que se están generando los gráficos
print("Generando gráficos de alta calidad...")
# Establecer el tema visual de seaborn con cuadrícula blanca
sns.set_theme(style="whitegrid")

# Gráfico 1: Distribución de la variable objetivo
# Crear una nueva figura de matplotlib con un tamaño específico (ancho 6, alto 4)
plt.figure(figsize=(6, 4))
# Generar un gráfico de conteo usando seaborn para la variable 'sepsis_label' con una paleta de colores
ax = sns.countplot(x='sepsis_label', data=df, hue='sepsis_label', palette="Set2", legend=False)
# Añadir un título al gráfico con tamaño y grosor de fuente especificados
plt.title("Distribución de Casos de Sepsis en UCI", fontsize=12, fontweight='bold')
# Añadir una etiqueta al eje X
plt.xlabel("Estado (0 = No Sepsis, 1 = Sepsis)")
# Añadir una etiqueta al eje Y
plt.ylabel("Número de Pacientes")
# Cambiar las etiquetas del eje X por "No Sepsis" y "Sepsis"
plt.xticks([0, 1], ['No Sepsis', 'Sepsis'])
# Ajustar automáticamente los parámetros de la subtrama para que quepa en el área de la figura
plt.tight_layout()
# Guardar el gráfico generado en el directorio de gráficos con alta resolución (dpi=300)
plt.savefig(os.path.join(PLOTS_DIR, "01_distribucion_sepsis.png"), dpi=300)
# Cerrar la figura actual para liberar memoria
plt.close()

# Gráfico 2: Edad vs Sepsis (Boxplot)
# Crear una nueva figura para el diagrama de caja con tamaño 7x4
plt.figure(figsize=(7, 4))
# Generar un diagrama de caja relacionando la presencia de sepsis con la edad
sns.boxplot(x='sepsis_label', y='age', data=df, hue='sepsis_label', palette="Pastel1", legend=False)
# Añadir un título al gráfico de caja
plt.title("Distribución de Edad según Presencia de Sepsis", fontsize=12, fontweight='bold')
# Añadir una etiqueta al eje X del diagrama de caja
plt.xlabel("Sepsis Label")
# Añadir una etiqueta al eje Y del diagrama de caja
plt.ylabel("Edad (Años)")
# Reemplazar las marcas numéricas del eje X con etiquetas descriptivas
plt.xticks([0, 1], ['No Sepsis', 'Sepsis'])
# Optimizar el diseño general del gráfico de caja
plt.tight_layout()
# Exportar la figura como un archivo PNG
plt.savefig(os.path.join(PLOTS_DIR, "02_edad_vs_sepsis.png"), dpi=300)
# Cerrar la ventana del gráfico
plt.close()

# Gráfico 3: Matriz de Correlación reducida (Heatmap)
# Iniciar una nueva figura para la matriz de correlación (mapa de calor) con dimensiones 8x6
plt.figure(figsize=(8, 6))
# Calcular la matriz de correlación para las variables clínicas clave y la variable objetivo
corr_matrix = df[features_clave + ['sepsis_label']].corr()
# Dibujar el mapa de calor basado en la matriz de correlación calculada, añadiendo anotaciones
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# Colocar un título representativo en la parte superior del mapa de calor
plt.title("Matriz de Correlación - Variables Clínicas Clave", fontsize=12, fontweight='bold')
# Ajustar los márgenes del gráfico de calor para que todo sea visible
plt.tight_layout()
# Guardar el mapa de calor resultante en la carpeta especificada
plt.savefig(os.path.join(PLOTS_DIR, "03_matriz_correlacion.png"), dpi=300)
# Finalizar la renderización y cerrar la figura de correlación
plt.close()

# Imprimir por consola la confirmación de la ruta donde se guardó el reporte
print(f"✅ Reporte guardado en: {TXT_OUTPUT}")
# Imprimir por consola la confirmación del directorio donde se exportaron los gráficos
print(f"✅ Gráficos exportados en: {PLOTS_DIR}")