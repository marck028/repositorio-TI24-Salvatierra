# Importa el módulo 'os' para interactuar con el sistema operativo
import os
# Importa la biblioteca 'pandas' y la renombra como 'pd' para análisis y manipulación de datos
import pandas as pd

# 1. Definir rutas locales
# Asigna la ruta del archivo de datos a la constante 'DATA_PATH'
DATA_PATH = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\raw\sepsis_icu_synthetic.csv"
# Asigna la ruta del directorio de salida a la constante 'OUTPUT_DIR'
OUTPUT_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
# Combina el directorio de salida y el nombre del archivo para obtener la ruta completa del archivo de resultados, asignándola a 'OUTPUT_FILE'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "01_preview_results.txt")

# Asegurar que la carpeta de resultados exista
# Crea el directorio de salida si no existe, sin generar error si ya está creado
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Imprime un mensaje en la consola indicando que se está cargando el dataset
print("Cargando el dataset...")
# Inicia un bloque 'try' para manejar posibles excepciones durante la ejecución
try:
    # 2. Cargar los datos
    # Lee el archivo CSV especificado en 'DATA_PATH' y carga su contenido en un DataFrame llamado 'df'
    df = pd.read_csv(DATA_PATH)
    # Imprime un mensaje confirmando que el dataset se cargó correctamente
    print(r"¡Dataset cargado con éxito desde D:\...\data\raw!")
    
    # 3. Recopilar la información para el archivo .txt
    # Abre (o crea) el archivo 'OUTPUT_FILE' en modo escritura ('w') con codificación UTF-8 usando un bloque 'with'
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Escribe una línea de separadores en el archivo
        f.write("==================================================\n")
        # Escribe el título del reporte en el archivo
        f.write("      REPORTE DE VISTA PREVIA DEL DATASET\n")
        # Escribe otra línea de separadores y un salto de línea adicional
        f.write("==================================================\n\n")
        
        # Dimensiones
        # Escribe un encabezado indicando que se mostrarán las dimensiones del dataset
        f.write(f"Dimensiones del dataset:\n")
        # Escribe el número total de filas del DataFrame, obtenido del primer elemento de 'df.shape'
        f.write(f"- Filas: {df.shape[0]}\n")
        # Escribe el número total de columnas del DataFrame, obtenido del segundo elemento de 'df.shape', seguido de un salto de línea adicional
        f.write(f"- Columnas: {df.shape[1]}\n\n")
        
        # Escribe un separador en el archivo
        f.write("--------------------------------------------------\n")
        # Escribe el título de la sección de estructura y tipos de datos
        f.write("1. ESTRUCTURA Y TIPOS DE DATOS POR COLUMNA\n")
        # Escribe otro separador en el archivo
        f.write("--------------------------------------------------\n")
        # Construir tabla de tipos de datos y nulos
        # Crea un nuevo DataFrame llamado 'info_df' usando un diccionario para consolidar información de las columnas
        info_df = pd.DataFrame({
            # Asigna los tipos de datos de cada columna al campo 'Tipo de Dato'
            'Tipo de Dato': df.dtypes,
            # Cuenta y asigna la cantidad de valores no nulos por columna al campo 'Valores No Nulos'
            'Valores No Nulos': df.notnull().sum(),
            # Cuenta y asigna la cantidad de valores nulos por columna al campo 'Valores Nulos'
            'Valores Nulos': df.isnull().sum(),
            # Calcula el porcentaje de valores nulos, lo redondea a dos decimales y lo asigna al campo '% Nulos'
            '% Nulos': (df.isnull().sum() / len(df) * 100).round(2)
        # Cierra la creación del DataFrame 'info_df'
        })
        # Convierte el DataFrame 'info_df' a formato de cadena y lo escribe en el archivo
        f.write(info_df.to_string())
        # Escribe dos saltos de línea adicionales para separar secciones
        f.write("\n\n")
        
        # Escribe un separador en el archivo
        f.write("--------------------------------------------------\n")
        # Escribe el título de la sección de vista previa de registros
        f.write("2. VISTA PREVIA DE LOS PRIMEROS 5 REGISTROS\n")
        # Escribe otro separador en el archivo
        f.write("--------------------------------------------------\n")
        # Guardar las primeras filas de manera transpuesta para que sea legible si hay muchas columnas
        # Escribe en el archivo las primeras 5 filas del DataFrame 'df' convertidas a cadena
        f.write(df.head().to_string())
        # Escribe un salto de línea en el archivo
        f.write("\n")
        
    # Imprime un mensaje indicando que el reporte se generó de forma exitosa, mostrando la ruta del archivo
    print(f"✅ El reporte ha sido generado exitosamente en:\n{OUTPUT_FILE}")

# Captura la excepción FileNotFoundError si el archivo especificado en 'DATA_PATH' no existe
except FileNotFoundError:
    # Imprime un mensaje de error indicando que no se encontró el archivo
    print(f"❌ Error: No se encontró el archivo en la ruta especificada:\n{DATA_PATH}")
    # Imprime un mensaje sugiriendo verificar que el archivo esté en el lugar correcto
    print("Por favor, asegúrate de haber descargado el CSV de Kaggle y moverlo a esa carpeta con ese nombre exacto.")
# Captura cualquier otra excepción no prevista y la guarda en la variable 'e'
except Exception as e:
    # Imprime un mensaje indicando que ocurrió un error inesperado, mostrando el mensaje de la excepción
    print(f"❌ Ocurrió un error inesperado: {e}")