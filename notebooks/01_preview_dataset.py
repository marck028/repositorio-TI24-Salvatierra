import os
import pandas as pd

# 1. Definir rutas locales
DATA_PATH = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\data\raw\sepsis_icu_synthetic.csv"
OUTPUT_DIR = r"D:\Documentos\GitHub\repositorio-TI24-Salvatierra\results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "01_preview_results.txt")

# Asegurar que la carpeta de resultados exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Cargando el dataset...")
try:
    # 2. Cargar los datos
    df = pd.read_csv(DATA_PATH)
    print(r"¡Dataset cargado con éxito desde D:\...\data\raw!")
    
    # 3. Recopilar la información para el archivo .txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("      REPORTE DE VISTA PREVIA DEL DATASET\n")
        f.write("==================================================\n\n")
        
        # Dimensiones
        f.write(f"Dimensiones del dataset:\n")
        f.write(f"- Filas: {df.shape[0]}\n")
        f.write(f"- Columnas: {df.shape[1]}\n\n")
        
        f.write("--------------------------------------------------\n")
        f.write("1. ESTRUCTURA Y TIPOS DE DATOS POR COLUMNA\n")
        f.write("--------------------------------------------------\n")
        # Construir tabla de tipos de datos y nulos
        info_df = pd.DataFrame({
            'Tipo de Dato': df.dtypes,
            'Valores No Nulos': df.notnull().sum(),
            'Valores Nulos': df.isnull().sum(),
            '% Nulos': (df.isnull().sum() / len(df) * 100).round(2)
        })
        f.write(info_df.to_string())
        f.write("\n\n")
        
        f.write("--------------------------------------------------\n")
        f.write("2. VISTA PREVIA DE LOS PRIMEROS 5 REGISTROS\n")
        f.write("--------------------------------------------------\n")
        # Guardar las primeras filas de manera transpuesta para que sea legible si hay muchas columnas
        f.write(df.head().to_string())
        f.write("\n")
        
    print(f"✅ El reporte ha sido generado exitosamente en:\n{OUTPUT_FILE}")

except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo en la ruta especificada:\n{DATA_PATH}")
    print("Por favor, asegúrate de haber descargado el CSV de Kaggle y moverlo a esa carpeta con ese nombre exacto.")
except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")