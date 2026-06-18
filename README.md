# Predicción Temprana de Sepsis en UCI usando Machine Learning

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Completo-brightgreen)

## 📋 Descripción del Proyecto

Sistema de machine learning para **predicción temprana de sepsis en unidades de cuidados intensivos (UCI)** utilizando técnicas de gradient boosting (XGBoost y LightGBM) con validación cruzada estratificada, optimización de hiperparámetros mediante GridSearchCV, e interpretabilidad clínica mediante SHAP.

### Objetivo Principal

Desarrollar un modelo de ML con:
- ✅ **Máxima sensibilidad** (detectar todos los casos de sepsis)
- ✅ **Alta especificidad** (minimizar falsas alarmas)
- ✅ **Validación rigurosa** (validación cruzada 5-fold)
- ✅ **Explicabilidad** (SHAP para interpretación clínica)
- ✅ **Calibración** (probabilidades confiables)

---

## 🎯 Resultados Principales

| Métrica | XGBoost | LightGBM |
|---------|---------|----------|
| **ROC-AUC (Test)** | 1.0000 | 1.0000 |
| **Sensitivity (Recall)** | 100% | 100% |
| **Specificity** | 99.15% | 99.53% |
| **Accuracy** | 99.28% | 99.60% |
| **Brier Score** | 0.0057 | 0.0029 ✅ |
| **False Negatives (Test)** | 0 | 0 |

### Hallazgos Clave

1. **Lactato** es el predictor dominante (89.8% de importancia)
2. **Sensibilidad perfecta** (0 casos de sepsis perdidos)
3. **Calibración excelente** (Brier Score < 0.003)
4. **Generalización comprobada** (CV AUC = 1.0000 ± 0.0000)

---

## 📁 Estructura del Proyecto

```
repositorio-TI24-Salvatierra/
│
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
│
├── data/
│   ├── raw/
│   │   └── sepsis_icu_synthetic.csv   # Dataset original (5,000 pacientes)
│   └── processed/
│       ├── X_train.csv                # Features de entrenamiento (3,750)
│       ├── X_test.csv                 # Features de prueba (1,250)
│       ├── y_train.csv                # Target de entrenamiento
│       └── y_test.csv                 # Target de prueba
│
├── notebooks/
│   ├── 01_preview_dataset.py
│   ├── 02_eda_visualizations.py
│   ├── 03_preprocessing.py
│   ├── 04_model_main_MEJORADO.py
│   ├── 05_model_comparison.py
│   ├── 06_hyperparameter_tuning.py
│   ├── 07_model_interpretability_shap_FIXED.py
│   ├── 08_calibration_analysis.py
│   └── run_all_analysis_v2.py
│
├── results/
│   ├── 01_preview_results.txt
│   ├── 02_eda_results.txt
│   ├── 03_preprocessing_results.txt
│   ├── 04_model_main_MEJORADO.txt
│   ├── 05_model_comparison_results.txt
│   ├── 06_hyperparameter_tuning_results.txt
│   ├── 07_shap_interpretability.txt
│   ├── 08_calibration_analysis.txt
│   ├── shap_feature_importance.csv
│   ├── shap_values_class1.npy
│   └── pipeline_execution.log
│
└── plots/
    ├── 01_distribucion_sepsis.png
    ├── 02_edad_vs_sepsis.png
    ├── 03_matriz_correlacion.png
    ├── 04_comparacion_curvas_roc.png
    ├── 05_metricas_por_fold.png
    ├── 06_matriz_confusion_test_mejorada.png
    ├── 07_curva_roc_mejorada.png
    ├── 08_feature_importance_top15.png
    ├── 09_comparacion_cv_vs_test.png
    ├── 10_gridsearch_top12_comparacion.png
    ├── 11_shap_summary_bar.png
    ├── 12_shap_dependence_top2.png
    ├── 13_shap_examples.png
    ├── 14_shap_distributions.png
    ├── 15_calibration_curves.png
    ├── 16_probability_distributions.png
    ├── 17_calibration_metrics_comparison.png
    └── 18_confidence_distribution.png
```

---

## 🔧 Requisitos

### Sistema Operativo
- Windows 10+, Linux (Ubuntu 18.04+), macOS 10.14+

### Python
- **Python 3.8+** (recomendado: 3.10+)
- **pip 21.0+**

### Recursos
- **RAM**: Mínimo 4 GB (recomendado: 8 GB)
- **Espacio disco**: 2-3 GB
- **Tiempo**: ~30-35 minutos (pipeline completo)

---

## 📦 Instalación

### 1. Clonar repositorio
```bash
git clone https://github.com/usuario/repositorio-TI24-Salvatierra.git
cd repositorio-TI24-Salvatierra
```

### 2. Crear ambiente virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar
```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, shap; print('✅ OK')"
```

---

## 🚀 Uso

### Opción 1: Pipeline Completo (RECOMENDADO)
```bash
cd notebooks
python run_all_analysis_v2.py
```

### Opción 2: Scripts Individuales
```bash
cd notebooks
python 01_preview_dataset.py
python 02_eda_visualizations.py
python 03_preprocessing.py
python 04_model_main_MEJORADO.py
python 06_hyperparameter_tuning.py
python 07_model_interpretability_shap_FIXED.py
python 08_calibration_analysis.py
```

### Opción 3: Jupyter Notebook
```bash
jupyter notebook
```

---

## 📊 Dataset

| Característica | Valor |
|---|---|
| **Muestras** | 5,000 pacientes UCI |
| **Variables (originales)** | 77 |
| **Variables (finales)** | 73 |
| **No Sepsis** | 4,250 (85%) |
| **Sepsis** | 750 (15%) |
| **Desbalanceo** | 5.67:1 |

**Variables eliminadas (Data Leakage):** 11
- `sofa_score`, `apache_iv`, `qsofa`, `sirs_criteria`
- `pao2_fio2_ratio` (r = -0.9945)
- `antibiotics_24h`, `fluids_ml_24h`, `vasopressors_flag`
- `mechanical_ventilation`, `subject_id`

---

## 🧠 Modelos

### XGBoost
```python
n_estimators=150
max_depth=5
learning_rate=0.05
scale_pos_weight=5.6607
```

**Test Metrics:**
- ROC-AUC: 1.0000
- Accuracy: 99.28%
- Recall: 100%

### LightGBM
```python
n_estimators=150
max_depth=5
learning_rate=0.05
num_leaves=31
```

**Test Metrics:**
- ROC-AUC: 1.0000
- Accuracy: 99.60%
- Recall: 100%

---

## 🔍 Validación

### Validación Cruzada
- **Método**: StratifiedKFold (5 pliegues)
- **ROC-AUC**: 1.0000 ± 0.0000
- **FN totales**: 1 de 2,815 casos (0.035%)

### GridSearchCV
- **XGBoost**: 54 combinaciones → Mejor AUC: 0.999997
- **LightGBM**: 81 combinaciones → Mejor AUC: 0.999994

---

## 🎯 Interpretabilidad (SHAP)

### Top 5 Features

| Rank | Variable | SHAP Abs | Significado Clínico |
|---|---|---|---|
| 1 | lactate_mmol | 5.6588 | Hipoxia tisular |
| 2 | creatinine | 0.7799 | Fallo renal |
| 3 | respiratory_rate_mean | 0.2456 | Distress respiratorio |
| 4 | ph_arterial | 0.2009 | Acidosis metabólica |
| 5 | respiratory_rate_min | 0.1953 | Variabilidad anormal |

---

## ⚖️ Calibración

| Métrica | XGBoost | LightGBM |
|---|---|---|
| Brier Score | 0.005659 | 0.002932 ✅ |
| Log Loss | 0.019022 | 0.010012 ✅ |
| ROC-AUC | 0.999975 | 0.999995 |

**Interpretación**: Error promedio < 0.3% en probabilidades

---

## 📈 18 Gráficos Generados

**EDA (3):** Distribución, edad, correlación
**Modelos (4):** ROC, métricas, confusión, características
**GridSearchCV (1):** Top 12 combinaciones
**SHAP (4):** Summary, dependence, ejemplos, distribuciones
**Calibración (4):** Curvas, probabilidades, métricas, confianza
**Comparación (2):** CV vs Test, XGB vs LGB

---

## 🎓 Documentos Académicos

### Informe LaTeX
Archivo: `Informe_Final.txt`
- ✅ 27 páginas profesionales
- ✅ Portada + resumen
- ✅ Metodología completa
- ✅ Resultados con tablas
- ✅ 5 apéndices (A-E)
- ✅ Guía de presentación oral

**Uso:**
1. Descarga archivo TXT
2. Copia TODO (Ctrl+A)
3. Ve a overleaf.com
4. Pega en main.tex
5. Recompile

---

## 📝 Reportes (9 archivos TXT)

1. Vista previa del dataset
2. Análisis exploratorio (EDA)
3. Preprocesamiento
4. Modelo XGBoost + CV
5. Comparación modelos
6. GridSearchCV
7. SHAP
8. Calibración
9. Pipeline log

---

## ⚠️ Limitaciones

1. Dataset sintético (no variabilidad clínica real)
2. Sin validación externa (requiere hospital real)
3. Sin temporalidad (secuencia de síntomas)
4. Variables proxy sintéticas

---

## 🔮 Trabajo Futuro

1. Incorporar datos temporales (LSTM, Transformers)
2. Validación externa en hospital real
3. Análisis por subgrupos clínicos
4. Análisis de interacciones
5. Análisis costo-beneficio

---

## 👨‍💼 Autor

**Marco Antonio Salvatierra Copa**
- Carrera: Ingeniería en Data Science y BI
- Universidad: Universidad del Valle
- Email: marco.salvatierra@univalle.edu.bo
- Instructor: Ing. Efraín F. Luna
- Fecha: Junio 18, 2026

---

## 📚 Referencias Clave

1. Singer et al. (2016). Sepsis-3 Definitions. JAMA 315(8), 801-810
2. Lundberg & Lee (2017). SHAP: Unified Approach to Interpreting Predictions
3. Chen & Guestrin (2016). XGBoost: Scalable Tree Boosting System
4. Ke et al. (2017). LightGBM: Fast Gradient Boosting Framework
5. Johnson et al. (2016). MIMIC-III Database. Scientific Data 3

---

## 📄 Licencia

MIT License - Ver archivo LICENSE

---

## ❓ FAQ

**P: ¿Por qué ROC-AUC = 1.0?**
R: Dataset sintético con variables muy predictivas. CV también da 1.0000.

**P: ¿Puedo usar en hospital real?**
R: Necesita validación externa primero.

**P: ¿Tiempo de ejecución?**
R: ~31.5 minutos (pipeline completo)

---

**Estado: COMPLETO ✅**

Última actualización: Junio 18, 2026