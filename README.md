# 🔬 Breast Cancer Classification — Challenge Intermedio

[![Status](https://img.shields.io/badge/Status-Completado-brightgreen)](https://github.com/HectorDelgado9997/challenge_intermedio)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

## 📌 Descripción

Sistema de clasificación binaria que predice si un tumor de mama es **maligno o benigno**
a partir de mediciones celulares, utilizando el dataset
**Breast Cancer Wisconsin Diagnostic** de UCI.

El proyecto implementa un pipeline completo de Machine Learning con seguimiento
de experimentos mediante MLflow, pruebas automatizadas con pytest, y una
arquitectura modular orientada a buenas prácticas de MLOps.

---

## 🎯 Objetivo

Entrenar, evaluar y comparar tres modelos de clasificación — Logistic Regression,
KNN y Decision Tree — sobre el dataset de cáncer de mama, registrando
automáticamente todos los experimentos con MLflow.

---

## 📁 Estructura del Repositorio

```text
challenge_intermedio/
├── data/
│   └── breast_cancer_wisconsin.csv   # Dataset fuente
├── docs/
│   ├── dataset_extraction.md         # Descripción del dataset
│   ├── model_construction.md         # Construcción y evaluación de modelos
│   ├── mlops_setup.md                # Configuración de MLflow
│   ├── technical_run_guide.md        # Guía de ejecución paso a paso
│   └── architecture.md               # Arquitectura del proyecto
├── notebooks/                        # Análisis exploratorio
├── outputs/                          # Gráficas y métricas generadas
├── src/
│   ├── config/                       # Configuración global
│   ├── data/                         # Carga y validación de datos
│   ├── mlops/                        # Tracking con MLflow
│   ├── models/                       # Entrenamiento y evaluación
│   ├── preprocessing/                # Preprocesamiento
│   └── utils/                        # Logger
├── tests/                            # Suite de pruebas pytest
├── requirements.txt
├── pytest.ini
└── run_pipeline.py                   # Punto de entrada del pipeline
```

---

## 🤖 Modelos Entrenados

| Modelo               | Tipo              | Librería     |
|----------------------|-------------------|--------------|
| Logistic Regression  | Lineal            | scikit-learn |
| K-Nearest Neighbors  | Basado en instancias | scikit-learn |
| Decision Tree        | No lineal         | scikit-learn |

---

## 📊 Dataset

| Propiedad       | Valor                           |
|-----------------|---------------------------------|
| Fuente          | UCI Machine Learning Repository |
| Muestras        | 569                             |
| Features        | 30 numéricas                    |
| Target          | M → 1 (Maligno), B → 0 (Benigno)|
| Valores nulos   | Ninguno                         |

---

## ⚙️ Instalación y Ejecución Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/HectorDelgado9997/challenge_intermedio.git
cd challenge_intermedio

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash
# source .venv/bin/activate       # Linux / Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el pipeline completo
python run_pipeline.py

# 5. Ver experimentos en MLflow
mlflow ui
```

> Para instrucciones detalladas, consulta [`docs/technical_run_guide.md`](docs/technical_run_guide.md)

---

## 🔁 Pipeline Carga de datos → Validación → Encoding → Split
└── Por cada modelo:
Cross-validation (5-fold, F1)
Entrenamiento
Evaluación (Precision, Recall, F1, ROC AUC)
Guardado de gráficas
Logging en MLflow
└── Exportar metrics_summary.csv
---

## 📈 Métricas de Evaluación

| Métrica          | Descripción                                      |
|------------------|--------------------------------------------------|
| F1 Score         | Métrica principal — balancea precision y recall  |
| ROC AUC          | Capacidad discriminativa del modelo              |
| Precision        | Exactitud en las predicciones positivas          |
| Recall           | Cobertura de los casos positivos reales          |
| Confusion Matrix | Desglose TP / FP / TN / FN                       |

---

## 🧪 Tests

```bash
pytest        # Ejecutar todos los tests
pytest -v     # Modo verbose
```

---

## 📚 Documentación

| Archivo                        | Contenido                            |
|--------------------------------|--------------------------------------|
| `docs/dataset_extraction.md`   | Origen, estructura y variables       |
| `docs/model_construction.md`   | Modelos, entrenamiento y métricas    |
| `docs/mlops_setup.md`          | Configuración y uso de MLflow        |
| `docs/technical_run_guide.md`  | Guía completa de ejecución           |
| `docs/architecture.md`         | Arquitectura, capas y flujo de datos |

---

## 🛠️ Stack Tecnológico

| Herramienta    | Uso                        |
|----------------|----------------------------|
| Python 3.9+    | Lenguaje principal         |
| scikit-learn   | Modelos ML                 |
| pandas / numpy | Manipulación de datos      |
| matplotlib     | Visualizaciones            |
| MLflow         | Tracking de experimentos   |
| FastAPI        | Servicio de predicción     |
| pytest         | Pruebas automatizadas      |
| joblib         | Serialización de modelos   |
| python-dotenv  | Variables de entorno       |

---

## 👤 Autor

**Héctor Delgado**
[![GitHub](https://img.shields.io/badge/GitHub-HectorDelgado9997-black)](https://github.com/HectorDelgado9997)
