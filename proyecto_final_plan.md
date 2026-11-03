# Planificación — Proyecto Final: Analítica de Datos

## Objetivo del proyecto
Construir una página web que permita usar **Regresión Logística** y una **Red Neuronal Artificial** para hacer predicciones. La página debe tener dos opciones:

1. **Predicción Individual**: muestra la predicción de un solo individuo.
2. **Predicción por lote**: muestra métricas, incluyendo la **matriz de confusión**.

Se permite el uso de IA en el desarrollo.

---

## Stack tecnológico recomendado

| Capa | Tecnología | Justificación |
|------|-----------|---------------|
| Lenguaje | Python | Estándar en analítica de datos y machine learning |
| Interfaz web | **Streamlit** | Permite crear la página web en puro Python, sin HTML/CSS/JS. Ideal para proyectos de data science |
| Alternativa web | Flask + HTML/CSS | Si el profesor exige una página web más tradicional |
| Modelos | `scikit-learn` | `LogisticRegression` y `MLPClassifier` (red neuronal) vienen listos |
| Manipulación de datos | `pandas`, `numpy` | Carga, limpieza y preparación del dataset |
| Guardado de modelos | `joblib` o `pickle` | Entrenar una vez y cargar el modelo ya entrenado en la web |
| Métricas | `sklearn.metrics` | `confusion_matrix`, `classification_report`, `accuracy_score`, etc. |

---

## Estructura propuesta del proyecto

```
proyecto_final/
├── data/                 # Dataset (CSV, Excel, etc.)
├── models/               # Modelos ya entrenados (.pkl o .joblib)
├── notebooks/            # Jupyter Notebook donde se entrenan y guardan los modelos
├── app.py                # Aplicación Streamlit (la "página web")
└── requirements.txt      # Librerías a instalar
```

---

## Pasos para ejecutar el proyecto

### Paso A: Elegir un dataset de clasificación binaria
Ejemplos públicos y sencillos:
- Titanic (supervivencia: sí/no)
- Diabetes Pima Indians (diabético: sí/no)
- Cáncer de mama (maligno/benigno)
- Aprobación de crédito (aprobado/rechazado)

### Paso B: Entrenar y guardar los modelos (Jupyter Notebook)
1. Cargar los datos con pandas.
2. Limpiar valores nulos y codificar variables categóricas.
3. Dividir en conjunto de entrenamiento y prueba.
4. Entrenar:
   - Regresión Logística: `LogisticRegression()`
   - Red Neuronal Artificial: `MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)`
5. Evaluar con matriz de confusión y reporte de clasificación.
6. Guardar ambos modelos con `joblib.dump()`.

### Paso C: Crear la aplicación web (Streamlit)
La app debe tener un menú con dos opciones:

**Opción 1 — Predicción Individual**
- Formulario donde el usuario ingrese los valores de un solo registro.
- Al presionar "Predecir", cargar el modelo seleccionado y mostrar:
  - Clase predicha (0 o 1).
  - Opcionalmente, la probabilidad de la predicción.

**Opción 2 — Predicción por lote**
- El usuario sube un archivo CSV/Excel con varios registros.
- La app predice todos los registros y muestra:
  - Métricas (accuracy, precisión, recall, F1-score).
  - Matriz de confusión (tabla o imagen).
  - Tabla con las predicciones fila por fila.

### Paso D: Despliegue (opcional, valorado)
- Subir la app a **Streamlit Community Cloud** (gratis) o **Render**.
- Entregar el link público de la aplicación.

---

## Ejemplo base de código (Streamlit)

```python
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

st.title("Proyecto Final - Predicción con ML")

opcion = st.sidebar.selectbox("Tipo de predicción", ["Individual", "Por lote"])
modelo_nombre = st.sidebar.selectbox("Modelo", ["Regresión Logística", "Red Neuronal"])

# Cargar modelo según elección
if modelo_nombre == "Regresión Logística":
    clf = joblib.load("models/logistica.pkl")
else:
    clf = joblib.load("models/red_neuronal.pkl")

if opcion == "Individual":
    # Formulario de inputs del usuario
    # Predecir un solo registro
    pass

else:
    archivo = st.file_uploader("Sube tu CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        predicciones = clf.predict(df)
        st.write("### Matriz de confusión")
        st.write(confusion_matrix(y_real, predicciones))  # y_real debe venir en el CSV
        st.write("### Reporte de clasificación")
        st.text(classification_report(y_real, predicciones))
```

---

## Dudas a definir antes de empezar

1. ¿Se permite usar Streamlit o se exige HTML/CSS/JavaScript?
2. ¿Ya hay un dataset asignado por el profesor o se elige uno libremente?
3. ¿Se tiene experiencia previa con Python y scikit-learn?

---