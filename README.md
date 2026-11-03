# DenguePredict — Proyecto final

Aplicación web en Flask para comparar **regresión logística** y **red neuronal artificial** en la predicción de dengue complicado usando `MASTER_CHART_F1000Research.xlsx`.

## 1. Instalar dependencias

Desde esta carpeta:

```bash
python -m pip install -r requirements.txt
```

## 2. Entrenar los modelos

Abrir `notebooks/entrenamiento_modelos.ipynb` en Jupyter y ejecutar todas las celdas. El notebook:

- Lee el Excel original.
- Elimina un registro duplicado.
- Elimina `FERRITIN` por tener más del 30% de valores ausentes.
- Elimina columnas casi constantes con una frecuencia dominante igual o superior al 95%.
- Imputa valores faltantes mediante la mediana dentro del pipeline.
- Codifica las variables binarias y crea `WBC_RANGE`, `HB_PCV_RATIO`, `SYMPTOM_COUNT` y `TREATMENT_COUNT`.
- Divide los datos de forma estratificada y entrena ambos modelos.
- Guarda `models/model_bundle.joblib`.

También puede ejecutarse desde la terminal:

```bash
jupyter nbconvert --to notebook --execute notebooks/entrenamiento_modelos.ipynb --output entrenamiento_ejecutado.ipynb
```

## 3. Ejecutar la página

```bash
python app.py
```

Abrir `http://127.0.0.1:5000`.

En predicción por lote se aceptan `.csv` y `.xlsx`. Si el archivo contiene la columna `GROUPS`, se calculan accuracy, precisión, recall, F1 y matriz de confusión. Las predicciones se descargan desde la aplicación.

## 4. Despliegue

Subir este proyecto a GitHub incluyendo `models/model_bundle.joblib` y desplegarlo en Render, Railway u otro servicio compatible con Flask. El comando de inicio recomendado es:

```bash
gunicorn app:app
```

Para producción, configurar una variable de entorno para `SECRET_KEY` y no usar `debug=True`.

## Interpretación de la variable objetivo

- `GROUPS = 1`: dengue no complicado.
- `GROUPS = 2`: dengue complicado.

Internamente se modela como `0` y `1`, respectivamente.
