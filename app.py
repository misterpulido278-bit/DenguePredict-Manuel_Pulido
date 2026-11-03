from __future__ import annotations

import io
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from data_processing import (
    BASE_INPUT_COLUMNS,
    TARGET_COLUMN,
    encode_target,
    prepare_prediction_data,
    standardize_columns,
    target_labels,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_bundle.joblib"
app = Flask(__name__)
app.secret_key = "proyecto-dengue-analitica-2026"


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No existen los modelos. Ejecute el notebook de entrenamiento primero.")
    return joblib.load(MODEL_PATH)


def model_options() -> dict[str, str]:
    return {"logistic": "Regresión logística", "neural": "Red neuronal artificial"}


def selected_model(bundle: dict, key: str):
    if key not in bundle["models"]:
        key = "logistic"
    return bundle["models"][key], model_options()[key]


def read_uploaded_file(file_storage) -> pd.DataFrame:
    filename = (file_storage.filename or "").lower()
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file_storage)
    return pd.read_csv(file_storage)


@app.context_processor
def inject_globals():
    return {"model_options": model_options()}


@app.route("/")
def index():
    bundle = load_bundle()
    return render_template("index.html", summary=bundle.get("summary", {}))


@app.route("/individual", methods=["GET", "POST"])
def individual():
    bundle = load_bundle()
    result = None
    form_values = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            raw = {column: request.form.get(column) for column in BASE_INPUT_COLUMNS}
            # Las columnas descartadas durante la limpieza se incluyen para que
            # la transformación sea compatible, aunque no entren al modelo.
            data = pd.DataFrame([raw])
            features = prepare_prediction_data(
                data,
                bundle["feature_columns"],
                bundle["columns_to_drop"],
            )
            model_key = request.form.get("model", "logistic")
            model, model_name = selected_model(bundle, model_key)
            prediction = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]
            result = {
                "model_name": model_name,
                "label": target_labels([prediction])[0],
                "probability": float(probabilities[prediction]) * 100,
                "complicated_probability": float(probabilities[1]) * 100,
            }
        except (ValueError, TypeError, KeyError) as error:
            flash(f"No se pudo procesar el formulario: {error}", "error")

    return render_template("individual.html", result=result, form_values=form_values)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    bundle = load_bundle()
    result = None
    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            flash("Seleccione un archivo CSV o Excel.", "error")
            return render_template("batch.html")
        try:
            raw_df = read_uploaded_file(uploaded)
            standardized = standardize_columns(raw_df)
            model_key = request.form.get("model", "logistic")
            model, model_name = selected_model(bundle, model_key)
            features = prepare_prediction_data(
                standardized,
                bundle["feature_columns"],
                bundle["columns_to_drop"],
            )
            predictions = model.predict(features).astype(int)
            output_df = raw_df.copy()
            output_df["PREDICTION"] = predictions + 1
            output_df["PREDICTION_LABEL"] = target_labels(predictions)

            metrics = None
            if TARGET_COLUMN in standardized.columns:
                actual = encode_target(standardized[TARGET_COLUMN])
                valid_rows = actual.notna()
                actual_valid = actual[valid_rows].astype(int)
                predictions_valid = predictions[valid_rows.to_numpy()]
                if len(actual_valid) > 0:
                    matrix = confusion_matrix(actual_valid, predictions_valid, labels=[0, 1]).tolist()
                    metrics = {
                        "accuracy": accuracy_score(actual_valid, predictions_valid),
                        "precision": precision_score(actual_valid, predictions_valid, zero_division=0),
                        "recall": recall_score(actual_valid, predictions_valid, zero_division=0),
                        "f1": f1_score(actual_valid, predictions_valid, zero_division=0),
                        "matrix": matrix,
                    }
                if not valid_rows.all():
                    flash("Se ignoraron filas con GROUPS vacío o inválido para calcular las métricas.", "warning")

            result = {
                "rows": len(output_df),
                "model_name": model_name,
                "metrics": metrics,
                "table": output_df.head(100).to_html(classes="data-table", index=False),
                "download": output_df.to_csv(index=False).encode("utf-8"),
            }
            # Guardamos temporalmente el CSV en sesión no es necesario: el
            # enlace usa el contenido enviado en un formulario posterior.
            app.config["LAST_BATCH_CSV"] = result["download"]
        except Exception as error:  # noqa: BLE001 - mensaje amigable para la interfaz
            flash(f"No se pudo procesar el archivo: {error}", "error")

    return render_template("batch.html", result=result)


@app.route("/download-results")
def download_results():
    content = app.config.get("LAST_BATCH_CSV")
    if not content:
        flash("Todavía no hay resultados para descargar.", "warning")
        return redirect(url_for("batch"))
    return send_file(
        io.BytesIO(content),
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name="predicciones_dengue.csv",
    )


if __name__ == "__main__":
    app.run(debug=True)
