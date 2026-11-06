"""Funciones compartidas para limpiar los datos y preparar predicciones.

El notebook y la aplicación Flask usan este mismo módulo para garantizar que
los datos de entrenamiento y los datos ingresados por el usuario reciban el
mismo tratamiento.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


TARGET_COLUMN = "GROUPS"

# Columnas que llegan desde el Excel original y que se conservan como entradas.
BASE_INPUT_COLUMNS = [
    "AGE",
    "GENDER",
    "DURATION_OF_FEVER",
    "HEADACHE",
    "MYALGIA",
    "ABDOMINAL_PAIN",
    "RASH",
    "VOMITING",
    "BREATHLESSNESS",
    "BLEEDING",
    "BP_AT_ADMISSION",
    "ORGANOMEGALY",
    "HB",
    "PCV",
    "LOWEST_WBC",
    "HIGHEST_WBC",
    "PLATELET_COUNT",
    "ALT",
    "CREATININE",
    "FERRITIN",
    "IV_FLUIDS",
    "ANTIBIOTICS",
    "ANTIPYRETICS",
    "STEROIDS",
]

BINARY_COLUMNS = [
    "AGE",
    "GENDER",
    "HEADACHE",
    "MYALGIA",
    "ABDOMINAL_PAIN",
    "RASH",
    "VOMITING",
    "BREATHLESSNESS",
    "BLEEDING",
    "BP_AT_ADMISSION",
    "ORGANOMEGALY",
    "IV_FLUIDS",
    "ANTIBIOTICS",
    "ANTIPYRETICS",
    "STEROIDS",
]

SYMPTOM_COLUMNS = ["HEADACHE", "MYALGIA", "ABDOMINAL_PAIN", "RASH", "VOMITING"]
TREATMENT_COLUMNS = ["IV_FLUIDS", "ANTIBIOTICS", "ANTIPYRETICS", "STEROIDS"]

# Se usa como límite documentado para detectar columnas casi constantes.
CONSTANT_COLUMN_THRESHOLD = 0.95
MISSING_COLUMN_THRESHOLD = 0.30


def _normalized_key(value: object) -> str:
    """Convierte encabezados variados a una clave comparable."""

    text = str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_")


HEADER_ALIASES = {
    "GROUPS_DENGUE_FEVER_OR_COMPLICATED_DENGUE": TARGET_COLUMN,
    "GROUP": TARGET_COLUMN,
    "TARGET": TARGET_COLUMN,
    "LABEL": TARGET_COLUMN,
    "DURATION_OF_FEVER": "DURATION_OF_FEVER",
    "BP_AT_ADMISSION": "BP_AT_ADMISSION",
    "PLATELET_COUNT": "PLATELET_COUNT",
    "LOWEST_WBC": "LOWEST_WBC",
    "HIGHEST_WBC": "HIGHEST_WBC",
    "IV_FLUIDS": "IV_FLUIDS",
}


def canonical_column_name(value: object) -> str:
    key = _normalized_key(value)
    return HEADER_ALIASES.get(key, key)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los encabezados del Excel, CSV o formulario."""

    result = df.copy()
    result.columns = [canonical_column_name(column) for column in result.columns]
    return result


def _encode_binary_value(value: object) -> object:
    """Codifica respuestas Sí/No y códigos 1/2 como 1/0."""

    if pd.isna(value):
        return value
    if isinstance(value, str):
        value_clean = value.strip().lower()
        if value_clean in {"si", "sí", "yes", "y", "true", "1"}:
            return 1
        if value_clean in {"no", "n", "false", "0", "2"}:
            return 0
    if value in (1, 1.0):
        return 1
    if value in (2, 2.0):
        return 0
    return value


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in BINARY_COLUMNS:
        if column in result.columns:
            result[column] = result[column].map(_encode_binary_value)
    return result


def encode_target(series: pd.Series) -> pd.Series:
    """Devuelve 0 = dengue no complicado y 1 = dengue complicado."""

    def encode(value: object) -> object:
        if pd.isna(value):
            return value
        if isinstance(value, str):
            value_clean = value.strip().lower()
            if "complic" in value_clean or value_clean in {"2", "true", "yes"}:
                return 1
            if "uncomplic" in value_clean or value_clean in {"1", "false", "no"}:
                return 0
        if value in (2, 2.0):
            return 1
        if value in (1, 1.0):
            return 0
        if value in (0, 0.0):
            return 0
        return value

    return series.map(encode).astype("Int64")


def find_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Detecta columnas con demasiados nulos o casi constantes."""

    columns_to_drop: list[str] = []
    for column in df.columns:
        if column == TARGET_COLUMN:
            continue
        missing_ratio = df[column].isna().mean()
        if missing_ratio >= MISSING_COLUMN_THRESHOLD:
            columns_to_drop.append(column)
            continue
        non_null = df[column].dropna()
        if not non_null.empty and non_null.value_counts(normalize=True).iloc[0] >= CONSTANT_COLUMN_THRESHOLD:
            columns_to_drop.append(column)
    return columns_to_drop


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas solicitadas en el proyecto."""

    result = df.copy()
    if {"LOWEST_WBC", "HIGHEST_WBC"}.issubset(result.columns):
        result["WBC_RANGE"] = result["HIGHEST_WBC"] - result["LOWEST_WBC"]
    if {"HB", "PCV"}.issubset(result.columns):
        result["HB_PCV_RATIO"] = result["HB"] / result["PCV"].replace(0, pd.NA)

    available_symptoms = [column for column in SYMPTOM_COLUMNS if column in result.columns]
    if available_symptoms:
        result["SYMPTOM_COUNT"] = result[available_symptoms].sum(axis=1, min_count=1)
    available_treatments = [column for column in TREATMENT_COLUMNS if column in result.columns]
    if available_treatments:
        result["TREATMENT_COUNT"] = result[available_treatments].sum(axis=1, min_count=1)
    return result


def prepare_training_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """Limpia el Excel y devuelve X, y, columnas eliminadas y un resumen."""

    df = standardize_columns(raw_df)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo {TARGET_COLUMN!r}.")

    initial_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df = encode_binary_columns(df)
    df[TARGET_COLUMN] = encode_target(df[TARGET_COLUMN])
    df = df.dropna(subset=[TARGET_COLUMN])

    columns_to_drop = find_columns_to_drop(df)
    cleaned = df.drop(columns=columns_to_drop, errors="ignore")
    features = add_engineered_features(cleaned.drop(columns=[TARGET_COLUMN]))
    features = features.apply(pd.to_numeric, errors="coerce")
    target = cleaned[TARGET_COLUMN].astype(int)

    summary = {
        "initial_rows": initial_rows,
        "duplicate_rows_removed": duplicate_rows,
        "final_rows": len(features),
        "columns_dropped": columns_to_drop,
        "missing_values_after_cleaning": features.isna().sum().to_dict(),
        "class_counts": target.value_counts().sort_index().to_dict(),
    }
    return features, target, columns_to_drop, summary


def prepare_prediction_data(
    raw_df: pd.DataFrame,
    feature_columns: Iterable[str],
    columns_to_drop: Iterable[str],
) -> pd.DataFrame:
    """Aplica a nuevos datos exactamente la transformación del entrenamiento."""

    df = standardize_columns(raw_df)
    df = encode_binary_columns(df)
    df = df.drop(columns=list(columns_to_drop), errors="ignore")
    # Los formularios HTTP entregan todos los valores como texto. Convertimos
    # antes de crear variables derivadas para evitar operaciones como
    # ``HIGHEST_WBC - LOWEST_WBC`` entre cadenas.
    df = df.apply(pd.to_numeric, errors="coerce")
    features = add_engineered_features(df.drop(columns=[TARGET_COLUMN], errors="ignore"))
    features = features.apply(pd.to_numeric, errors="coerce")

    feature_columns = list(feature_columns)
    for column in feature_columns:
        if column not in features.columns:
            features[column] = pd.NA
    return features[feature_columns]


def target_labels(values: Iterable[int]) -> list[str]:
    return ["Dengue no complicado" if int(value) == 0 else "Dengue complicado" for value in values]
