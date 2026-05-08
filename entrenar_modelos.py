"""
Entrena clasificacion binaria sobre Energy Efficiency (UCI):
  eficiente=1 si Y1 < mediana(Y1), 0 en caso contrario.

Modelos:
  1. Regresion Logistica.
  2. MLPClassifier optimizada con GridSearchCV(cv=3).

Pipeline: split 70/30 estratificado -> StandardScaler en train -> noise
augmentation (K=5, std=0.15) -> entrenar -> evaluar en test (sin aumentar) ->
exportar a ONNX.

Salidas (consumidas por la web en docs/):
  docs/modelos/{logreg,mlp}.onnx, scaler.json, features.json, metricas.json,
  umbral.json + matrices_confusion/*.png + ejemplo_lote.csv.
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ENB2012_data.xlsx"
MODELS_DIR = ROOT / "docs" / "modelos"
CM_DIR = ROOT / "matrices_confusion"
EJEMPLO_CSV = ROOT / "ejemplo_lote.csv"
EJEMPLO_CSV_DOCS = ROOT / "docs" / "ejemplo_lote.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)


# 1. Carga y construccion del target binario
print(f">> Leyendo {DATA_PATH.name}...")
df = pd.read_excel(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]

# Mediana sobre todo el dataset -> garantiza balance perfecto 50/50.
umbral = float(df["Y1"].median())
print(f">> Mediana de Y1 = {umbral:.4f}")

y = (df["Y1"] < umbral).astype(int)
X = df[FEATURES].copy()
print(f">> Distribucion de clases: {y.value_counts().to_dict()}")

# stratify=y preserva la proporcion 50/50 en train y test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


# 2. Escalado (fit SOLO en train para evitar data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Noise augmentation
# El dataset tiene features cuasi-categoricas (X5 toma {3.5, 7}, X7 toma
# {0, 0.1, 0.25, 0.4}, etc.). Sin augmentar, los modelos aprenden fronteras
# afiladas y diverger al evaluar valores intermedios. Generamos K copias del
# train con ruido gaussiano para enseñar al modelo a interpolar suavemente.
# El test NO se aumenta -> metricas honestas.
K, NOISE_STD = 5, 0.15
rng = np.random.RandomState(42)
augm_X = [X_train_scaled] + [X_train_scaled + rng.normal(0, NOISE_STD, X_train_scaled.shape) for _ in range(K)]
augm_y = [y_train.values] * (K + 1)
X_train_aug = np.vstack(augm_X)
y_train_aug = np.concatenate(augm_y)
print(f">> Noise augmentation: {X_train_scaled.shape[0]} -> {X_train_aug.shape[0]} muestras (K={K}, std={NOISE_STD}).")


# 4. Entrenamiento
print("\n>> Entrenando Regresion Logistica...")
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train_aug, y_train_aug)

print("\n>> Entrenando MLP con GridSearchCV...")
param_grid = {
    "hidden_layer_sizes": [(16,), (32,), (16, 8), (32, 16)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-4, 1e-3],
    "learning_rate_init": [1e-3, 1e-2],
}
grid = GridSearchCV(
    MLPClassifier(max_iter=2000, random_state=42),
    param_grid=param_grid, cv=3, n_jobs=-1, scoring="accuracy", verbose=1,
)
grid.fit(X_train_aug, y_train_aug)
mlp = grid.best_estimator_
print(f">> Mejores hiperparametros MLP: {grid.best_params_}")


# 5. Evaluacion (sobre el test set NO aumentado)
def evaluar(modelo, nombre: str) -> dict:
    """Calcula metricas y matriz de confusion. CM = [[TN, FP], [FN, TP]]."""
    y_pred = modelo.predict(X_test_scaled)
    return {
        "nombre": nombre,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


metricas = {
    "logreg": evaluar(logreg, "Regresion Logistica"),
    "mlp": evaluar(mlp, "Red Neuronal (MLP)"),
    "mlp_best_params": {k: list(v) if isinstance(v, tuple) else v
                        for k, v in grid.best_params_.items()},
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "augmentation": {"K": K, "noise_std": NOISE_STD},
}
(MODELS_DIR / "metricas.json").write_text(
    json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
)
for key in ("logreg", "mlp"):
    m = metricas[key]
    print(f"   {m['nombre']}: acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
          f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")


# 6. Matrices de confusion (PNG)
def plot_cm(cm, titulo, out_path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Greens")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["No eficiente", "Eficiente"],
           yticklabels=["No eficiente", "Eficiente"],
           xlabel="Prediccion", ylabel="Real", title=titulo)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > vmax / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


plot_cm(np.array(metricas["logreg"]["confusion_matrix"]), "Regresion Logistica", CM_DIR / "logreg_cm.png")
plot_cm(np.array(metricas["mlp"]["confusion_matrix"]), "Red Neuronal (MLP)", CM_DIR / "mlp_cm.png")
print(f">> Matrices de confusion: {CM_DIR}")


# 7. Scaler, features y umbral a JSON
# El scaler va como JSON (no incrustado en ONNX) para que el preprocesamiento
# sea auditable y modificable desde JS sin regenerar modelos.
(MODELS_DIR / "scaler.json").write_text(json.dumps({
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_names": FEATURES,
}, indent=2), encoding="utf-8")

(MODELS_DIR / "features.json").write_text(json.dumps({
    "feature_names": FEATURES,
    "feature_labels": [
        "Compacidad Relativa", "Area de Superficie (m²)", "Area de Muros (m²)",
        "Area de Techo (m²)", "Altura Total (m)", "Orientacion (2-5)",
        "Area Acristalada (0-0.4)", "Distribucion del Acristalamiento (0-5)",
    ],
    "feature_help": [
        "Indice geometrico del edificio. Rangos tipicos: 0.62 a 0.98.",
        "Superficie total exterior. Rangos: 514 a 808 m².",
        "Area total de muros. Rangos: 245 a 416 m².",
        "Area de techo. Rangos: 110 a 220 m².",
        "Altura del edificio. Solo dos niveles: 3.5 m o 7 m.",
        "Orientacion cardinal codificada del 2 al 5.",
        "Fraccion del area acristalada. Valores: 0, 0.1, 0.25 o 0.4.",
        "Patron de distribucion del acristalamiento, codificado del 0 al 5.",
    ],
    "feature_min": [float(X[c].min()) for c in FEATURES],
    "feature_max": [float(X[c].max()) for c in FEATURES],
    "feature_mean": [float(X[c].mean()) for c in FEATURES],
    # Para features con <= 10 valores unicos (cuasi-categoricas), guardamos la
    # lista para que la UI pueda detectar inputs OOD.
    "feature_options": [
        sorted([float(v) for v in X[c].unique().tolist()])
        if X[c].nunique() <= 10 else None
        for c in FEATURES
    ],
    "classes": {"0": "No eficiente", "1": "Eficiente"},
}, indent=2, ensure_ascii=False), encoding="utf-8")

(MODELS_DIR / "umbral.json").write_text(json.dumps({
    "umbral_y1_mediana": umbral,
    "descripcion": "Si Y1 < mediana del dataset -> Eficiente (1), si no -> No eficiente (0).",
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f">> JSONs guardados en {MODELS_DIR}")


# 8. Conversion a ONNX
# zipmap=False -> probabilidades como tensor [N,2] plano (facil de leer en JS).
# target_opset=15 -> compatible con onnxruntime-web 1.17.x.
def exportar_onnx(modelo, ruta, n_features):
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(
        modelo, initial_types=initial_type,
        options={id(modelo): {"zipmap": False}}, target_opset=15,
    )
    ruta.write_bytes(onx.SerializeToString())


exportar_onnx(logreg, MODELS_DIR / "logreg.onnx", X_train_scaled.shape[1])
exportar_onnx(mlp, MODELS_DIR / "mlp.onnx", X_train_scaled.shape[1])
print(f">> Modelos ONNX guardados en {MODELS_DIR}")


# 9. CSV de ejemplo (20 filas del test set con Y1 cruda para validar metricas)
ejemplo = X_test.copy()
ejemplo["Y1"] = df.loc[X_test.index, "Y1"].values
ejemplo_head = ejemplo.head(20)
ejemplo_head.to_csv(EJEMPLO_CSV, index=False)
ejemplo_head.to_csv(EJEMPLO_CSV_DOCS, index=False)  # copia accesible desde GitHub Pages
print(f">> CSV de ejemplo: {EJEMPLO_CSV} y {EJEMPLO_CSV_DOCS}")

print("\n>> Listo.")
