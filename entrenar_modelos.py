"""
=============================================================================
PIPELINE DE ENTRENAMIENTO Y EXPORTACION DE MODELOS DE CLASIFICACION BINARIA
sobre el dataset Energy Efficiency (UCI, ENB2012_data.xlsx).
=============================================================================

PROBLEMA
--------
El dataset original es de regresion (predecir Heating Load Y1 y Cooling Load Y2
de 768 edificios simulados a partir de 8 features arquitectonicas).
Aqui lo convertimos en clasificacion binaria:
    eficiente (1) si Y1 < mediana(Y1)
    no eficiente (0) si Y1 >= mediana(Y1)
Usar la mediana garantiza un balance perfecto de clases (384/384), por lo que
la accuracy es una metrica honesta sin necesidad de ponderar clases.

MODELOS
-------
1. Regresion Logistica (LogisticRegression con max_iter=5000): baseline lineal
   interpretable. Sin tuning porque tiene esencialmente un solo hiperparametro
   relevante (C, regularizacion L2) y con datos escalados converge bien.

2. Red Neuronal Artificial (MLPClassifier) optimizada con GridSearchCV(cv=3):
   barre arquitectura, activacion, regularizacion y tasa de aprendizaje.

PIPELINE
--------
1. Cargar XLSX, construir target binario.
2. train/test split 70/30 estratificado por target (random_state=42).
3. StandardScaler ajustado SOLO en train (evita data leakage).
4. Entrenar ambos modelos sobre features escaladas.
5. Calcular metricas (accuracy, precision, recall, F1, matriz de confusion)
   sobre el set de prueba.
6. Exportar a ONNX para que el navegador pueda hacer inferencia con
   onnxruntime-web (sin servidor de por medio).

SALIDAS (artefactos consumidos por la web en docs/)
---------------------------------------------------
docs/modelos/logreg.onnx     - Modelo de Regresion Logistica en formato ONNX.
docs/modelos/mlp.onnx        - Modelo MLP en formato ONNX.
docs/modelos/scaler.json     - mean y scale del StandardScaler (para escalar
                               las entradas dentro del navegador en JS).
docs/modelos/features.json   - Nombres tecnicos (X1..X8), nombres amigables
                               para la UI, ayudas, rangos min/max y media de
                               cada feature en el dataset.
docs/modelos/metricas.json   - Metricas de cada modelo + mejores hiperparametros
                               del MLP. Se usan para mostrar la accuracy junto
                               al nombre de cada modelo en la UI.
docs/modelos/umbral.json     - Mediana de Y1 usada para binarizar. Permite que
                               la web reclasifique CSVs que traigan Y1 cruda.

matrices_confusion/*.png     - Visualizaciones de la matriz de confusion
                               para incluir en informes / presentaciones.
ejemplo_lote.csv             - 20 filas reales del set de prueba (incluye Y1)
                               para que el usuario pueda probar la prediccion
                               por lotes sin tener que armar un CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# =============================================================================
# CONFIGURACION DE RUTAS
# =============================================================================
# ROOT apunta a la carpeta donde vive este script (proyecto-final/).
# Todas las demas rutas se derivan de ahi para que el script funcione sin
# importar desde donde se invoque.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ENB2012_data.xlsx"
MODELS_DIR = ROOT / "docs" / "modelos"
CM_DIR = ROOT / "matrices_confusion"
EJEMPLO_CSV = ROOT / "ejemplo_lote.csv"
EJEMPLO_CSV_DOCS = ROOT / "docs" / "ejemplo_lote.csv"

# parents=True crea carpetas intermedias; exist_ok=True evita error si ya existen.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. CARGA Y CONSTRUCCION DEL PROBLEMA DE CLASIFICACION
# =============================================================================
print(f">> Leyendo {DATA_PATH.name}...")
df = pd.read_excel(DATA_PATH)

# Limpieza preventiva: quitar espacios accidentales en nombres de columnas
# (algunas versiones del XLSX traen "X1 " con espacio al final).
df.columns = [c.strip() for c in df.columns]
print(f">> Filas: {len(df)}, columnas: {list(df.columns)}")

# Las 8 features del dataset (caracteristicas arquitectonicas del edificio).
# Y1 (Heating Load) y Y2 (Cooling Load) NO son features, son targets — usarlas
# como entrada seria data leakage trivial.
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]

# Mediana de Y1 sobre TODO el dataset (no solo train) -> umbral global y estable
# para binarizar. Usar la mediana garantiza clases perfectamente balanceadas.
umbral = float(df["Y1"].median())
print(f">> Mediana de Y1 (Heating Load) = {umbral:.4f}")

# Construccion del target:
#   1 = eficiente (carga de calefaccion menor que la mediana)
#   0 = no eficiente
y = (df["Y1"] < umbral).astype(int)
X = df[FEATURES].copy()

print(f">> Distribucion de clases: {y.value_counts().to_dict()}")

# Split 70/30:
#   - test_size=0.30 -> 30% del dataset para evaluacion final.
#   - random_state=42 -> reproducibilidad.
#   - stratify=y -> preserva la proporcion 50/50 de clases en train y test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


# =============================================================================
# 2. ESCALADO (StandardScaler)
# =============================================================================
# StandardScaler transforma cada feature a media=0 y desviacion=1.
# IMPORTANTE: fit_transform SOLO sobre X_train. Si ajustaramos sobre el dataset
# completo o sobre X_test, estariamos filtrando informacion del set de prueba
# al entrenamiento (data leakage), y las metricas serian optimistas.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =============================================================================
# 3. MODELO 1: REGRESION LOGISTICA
# =============================================================================
# max_iter=5000 -> margen amplio para garantizar convergencia (el default 100
# a veces no alcanza con datos pequeños y converge a un warning).
# No tuneamos porque LogReg con datos escalados y regularizacion L2 default
# rinde bien sin necesidad de barrer C.
print("\n>> Entrenando Regresion Logistica...")
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train_scaled, y_train)


# =============================================================================
# 4. MODELO 2: MLP CON GridSearchCV
# =============================================================================
# El espacio de busqueda cubre las 4 dimensiones que mas afectan a un MLP:
#   - hidden_layer_sizes: arquitectura (1 o 2 capas, 8-32 neuronas).
#   - activation: relu vs tanh.
#   - alpha: regularizacion L2 (1e-4 default vs 1e-3 mas fuerte).
#   - learning_rate_init: tasa de aprendizaje inicial del optimizador adam.
# 4*2*2*2 = 32 combinaciones, cada una entrenada con cv=3 -> 96 fits totales.
# n_jobs=-1 paraleliza sobre todos los cores disponibles.
print("\n>> Entrenando Red Neuronal (MLP) con GridSearchCV...")
param_grid = {
    "hidden_layer_sizes": [(16,), (32,), (16, 8), (32, 16)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-4, 1e-3],
    "learning_rate_init": [1e-3, 1e-2],
}

# max_iter=2000 da margen para que el optimizador converja en cada combinacion.
# random_state=42 fija la inicializacion de pesos para reproducibilidad.
mlp_base = MLPClassifier(max_iter=2000, random_state=42)

grid = GridSearchCV(
    mlp_base,
    param_grid=param_grid,
    cv=3,                  # 3-fold CV: balance entre robustez y costo computacional.
    n_jobs=-1,             # Paraleliza en todos los cores disponibles.
    scoring="accuracy",    # Las clases estan balanceadas (50/50), accuracy es honesta.
    verbose=1,
)
grid.fit(X_train_scaled, y_train)
mlp = grid.best_estimator_  # Mejor modelo segun la metrica de scoring.
print(f">> Mejores hiperparametros MLP: {grid.best_params_}")


# =============================================================================
# 5. EVALUACION (sobre el set de prueba que el modelo NUNCA vio)
# =============================================================================
def evaluar(modelo, nombre: str) -> dict:
    """
    Calcula las 4 metricas estandar de clasificacion + matriz de confusion.

    Convencion de la matriz de confusion en sklearn (con clases [0, 1]):
        cm[0][0] = TN (verdaderos negativos: real=0, pred=0 -> no eficientes detectados)
        cm[0][1] = FP (falsos positivos:    real=0, pred=1 -> falsos eficientes)
        cm[1][0] = FN (falsos negativos:    real=1, pred=0 -> eficientes no detectados)
        cm[1][1] = TP (verdaderos positivos: real=1, pred=1 -> eficientes detectados)

    En este dominio el FP es el error mas costoso (etiquetar como eficiente
    un edificio que no lo es).
    """
    y_pred = modelo.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "nombre": nombre,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        # precision = TP / (TP + FP). Que tan confiables son los "eficiente".
        "precision": float(precision_score(y_test, y_pred)),
        # recall = TP / (TP + FN). Que % de los realmente eficientes detectamos.
        "recall": float(recall_score(y_test, y_pred)),
        # f1 = media armonica de precision y recall.
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": cm.tolist(),
    }


# Estructura del JSON consumida por la web:
#   - metricas.logreg / metricas.mlp -> resultados de cada modelo.
#   - metricas.mlp_best_params -> visible en "Sobre el proyecto" si se quisiera.
#   - n_train / n_test -> contexto del split.
metricas = {
    "logreg": evaluar(logreg, "Regresion Logistica"),
    "mlp": evaluar(mlp, "Red Neuronal (MLP)"),
    "mlp_best_params": {
        # Las tuplas no son JSON-serializables, las convertimos a lista.
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in grid.best_params_.items()
    },
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
(MODELS_DIR / "metricas.json").write_text(
    json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\n>> Metricas guardadas en {MODELS_DIR / 'metricas.json'}")
for key in ("logreg", "mlp"):
    m = metricas[key]
    print(
        f"   {m['nombre']}: acc={m['accuracy']:.4f}  "
        f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  f1={m['f1']:.4f}"
    )


# =============================================================================
# 6. VISUALIZACION DE LA MATRIZ DE CONFUSION (PNG)
# =============================================================================
def plot_cm(cm: np.ndarray, titulo: str, out_path: Path) -> None:
    """
    Renderiza la matriz de confusion con un heatmap y los numeros encima.
    Texto blanco si la celda es oscura, negro si es clara, para legibilidad.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Greens")  # Verde, coherente con la paleta de la web.
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No eficiente", "Eficiente"])
    ax.set_yticklabels(["No eficiente", "Eficiente"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(titulo)

    # Sobreescribir cada celda con su valor numerico.
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i][j]),
                ha="center", va="center",
                color="white" if cm[i][j] > vmax / 2 else "black",
                fontsize=14, fontweight="bold",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)  # Libera memoria; importante si se llama muchas veces.


plot_cm(np.array(metricas["logreg"]["confusion_matrix"]),
        "Regresion Logistica", CM_DIR / "logreg_cm.png")
plot_cm(np.array(metricas["mlp"]["confusion_matrix"]),
        "Red Neuronal (MLP)", CM_DIR / "mlp_cm.png")
print(f">> Matrices de confusion guardadas en {CM_DIR}")


# =============================================================================
# 7. EXPORTACION DEL SCALER, FEATURES Y UMBRAL A JSON
# =============================================================================
# El scaler se guarda como JSON (no se incrusta en el grafo ONNX) para que el
# preprocesamiento sea auditable y modificable desde JS sin regenerar modelos.
# La transformacion equivalente en JS es: z[i] = (x[i] - mean[i]) / scale[i].
(MODELS_DIR / "scaler.json").write_text(json.dumps({
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_names": FEATURES,
}, indent=2), encoding="utf-8")

# features.json contiene todo lo que la UI necesita para construir el formulario:
# nombres tecnicos (X1..X8) en el orden esperado por el modelo, etiquetas
# legibles en español, ayudas, rangos para validar inputs y la media para el
# boton "Cargar valores promedio".
features_meta = {
    "feature_names": FEATURES,
    "feature_labels": [
        "Compacidad Relativa",
        "Area de Superficie (m²)",
        "Area de Muros (m²)",
        "Area de Techo (m²)",
        "Altura Total (m)",
        "Orientacion (2-5)",
        "Area Acristalada (0-0.4)",
        "Distribucion del Acristalamiento (0-5)",
    ],
    "feature_help": [
        "Indice geometrico del edificio. Rangos tipicos del dataset: 0.62 a 0.98.",
        "Superficie total exterior. Rangos: 514 a 808 m².",
        "Area total de muros. Rangos: 245 a 416 m².",
        "Area de techo. Rangos: 110 a 220 m².",
        "Altura del edificio. Solo dos niveles en el dataset: 3.5 m o 7 m.",
        "Orientacion cardinal codificada del 2 al 5.",
        "Fraccion del area acristalada. Valores: 0, 0.1, 0.25 o 0.4.",
        "Patron de distribucion del acristalamiento, codificado del 0 al 5.",
    ],
    # Min/max calculados dinamicamente del dataset (no hardcoded), para que
    # los <input min="..." max="..."> reflejen la realidad de los datos.
    "feature_min": [float(X[c].min()) for c in FEATURES],
    "feature_max": [float(X[c].max()) for c in FEATURES],
    "feature_mean": [float(X[c].mean()) for c in FEATURES],
    # Para features con pocos valores unicos (categoricas disfrazadas de
    # numericas: X5 altura, X6 orientacion, X7 acristalamiento, X8 distribucion)
    # guardamos la lista de valores validos para poder renderizar un <select>
    # en la UI y prevenir entradas out-of-distribution.
    # Para features con muchos valores (X1-X4, continuas) guardamos None.
    "feature_options": [
        sorted([float(v) for v in X[c].unique().tolist()])
        if X[c].nunique() <= 10 else None
        for c in FEATURES
    ],
    # Mapeo legible para el JS. Las claves son strings porque JSON no soporta
    # enteros como claves de objetos.
    "classes": {"0": "No eficiente", "1": "Eficiente"},
}
(MODELS_DIR / "features.json").write_text(
    json.dumps(features_meta, indent=2, ensure_ascii=False), encoding="utf-8"
)

# umbral.json permite que la web vuelva a clasificar un CSV que solo traiga Y1
# (sin la columna "eficiente") usando exactamente el mismo umbral que aqui.
(MODELS_DIR / "umbral.json").write_text(json.dumps({
    "umbral_y1_mediana": umbral,
    "descripcion": (
        "Si Heating Load (Y1) < mediana del dataset, el edificio se considera "
        "Eficiente (clase 1). Si Y1 >= mediana, No eficiente (clase 0)."
    ),
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f">> scaler.json, features.json y umbral.json guardados en {MODELS_DIR}")


# =============================================================================
# 8. CONVERSION A ONNX
# =============================================================================
def exportar_onnx(modelo, ruta: Path, n_features: int) -> None:
    """
    Convierte un estimador scikit-learn a ONNX y lo guarda en disco.

    Detalles:
    - initial_type define la firma de entrada del grafo ONNX:
        nombre "input", tipo float32, shape [None, n_features]
        (None = batch dinamico, cualquier numero de filas).
    - options={id(modelo): {"zipmap": False}} hace que la salida de
      probabilidades sea un tensor [N, 2] en vez de un array de
      diccionarios. Esto simplifica enormemente el consumo desde JS.
    - target_opset=15 es estable y compatible con onnxruntime-web 1.17.x
      (ediciones mas nuevas pueden no estar soportadas en el navegador).
    """
    initial_type = [("input", FloatTensorType([None, n_features]))]
    opciones = {id(modelo): {"zipmap": False}}
    onx = convert_sklearn(modelo, initial_types=initial_type,
                          options=opciones, target_opset=15)
    ruta.write_bytes(onx.SerializeToString())


exportar_onnx(logreg, MODELS_DIR / "logreg.onnx", X_train_scaled.shape[1])
exportar_onnx(mlp, MODELS_DIR / "mlp.onnx", X_train_scaled.shape[1])
print(f">> Modelos ONNX guardados en {MODELS_DIR}")


# =============================================================================
# 9. CSV DE EJEMPLO PARA PROBAR EL LOTE
# =============================================================================
# Tomamos 20 filas del set de prueba (no de train, para que el usuario pueda
# verificar las metricas del modelo en datos no vistos) e incluimos la
# columna Y1 cruda. La web detecta Y1 automaticamente, aplica el umbral,
# y calcula la matriz de confusion en vivo.
ejemplo = X_test.copy()
ejemplo["Y1"] = df.loc[X_test.index, "Y1"].values
ejemplo_head = ejemplo.head(20)

# Guardamos dos copias del CSV:
#   - ejemplo_lote.csv (raiz): visible en el repo para que el usuario lo vea.
#   - docs/ejemplo_lote.csv: accesible desde el sitio publicado en GitHub Pages
#     (que solo sirve la carpeta docs/).
ejemplo_head.to_csv(EJEMPLO_CSV, index=False)
ejemplo_head.to_csv(EJEMPLO_CSV_DOCS, index=False)
print(f">> CSV de ejemplo guardado en {EJEMPLO_CSV} y {EJEMPLO_CSV_DOCS}")

print("\n>> Listo. Todos los artefactos estan en su lugar.")
