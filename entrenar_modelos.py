"""
=============================================================================
PIPELINE DE ENTRENAMIENTO Y EXPORTACION A ONNX
Dataset: Energy Efficiency (UCI, ENB2012_data.xlsx)
=============================================================================

PROBLEMA
--------
El dataset original es de regresion (predecir Heating Load Y1 y Cooling Load
Y2 de 768 edificios simulados a partir de 8 features arquitectonicas X1..X8).
Aqui lo convertimos en clasificacion binaria:

    eficiente (1) = (Y1 < mediana(Y1))   # baja carga de calefaccion
    no eficiente (0) = (Y1 >= mediana(Y1))

Usar la mediana garantiza un balance perfecto de clases (384/384). Asi la
accuracy es una metrica honesta sin necesidad de ponderar clases.

PIPELINE
--------
1. Leer XLSX y construir el target binario.
2. train/test split 70/30 estratificado por target (random_state=42).
3. StandardScaler ajustado SOLO en train (no toca el test -> no hay leakage).
4. Noise augmentation en el set de train: K=5 copias con ruido gaussiano.
5. Entrenar LogReg y MLP (con GridSearchCV para el MLP).
6. Evaluar en el set de test (no aumentado).
7. Exportar todo lo que la web necesita: modelos en ONNX, scaler, features,
   metricas y umbral en JSON.

¿POR QUE NOISE AUGMENTATION?
----------------------------
El dataset es una simulacion con valores discretos: X5 (Altura) toma solo
{3.5, 7}, X7 (Acristalada) toma solo {0, 0.1, 0.25, 0.4}, etc. Sin augmentar,
los modelos aprenden fronteras "afiladas" entre esos puntos discretos. Cuando
un usuario ingresa un valor intermedio (ej. Altura=6), LogReg (lineal) y MLP
(no lineal) extrapolan de formas distintas y pueden dar veredictos opuestos.
Anadiendo ruido gaussiano pequenio en el espacio escalado, ambos modelos
aprenden a producir predicciones suaves alrededor de cada punto -> en zonas
intermedias coinciden mucho mas.

SALIDAS (consumidas por la web en docs/)
----------------------------------------
docs/modelos/logreg.onnx     - Regresion Logistica en formato ONNX.
docs/modelos/mlp.onnx        - MLP en formato ONNX.
docs/modelos/scaler.json     - mean y scale del StandardScaler (para escalar
                               las entradas dentro del navegador en JS).
docs/modelos/features.json   - Nombres tecnicos (X1..X8), labels en español,
                               ayudas, rangos min/max/mean y feature_options
                               (lista de valores discretos para cada feature
                               cuasi-categorica, usada por la UI para detectar
                               inputs OOD y mostrar warnings).
docs/modelos/metricas.json   - Accuracy, precision, recall, F1 y matriz de
                               confusion de cada modelo + mejores
                               hiperparametros del MLP + parametros del
                               augmentation.
docs/modelos/umbral.json     - Mediana de Y1. Permite que la web reclasifique
                               un CSV que solo traiga Y1 cruda.

matrices_confusion/*.png     - Visualizaciones para incluir en informes.
ejemplo_lote.csv             - 20 filas reales del set de prueba (con Y1)
                               para que el usuario pueda probar el lote sin
                               armar un CSV manualmente.
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


# =============================================================================
# CONFIGURACION DE RUTAS
# =============================================================================
# Todas las rutas se derivan de ROOT (carpeta del script) para que el script
# sea invocable desde cualquier directorio sin romper.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ENB2012_data.xlsx"
MODELS_DIR = ROOT / "docs" / "modelos"
CM_DIR = ROOT / "matrices_confusion"
EJEMPLO_CSV = ROOT / "ejemplo_lote.csv"
EJEMPLO_CSV_DOCS = ROOT / "docs" / "ejemplo_lote.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. CARGA Y CONSTRUCCION DEL TARGET BINARIO
# =============================================================================
print(f">> Leyendo {DATA_PATH.name}...")
df = pd.read_excel(DATA_PATH)
# Algunas versiones del XLSX traen espacios accidentales en los headers.
df.columns = [c.strip() for c in df.columns]

# X1..X8 son features arquitectonicas; Y1 (Heating Load) y Y2 (Cooling Load)
# son targets de regresion del dataset original. Usar Y1 o Y2 como feature
# seria leakage trivial -> los excluimos.
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]

# Mediana sobre TODO el dataset (no solo train) -> umbral global y estable.
# Usar la mediana garantiza balance 50/50 por construccion.
umbral = float(df["Y1"].median())
print(f">> Mediana de Y1 = {umbral:.4f}")

y = (df["Y1"] < umbral).astype(int)
X = df[FEATURES].copy()
print(f">> Distribucion de clases: {y.value_counts().to_dict()}")

# stratify=y preserva la proporcion 50/50 de clases en train y test.
# random_state=42 -> reproducibilidad; test_size=0.30 -> 30% para evaluacion.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


# =============================================================================
# 2. ESCALADO (StandardScaler)
# =============================================================================
# StandardScaler transforma cada feature a media=0, desviacion=1.
#
# CRITICO: fit_transform SOLO en X_train. Si ajustaramos el scaler sobre X
# completo o sobre X_test, estariamos filtrando informacion del set de prueba
# al entrenamiento (data leakage), y las metricas serian optimistas.
#
# El test se transforma con el scaler ya ajustado (transform, no fit_transform).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =============================================================================
# 3. NOISE AUGMENTATION DEL SET DE ENTRENAMIENTO
# =============================================================================
# Generamos K copias de X_train con ruido gaussiano pequenio (en el espacio
# escalado). Esto enseña al modelo a producir predicciones suaves en la
# vecindad de cada punto y reduce dramaticamente la divergencia entre LogReg
# y MLP cuando el usuario ingresa valores intermedios.
#
# IMPORTANTE: solo se augmenta train. El test queda intacto -> las metricas
# siguen reflejando el desempeño real sobre datos no vistos.
K = 5             # Numero de copias adicionales -> 6x mas datos de train.
NOISE_STD = 0.15  # 15% de una desviacion estandar (datos ya estandarizados).
rng = np.random.RandomState(42)
augm_X = [X_train_scaled] + [
    X_train_scaled + rng.normal(0, NOISE_STD, X_train_scaled.shape)
    for _ in range(K)
]
augm_y = [y_train.values] * (K + 1)
X_train_aug = np.vstack(augm_X)
y_train_aug = np.concatenate(augm_y)
print(f">> Noise augmentation: {X_train_scaled.shape[0]} -> {X_train_aug.shape[0]} "
      f"muestras (K={K}, std={NOISE_STD}).")


# =============================================================================
# 4. MODELO 1: REGRESION LOGISTICA
# =============================================================================
# max_iter=5000 da margen suficiente para convergencia (el default de 100
# ocasionalmente lanza warnings con datos pequenios).
# No se tunea: con datos escalados y regularizacion L2 default, LogReg
# converge a una solucion estable y un grid no aporta valor practico.
print("\n>> Entrenando Regresion Logistica...")
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train_aug, y_train_aug)


# =============================================================================
# 5. MODELO 2: MLP CON GridSearchCV
# =============================================================================
# El espacio cubre las 4 dimensiones que mas afectan a un MLP:
#   - hidden_layer_sizes: arquitectura (1 o 2 capas, 8-32 neuronas).
#   - activation: relu vs tanh.
#   - alpha: regularizacion L2 (1e-4 default vs 1e-3 mas fuerte).
#   - learning_rate_init: tasa de aprendizaje inicial del optimizador adam.
# 4*2*2*2 = 32 combinaciones, cada una con cv=3 -> 96 fits totales.
# n_jobs=-1 paraleliza en todos los cores disponibles.
print("\n>> Entrenando MLP con GridSearchCV...")
param_grid = {
    "hidden_layer_sizes": [(16,), (32,), (16, 8), (32, 16)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-4, 1e-3],
    "learning_rate_init": [1e-3, 1e-2],
}
grid = GridSearchCV(
    MLPClassifier(max_iter=2000, random_state=42),  # random_state fija pesos iniciales.
    param_grid=param_grid,
    cv=3,                # 3-fold CV: balance entre robustez y costo computacional.
    n_jobs=-1,           # Paraleliza en todos los cores.
    scoring="accuracy",  # Las clases estan balanceadas 50/50 -> accuracy es honesta.
    verbose=1,
)
grid.fit(X_train_aug, y_train_aug)
mlp = grid.best_estimator_  # Mejor modelo segun la metrica de scoring.
print(f">> Mejores hiperparametros MLP: {grid.best_params_}")


# =============================================================================
# 6. EVALUACION (sobre el set de test que el modelo NUNCA vio)
# =============================================================================
def evaluar(modelo, nombre: str) -> dict:
    """
    Calcula las 4 metricas estandar de clasificacion binaria + matriz de
    confusion de un modelo entrenado, evaluado sobre X_test_scaled.

    CONVENCION DE LA MATRIZ DE CONFUSION (formato sklearn):
        cm[0][0] = TN (real=0, pred=0): no eficientes correctamente clasificados.
        cm[0][1] = FP (real=0, pred=1): falsos eficientes (error mas costoso
                                        en certificaciones de eficiencia).
        cm[1][0] = FN (real=1, pred=0): eficientes no detectados.
        cm[1][1] = TP (real=1, pred=1): eficientes correctamente clasificados.
    """
    y_pred = modelo.predict(X_test_scaled)
    return {
        "nombre": nombre,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        # precision = TP / (TP + FP). Que tan confiables son los "eficiente".
        "precision": float(precision_score(y_test, y_pred)),
        # recall = TP / (TP + FN). Que % de los realmente eficientes detectamos.
        "recall": float(recall_score(y_test, y_pred)),
        # f1 = media armonica de precision y recall.
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


metricas = {
    "logreg": evaluar(logreg, "Regresion Logistica"),
    "mlp": evaluar(mlp, "Red Neuronal (MLP)"),
    # Las tuplas no son JSON-serializables -> convertir a lista.
    "mlp_best_params": {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in grid.best_params_.items()
    },
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


# =============================================================================
# 7. VISUALIZACION DE LA MATRIZ DE CONFUSION (PNG)
# =============================================================================
def plot_cm(cm: np.ndarray, titulo: str, out_path: Path) -> None:
    """
    Renderiza la matriz de confusion como heatmap con numeros encima.
    Texto blanco si la celda es oscura, negro si es clara, para legibilidad.
    Cmap "Greens" coherente con la paleta de la UI.
    """
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
    plt.close(fig)  # Libera memoria.


plot_cm(np.array(metricas["logreg"]["confusion_matrix"]),
        "Regresion Logistica", CM_DIR / "logreg_cm.png")
plot_cm(np.array(metricas["mlp"]["confusion_matrix"]),
        "Red Neuronal (MLP)", CM_DIR / "mlp_cm.png")
print(f">> Matrices de confusion: {CM_DIR}")


# =============================================================================
# 8. EXPORTACION DEL SCALER, FEATURES Y UMBRAL A JSON
# =============================================================================
# scaler.json: el StandardScaler se guarda como JSON (no se incrusta dentro
# del grafo ONNX) para que el preprocesamiento sea auditable y modificable
# desde JS sin tener que regenerar los modelos. La transformacion equivalente
# en JS es: z[i] = (x[i] - mean[i]) / scale[i].
(MODELS_DIR / "scaler.json").write_text(json.dumps({
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_names": FEATURES,
}, indent=2), encoding="utf-8")

# features.json: contiene todo lo que la UI necesita para construir el
# formulario y validar inputs (ver detectarOOD en docs/app.js).
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
    # Min/max calculados del dataset real (no hardcoded) -> validacion HTML5
    # nativa via <input min="..." max="...">.
    "feature_min": [float(X[c].min()) for c in FEATURES],
    "feature_max": [float(X[c].max()) for c in FEATURES],
    "feature_mean": [float(X[c].mean()) for c in FEATURES],
    # Para features con <= 10 valores unicos (cuasi-categoricas: X3..X8),
    # guardamos la lista de valores observados. La UI usa esto para detectar
    # inputs OOD y mostrar warnings cuando el usuario tipea un valor que el
    # modelo nunca vio durante el entrenamiento.
    # Para X1, X2 (continuas) guardamos None -> la UI hace check de rango.
    "feature_options": [
        sorted([float(v) for v in X[c].unique().tolist()])
        if X[c].nunique() <= 10 else None
        for c in FEATURES
    ],
    # Mapeo legible para el JS. Las claves son strings porque JSON no
    # soporta enteros como claves de objetos.
    "classes": {"0": "No eficiente", "1": "Eficiente"},
}, indent=2, ensure_ascii=False), encoding="utf-8")

# umbral.json: permite que la web reclasifique un CSV que solo traiga Y1
# cruda (sin la columna "eficiente"), aplicando exactamente el mismo umbral
# que se uso aqui durante el entrenamiento.
(MODELS_DIR / "umbral.json").write_text(json.dumps({
    "umbral_y1_mediana": umbral,
    "descripcion": (
        "Si Y1 < mediana del dataset -> Eficiente (1), si no -> No eficiente (0)."
    ),
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f">> JSONs guardados en {MODELS_DIR}")


# =============================================================================
# 9. CONVERSION A ONNX
# =============================================================================
def exportar_onnx(modelo, ruta: Path, n_features: int) -> None:
    """
    Convierte un estimador scikit-learn a ONNX y lo guarda en disco.

    Detalles importantes:
    - initial_type define la firma del grafo ONNX:
        nombre "input" (lo que JS pondra como llave en feeds), tipo float32,
        shape [None, n_features] (None = batch dinamico, cualquier numero
        de filas se acepta).
    - options={id(modelo): {"zipmap": False}} hace que la salida de
      probabilidades sea un tensor [N, 2] plano en vez de un array de
      diccionarios. Esto simplifica enormemente el consumo desde JS:
      con zipmap=True habria que iterar mapas; con False es solo un
      Float32Array al que accedes por indice.
    - target_opset=15: version del operador set de ONNX, estable y
      compatible con onnxruntime-web 1.17.x. Versiones mas nuevas pueden
      no estar implementadas en el navegador.
    """
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(
        modelo, initial_types=initial_type,
        options={id(modelo): {"zipmap": False}},
        target_opset=15,
    )
    ruta.write_bytes(onx.SerializeToString())


exportar_onnx(logreg, MODELS_DIR / "logreg.onnx", X_train_scaled.shape[1])
exportar_onnx(mlp, MODELS_DIR / "mlp.onnx", X_train_scaled.shape[1])
print(f">> Modelos ONNX guardados en {MODELS_DIR}")


# =============================================================================
# 10. CSV DE EJEMPLO PARA PROBAR EL LOTE
# =============================================================================
# 20 filas del set de prueba (no de train, asi el usuario puede verificar
# las metricas reportadas) con la columna Y1 incluida. La web detecta Y1,
# aplica el umbral, y calcula la matriz de confusion en vivo.
ejemplo = X_test.copy()
ejemplo["Y1"] = df.loc[X_test.index, "Y1"].values
ejemplo_head = ejemplo.head(20)

# Dos copias del archivo:
#   - ejemplo_lote.csv (raiz): visible en el repositorio.
#   - docs/ejemplo_lote.csv: accesible desde el sitio publicado en GitHub
#     Pages (que solo sirve la carpeta docs/).
ejemplo_head.to_csv(EJEMPLO_CSV, index=False)
ejemplo_head.to_csv(EJEMPLO_CSV_DOCS, index=False)
print(f">> CSV de ejemplo: {EJEMPLO_CSV} y {EJEMPLO_CSV_DOCS}")

print("\n>> Listo. Todos los artefactos estan en su lugar.")
