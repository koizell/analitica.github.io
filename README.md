# Clasificador de Eficiencia Energetica de Edificios

Proyecto final que entrena dos clasificadores binarios sobre el dataset
**Energy Efficiency** de UCI y los despliega como una aplicacion **100% estatica**
en GitHub Pages. La inferencia se ejecuta totalmente en el navegador (no hay
backend) usando modelos exportados a **ONNX** y cargados con `onnxruntime-web`.

## Que hace el proyecto

A partir de 8 parametros arquitectonicos de un edificio (compacidad, areas,
altura, orientacion, acristalamiento), predice si tendra una **carga de
calefaccion baja** ("eficiente") o no.

El dataset original de UCI plantea un problema de **regresion** sobre la carga
de calefaccion (`Y1`) y la de refrigeracion (`Y2`). Como el proyecto pide
matriz de confusion (algo de clasificacion), lo transformamos en un problema
binario:

- **Eficiente (1):** `Y1 < mediana(Y1)` (baja carga de calefaccion).
- **No eficiente (0):** `Y1 >= mediana(Y1)`.

Tanto `Y1` como `Y2` se descartan como features (seria *data leakage*).

## Modelos

1. **Regresion Logistica** (`sklearn.linear_model.LogisticRegression`).
2. **Red Neuronal Artificial** (`sklearn.neural_network.MLPClassifier`)
   optimizada con `GridSearchCV` (cv=3) sobre `hidden_layer_sizes`,
   `activation`, `alpha` y `learning_rate_init`.

Ambos se evaluan con **accuracy, precision, recall, F1-score** y matriz de
confusion sobre un set de prueba estratificado del 30%.

## Estructura del repositorio

```
proyecto-final/
├── ENB2012_data.xlsx        # dataset original (UCI)
├── entrenar_modelos.py      # pipeline: entrena, evalua, exporta a ONNX
├── requirements.txt
├── ejemplo_lote.csv         # 20 filas reales para probar el lote (se genera)
├── docs/                    # carpeta servida por GitHub Pages
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   ├── ejemplo_lote.csv     # copia accesible desde la web
│   └── modelos/
│       ├── logreg.onnx
│       ├── mlp.onnx
│       ├── scaler.json      # mean y scale del StandardScaler
│       ├── features.json    # nombres tecnicos + amigables + rangos
│       ├── metricas.json    # accuracy, precision, recall, f1, cm
│       └── umbral.json      # mediana de Y1 usada para binarizar
├── matrices_confusion/
│   ├── logreg_cm.png
│   └── mlp_cm.png
└── README.md
```

## Como entrenar los modelos localmente

Requisitos: Python 3.10+ recomendado.

```powershell
cd proyecto-final
pip install -r requirements.txt
python entrenar_modelos.py
```

El script lee `ENB2012_data.xlsx`, entrena ambos modelos, guarda metricas e
imagenes, y exporta los archivos `.onnx` y JSONs que la web necesita.

## Probar localmente (recomendado antes de desplegar)

`onnxruntime-web` no funciona desde `file://`, hay que servir la carpeta
`docs/` por HTTP:

```powershell
cd proyecto-final\docs
python -m http.server 8000
# abre http://localhost:8000 en el navegador
```

## Como desplegar en GitHub Pages

1. Crea un repositorio nuevo en GitHub (vacio, sin README ni .gitignore).
2. Sube el contenido de `proyecto-final/` (ver comandos de git mas abajo).
3. En GitHub, ve a **Settings -> Pages**.
4. En **Build and deployment**:
   - Source: **Deploy from a branch**.
   - Branch: `main` (o `master`), folder: `/docs`.
5. Guarda. En 1-2 minutos GitHub te dara la URL publica
   (algo como `https://<tu-usuario>.github.io/<tu-repo>/`).

> **Importante:** los archivos `docs/modelos/*.onnx` deben estar commiteados al
> repositorio para que la web pueda cargarlos.

## Demo

Reemplaza este enlace con la URL de tu propio despliegue:
`https://<tu-usuario>.github.io/<tu-repo>/`

---

## Sustentacion tecnica

Esta seccion explica las decisiones de diseño del proyecto. Es relevante
porque el profesor pidio que el uso de IA en la construccion del proyecto
estuviera sustentado.

### Por que binarizar usando la mediana

La mediana garantiza un **balance perfecto** entre clases (50/50), lo que
significa que la *accuracy* es una metrica honesta sin ajustes especiales y
que un modelo trivial (predecir siempre la clase mayoritaria) acertaria solo
el 50% — un baseline claro que nuestros modelos deben superar.

Alternativas consideradas:
- Usar un umbral normativo (ej. EnergyStar): rompe con el dataset porque la
  distribucion no se alinea con esos cortes.
- Usar la media: produce clases ligeramente desbalanceadas porque la
  distribucion de `Y1` no es perfectamente simetrica.

### Por que escalar con StandardScaler

- **Para el MLP:** indispensable. Sin escalar, las features con magnitudes muy
  distintas (Area de Superficie en cientos vs. Compacidad Relativa en (0,1))
  dominan los gradientes y la red no converge bien.
- **Para Regresion Logistica:** util pero no critico. Con regularizacion L2
  (default de scikit-learn) tambien se beneficia de inputs normalizados, y
  ademas hace que las dos pipelines sean comparables (mismo preprocesamiento).

El `StandardScaler` se entrena solo con `X_train` y se serializa a JSON
(`mean` y `scale`). En el navegador, JS aplica `(x - mean) / scale` antes
de pasar la fila al modelo ONNX.

### Por que GridSearchCV solo para el MLP

La regresion logistica tiene **un hiperparametro relevante** (la fuerza de
regularizacion `C`). Con datos escalados y `max_iter=5000` converge a una
solucion estable, y un grid no aporta valor practico para 768 muestras.

El MLP tiene un espacio de hiperparametros considerablemente mayor:
arquitectura (`hidden_layer_sizes`), funcion de activacion, regularizacion
(`alpha`) y tasa de aprendizaje inicial. El barrido sobre estos cuatro ejes
con `cv=3` es una busqueda razonable que no sobreajusta a una particion
particular. Se usa `accuracy` como `scoring` porque las clases estan
perfectamente balanceadas por construccion.

### Por que ONNX para llevar los modelos al navegador

GitHub Pages solo sirve archivos estaticos: no hay donde correr Python.
Las opciones realistas eran:

1. **Backend Flask + Render/Heroku:** complica la entrega (otro servicio que
   mantener) y agrega latencia de red.
2. **Reescribir los modelos a mano en JS:** factible para LogReg pero fragil
   y propenso a divergencias; impractico para un MLP optimizado por grid
   search.
3. **ONNX:** formato abierto que `skl2onnx` exporta directamente desde
   scikit-learn, y `onnxruntime-web` (WebAssembly) lo ejecuta en cualquier
   navegador moderno. Es el camino mas limpio, deja toda la inferencia en el
   cliente y elimina la dependencia de un servidor.

Se opto por mantener el `StandardScaler` fuera del grafo ONNX (en JSON aparte
y aplicado en JS) para que el preprocesamiento sea auditable y modificable
sin tener que regenerar los modelos.

### Como interpretar las metricas en este contexto

Con la convencion `[[TN, FP], [FN, TP]]` y *Eficiente = 1* como clase positiva:

|                  | Predicho No eficiente | Predicho Eficiente |
|------------------|:---------------------:|:------------------:|
| **Real No eficiente** | TN                    | FP (mas costoso)   |
| **Real Eficiente**    | FN                    | TP                 |

- **Falso Positivo (FP):** clasificar como eficiente un edificio que en
  realidad no lo es. En un sistema real de etiquetado o certificacion, esto
  podria emitir una etiqueta verde indebidamente -> **dañoso para usuarios y
  para la credibilidad del sello**. Es el error mas costoso.
- **Falso Negativo (FN):** clasificar como no eficiente un edificio que si lo
  es. Implica una sub-valoracion (perdida de incentivo) pero no compromete la
  integridad del sistema.

Por eso, ademas de la *accuracy* general, conviene mirar:
- **Precision** sobre la clase Eficiente: que tan confiables son las etiquetas
  positivas que el modelo emite.
- **Recall** sobre la clase Eficiente: que porcentaje de edificios realmente
  eficientes capturamos.

### Limitaciones honestas

- Dataset **simulado**, no edificios reales.
- Solo 768 muestras y 8 features arquitectonicas — no contempla clima,
  materiales, uso, ocupacion ni eficiencia de equipos.
- El umbral es la mediana del propio dataset; **no equivale a un estandar
  normativo** (EnergyStar, Passivhaus, etc.).
- Solo modela carga de calefaccion (`Y1`), no la de refrigeracion (`Y2`).
  Un edificio "eficiente" segun este modelo podria ser ineficiente en climas
  calidos.

## Privacidad

Toda la inferencia ocurre en el navegador del usuario. Los datos del
formulario o del archivo subido **nunca salen del dispositivo** ni se envian
a ningun servidor.
