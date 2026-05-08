/* =============================================================================
 * CLASIFICADOR DE EFICIENCIA ENERGETICA - LOGICA DEL FRONTEND
 *
 * Este archivo orquesta toda la inferencia que ocurre en el navegador:
 *   1. Carga de los modelos ONNX (Regresion Logistica y Red Neuronal)
 *      y de los JSON con scaler, features, metricas y umbral.
 *   2. Construccion dinamica del formulario de prediccion individual
 *      (los campos se generan a partir de features.json).
 *   3. Escalado StandardScaler en JavaScript (replicando el de sklearn).
 *   4. Inferencia con onnxruntime-web (ONNX Runtime para WebAssembly).
 *   5. Lectura de archivos CSV (PapaParse) y XLSX (SheetJS) para lotes.
 *   6. Calculo en vivo de matriz de confusion y metricas.
 *   7. Render de resultados (badges, prob bar, heatmap, tabla, descarga CSV).
 *
 * DEPENDENCIAS (cargadas desde CDN en index.html):
 *   - ort.min.js          -> onnxruntime-web (motor de inferencia ONNX)
 *   - papaparse.min.js    -> parser de CSV en streaming
 *   - xlsx.full.min.js    -> SheetJS, lector de XLSX/XLS
 *
 * NO HAY BACKEND. Toda la informacion (los XLSX subidos, los inputs del
 * formulario, los resultados) permanece en la memoria del navegador. El
 * usuario puede operar offline una vez cargados los modelos.
 * ===========================================================================*/


/* -----------------------------------------------------------------------------
 * CONFIGURACION DE onnxruntime-web
 * ---------------------------------------------------------------------------*/
// onnxruntime-web descarga sus binarios .wasm a peticion. Por defecto los busca
// junto al script principal (lo que falla en GitHub Pages). Forzamos que los
// busque en el mismo CDN desde donde cargamos ort.min.js, asi siempre coinciden
// las versiones del JS y del WASM.
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";


/* -----------------------------------------------------------------------------
 * REGISTRO DE MODELOS
 * ---------------------------------------------------------------------------*/
// Un objeto por modelo con su URL relativa y el slot donde guardaremos la
// sesion ONNX una vez creada. Las sesiones son objetos pesados (cargan
// el grafo en memoria), asi que las creamos una sola vez al inicio.
const MODELOS = {
  logreg: { url: "modelos/logreg.onnx", session: null, nombre: "Regresion Logistica" },
  mlp:    { url: "modelos/mlp.onnx",    session: null, nombre: "Red Neuronal" },
};


/* -----------------------------------------------------------------------------
 * ESTADO GLOBAL
 * ---------------------------------------------------------------------------*/
// Datos cargados desde los JSON al arrancar.
let scaler = null;        // { mean: [...], scale: [...], feature_names: [...] }
let featuresInfo = null;  // { feature_names, feature_labels, feature_help, ... }
let metricasGlobal = null; // { logreg: {...}, mlp: {...}, mlp_best_params: {...} }
let umbralInfo = null;    // { umbral_y1_mediana, descripcion }

// Estado del flujo de prediccion por lotes.
let datosLote = null;          // Array de objetos: filas crudas del archivo subido.
let columnasLote = null;       // Array de strings: nombres de columnas detectadas.
let resultadoLoteCache = null; // Ultimo resultado de batch para el boton "Descargar".


/* -----------------------------------------------------------------------------
 * 1. CARGA INICIAL DE RECURSOS
 *
 * Hace 4 fetches en paralelo a los JSON (rapido, son archivos pequeños),
 * construye el formulario, y luego inicializa las dos sesiones ONNX en
 * paralelo. Si algo falla, deja un mensaje rojo en pantalla y no avanza.
 * ---------------------------------------------------------------------------*/
async function cargarRecursos() {
  const estado = document.getElementById("estado-modelo");
  try {
    // Promise.all dispara las 4 peticiones simultaneamente y resuelve cuando
    // todas terminan. Mucho mas rapido que esperar una tras otra.
    const [scalerRes, featRes, metRes, umbRes] = await Promise.all([
      fetch("modelos/scaler.json"),
      fetch("modelos/features.json"),
      fetch("modelos/metricas.json"),
      fetch("modelos/umbral.json"),
    ]);
    if (!scalerRes.ok || !featRes.ok || !metRes.ok || !umbRes.ok) {
      throw new Error("Faltan archivos JSON en modelos/. ¿Corriste entrenar_modelos.py?");
    }

    scaler = await scalerRes.json();
    featuresInfo = await featRes.json();
    metricasGlobal = await metRes.json();
    umbralInfo = await umbRes.json();

    // El formulario depende de featuresInfo, asi que lo construimos despues
    // de tener los JSON pero antes de cargar los modelos (asi el usuario ve
    // la UI completa lo antes posible).
    construirFormulario();
    pintarAccuracyEnSelector();
    pintarUmbral();

    // Carga las sesiones ONNX en paralelo. Cada InferenceSession.create
    // descarga el .onnx, lo parsea y prepara los kernels de WebAssembly.
    // executionProviders=["wasm"] usa el backend de WebAssembly puro
    // (compatible con todos los navegadores modernos).
    await Promise.all(
      Object.entries(MODELOS).map(async ([key, m]) => {
        m.session = await ort.InferenceSession.create(m.url, {
          executionProviders: ["wasm"],
        });
      })
    );

    estado.textContent = "Modelos cargados. Listos para predecir.";
    estado.classList.add("ok");
    document.getElementById("btn-procesar").disabled = false;
  } catch (err) {
    console.error(err);
    estado.textContent = `Error: ${err.message}`;
    estado.classList.add("error");
  }
}

/**
 * Pinta la accuracy de cada modelo (leida de metricas.json) junto a su
 * nombre en las radio cards del selector. Util para que el usuario sepa
 * cual modelo elegir antes de probar.
 */
function pintarAccuracyEnSelector() {
  document.querySelectorAll(".modelo-acc").forEach(el => {
    const m = metricasGlobal[el.dataset.modelo];
    if (m) el.textContent = `Accuracy: ${(m.accuracy * 100).toFixed(2)}%`;
  });
}

/**
 * Muestra el umbral usado para binarizar Y1. El usuario puede asi
 * interpretar las predicciones en terminos de la mediana real del dataset.
 */
function pintarUmbral() {
  document.getElementById("info-umbral").textContent =
    `Umbral aplicado: Y1 < ${umbralInfo.umbral_y1_mediana.toFixed(2)} kWh/m² → Eficiente.`;
}


/* -----------------------------------------------------------------------------
 * 2. CONSTRUCCION DINAMICA DEL FORMULARIO INDIVIDUAL
 *
 * En vez de hardcodear 8 campos en el HTML, los generamos a partir de
 * featuresInfo.feature_names. Esto garantiza que el orden de los inputs
 * SIEMPRE coincida con el orden esperado por el modelo (no hay riesgo
 * de mezclar X3 con X4 al editar el HTML).
 * ---------------------------------------------------------------------------*/
function construirFormulario() {
  const grid = document.getElementById("form-grid");
  grid.innerHTML = "";

  featuresInfo.feature_names.forEach((name, idx) => {
    // name -> nombre tecnico (X1..X8), usado como id del input y para mapear
    //         despues al orden esperado por el modelo.
    // label -> nombre amigable en español, mostrado al usuario.
    const label = featuresInfo.feature_labels[idx] || name;
    const help = featuresInfo.feature_help?.[idx] || "";
    // Min y max calculados del dataset real -> validacion HTML5 nativa.
    const min = featuresInfo.feature_min?.[idx];
    const max = featuresInfo.feature_max?.[idx];

    const div = document.createElement("div");
    div.className = "form-field";
    // Todas las features se ingresan como input numerico libre. La validacion
    // OOD se hace en el momento de predecir (ver detectarOOD).
    div.innerHTML = `
      <label for="f-${name}">${label}</label>
      <input type="number" step="any" id="f-${name}" name="${name}"
             min="${min}" max="${max}" required />
      <span class="help">${help}</span>
    `;
    grid.appendChild(div);
  });
}

/**
 * Detecta valores fuera del dominio observado durante el entrenamiento.
 * Devuelve una lista de mensajes legibles para mostrar como warning.
 *
 * Hay dos tipos de feature en el dataset:
 *   - Continuas (X1, X2): warning si el valor cae fuera de [min, max].
 *   - Cuasi-categoricas (X3..X8): solo toman ciertos valores discretos en
 *     el dataset. Warning si el valor ingresado no coincide exactamente con
 *     ninguno de los valores validos.
 *
 * En cualquiera de los dos casos, el modelo esta extrapolando y los dos
 * clasificadores (LogReg vs MLP) pueden divergir en sus predicciones.
 */
function detectarOOD() {
  const warnings = [];
  featuresInfo.feature_names.forEach((name, idx) => {
    const el = document.getElementById(`f-${name}`);
    if (!el || el.value === "") return;
    const v = Number(el.value);
    if (isNaN(v)) return;

    const opciones = featuresInfo.feature_options?.[idx];
    const label = featuresInfo.feature_labels[idx];

    if (Array.isArray(opciones) && opciones.length > 0) {
      // Feature cuasi-categorica: el valor debe coincidir con alguno de la lista.
      // Tolerancia pequeña para diferencias de punto flotante (ej. 0.1 vs 0.1000001).
      const TOL = 1e-6;
      const match = opciones.some(o => Math.abs(o - v) < TOL);
      if (!match) {
        warnings.push(
          `${label}: ${v} no es un valor presente en el dataset (validos: ${opciones.join(", ")})`
        );
      }
    } else {
      // Feature continua: debe estar en [min, max].
      const min = featuresInfo.feature_min[idx];
      const max = featuresInfo.feature_max[idx];
      if (v < min || v > max) {
        warnings.push(
          `${label}: ${v} fuera de [${min.toFixed(2)}, ${max.toFixed(2)}]`
        );
      }
    }
  });
  return warnings;
}


/* -----------------------------------------------------------------------------
 * 3. ESCALADO EN JS (replica de sklearn.preprocessing.StandardScaler)
 *
 * sklearn StandardScaler.transform aplica: z = (x - mean) / scale
 * donde mean y scale son los aprendidos en X_train durante fit.
 *
 * Para que la inferencia en navegador de exactamente los mismos resultados
 * que en Python, debemos aplicar EXACTAMENTE la misma transformacion. Por
 * eso guardamos mean y scale en scaler.json y los usamos aqui.
 * ---------------------------------------------------------------------------*/
function escalar(fila) {
  // Float32Array es el tipo que ONNX Runtime espera para tensores float32.
  // Crearlo aqui evita una conversion extra al construir el tensor.
  const out = new Float32Array(fila.length);
  for (let i = 0; i < fila.length; i++) {
    out[i] = (fila[i] - scaler.mean[i]) / scaler.scale[i];
  }
  return out;
}


/* -----------------------------------------------------------------------------
 * 4. INFERENCIA ONNX PARA UN BATCH (N filas x F features)
 *
 * Layout del tensor de entrada:
 *   - Shape [N, F] (N filas, F=8 features).
 *   - Almacenamiento row-major (fila por fila, contiguo en memoria).
 *   - Tipo float32.
 *
 * El modelo devuelve dos tensores:
 *   - "label": int64 [N], la clase predicha (0 o 1).
 *   - "probabilities": float32 [N, 2], probabilidad de cada clase.
 *     Como exportamos con zipmap=False, es un tensor plano (no un array
 *     de diccionarios), facil de leer linealmente.
 * ---------------------------------------------------------------------------*/
async function predecirBatch(filas) {
  // Determina cual modelo usar leyendo el radio button activo. El selector
  // es unico y compartido por las dos pestañas (individual y lote), asi
  // que la seleccion persiste entre vistas.
  const modeloKey = document.querySelector('input[name="modelo"]:checked').value;
  const session = MODELOS[modeloKey].session;
  if (!session) throw new Error("Modelo no cargado.");

  const N = filas.length;
  const F = scaler.mean.length;

  // Aplanamos el batch en un solo Float32Array de tamaño N*F en orden row-major.
  // ONNX espera el tensor "plano" con la shape declarada por separado.
  const data = new Float32Array(N * F);
  for (let i = 0; i < N; i++) {
    // .set copia los F valores escalados de la fila i en la posicion i*F.
    data.set(escalar(filas[i]), i * F);
  }

  const tensor = new ort.Tensor("float32", data, [N, F]);
  // run() recibe un objeto { nombreEntrada: tensor } y devuelve un objeto con
  // todas las salidas del grafo. La entrada se llama "input" porque asi la
  // declaramos en exportar_onnx() en Python.
  const results = await session.run({ input: tensor });

  // Localizamos las salidas por nombre de forma flexible: skl2onnx puede
  // generarlas como "label"/"probabilities", "output_label"/"output_probability",
  // u otros prefijos segun la version. Una busqueda por regex es robusta
  // a esas variaciones.
  const outNames = Object.keys(results);
  const labelOut = results[outNames.find(n => /label/i.test(n))] || results[outNames[0]];
  const probOut  = results[outNames.find(n => /prob/i.test(n))]  || results[outNames[1]];

  // labelOut.data puede ser BigInt64Array (en algunos opsets) o Int64Array.
  // Number(v) convierte ambos a numero JS estandar para no contaminar el
  // resto del codigo con tipos especiales.
  const predicciones = Array.from(labelOut.data, v => Number(v));

  // probOut.data viene plano: [p0_clase0, p0_clase1, p1_clase0, p1_clase1, ...].
  // Lo desplegamos en filas de 2 elementos para facilitar el consumo.
  const probas = [];
  for (let i = 0; i < N; i++) {
    probas.push([probOut.data[i * 2], probOut.data[i * 2 + 1]]);
  }
  return { predicciones, probas, modelo: MODELOS[modeloKey].nombre };
}


/* -----------------------------------------------------------------------------
 * 5. LECTURA Y VALIDACION DEL FORMULARIO INDIVIDUAL
 * ---------------------------------------------------------------------------*/
/**
 * Recorre los 8 inputs en el orden definido por feature_names, valida que
 * todos sean numeros, y devuelve el array de valores. Si algun campo es
 * invalido lo marca visualmente (clase .invalido) y devuelve null.
 */
function leerFormulario() {
  const valores = [];
  let ok = true;
  for (const name of featuresInfo.feature_names) {
    const input = document.getElementById(`f-${name}`);
    const v = input.value.trim();
    if (v === "" || isNaN(Number(v))) {
      input.classList.add("invalido");
      ok = false;
    } else {
      input.classList.remove("invalido");
      valores.push(Number(v));
    }
  }
  return ok ? valores : null;
}

// Submit del formulario individual: lee, valida, predice y muestra el resultado.
document.getElementById("form-individual").addEventListener("submit", async (e) => {
  e.preventDefault();  // Evita el recarga de pagina default del form.
  const valores = leerFormulario();
  if (!valores) {
    alert("Completa todos los campos como numeros validos.");
    return;
  }
  try {
    // predecirBatch acepta N filas; aqui pasamos un batch de 1.
    const { predicciones, probas } = await predecirBatch([valores]);
    // Detectamos extrapolacion antes de mostrar el resultado: si el usuario
    // ingreso valores continuos fuera del rango de entrenamiento, los modelos
    // pueden divergir mucho entre si (LogReg vs MLP) porque extrapolan de
    // formas distintas. Lo avisamos visiblemente.
    const warnings = detectarOOD();
    mostrarResultadoIndividual(predicciones[0], probas[0], warnings);
  } catch (err) {
    console.error(err);
    alert("Error: " + err.message);
  }
});

/**
 * Pinta el resultado de la prediccion individual:
 *   - Tarjeta verde (eficiente) o roja (no eficiente) con icono y texto.
 *   - Probabilidades de ambas clases en porcentaje.
 *   - Barra de probabilidad horizontal (proporcional a la prob de eficiente).
 *   - Texto interpretativo contextual.
 */
function mostrarResultadoIndividual(pred, proba, warnings = []) {
  const cont = document.getElementById("resultado-individual");
  cont.classList.remove("oculto");

  const card = document.getElementById("diagnostico-card");
  const icono = document.getElementById("ind-icono");
  const diag = document.getElementById("ind-prediccion");
  const interp = document.getElementById("interpretacion");

  if (pred === 1) {
    card.className = "diagnostico-card eficiente";
    icono.textContent = "🟢";
    diag.textContent = "Edificio Eficiente";
    interp.textContent =
      "Este edificio probablemente tendra una baja carga de calefaccion (por debajo de la mediana del dataset). " +
      "Caracteristicas como una alta compacidad relativa y menor area de superficie suelen favorecer este resultado.";
  } else {
    card.className = "diagnostico-card no-eficiente";
    icono.textContent = "🔴";
    diag.textContent = "Edificio No Eficiente";
    interp.textContent =
      "Este edificio probablemente tendra una carga de calefaccion alta. " +
      "Considera revisar la compacidad, el area acristalada y la altura: factores que tipicamente impactan el consumo energetico.";
  }

  // proba[0] = P(no eficiente), proba[1] = P(eficiente).
  document.getElementById("ind-prob-ef").textContent = (proba[1] * 100).toFixed(2) + "%";
  document.getElementById("ind-prob-no").textContent = (proba[0] * 100).toFixed(2) + "%";
  document.getElementById("prob-bar").style.width = (proba[1] * 100).toFixed(2) + "%";

  // Banner de extrapolacion si hay valores fuera del rango de entrenamiento.
  // En esa zona LogReg y MLP pueden divergir bastante (es un comportamiento
  // esperado, no un error del modelo).
  const oodBox = document.getElementById("ood-warning");
  if (warnings.length > 0) {
    oodBox.classList.remove("oculto");
    oodBox.innerHTML =
      "<strong>⚠ Valores fuera del rango de entrenamiento:</strong><ul>" +
      warnings.map(w => `<li>${w}</li>`).join("") +
      "</ul><p>El modelo esta extrapolando, por lo que LogReg y la Red Neuronal pueden " +
      "dar resultados diferentes. Para predicciones confiables usa valores dentro de los rangos indicados.</p>";
  } else {
    oodBox.classList.add("oculto");
  }

  // Hace scroll suave para que el resultado quede visible si el form era largo.
  cont.scrollIntoView({ behavior: "smooth", block: "nearest" });
}


/* -----------------------------------------------------------------------------
 * 6. BOTONES AUXILIARES DEL FORMULARIO INDIVIDUAL
 * ---------------------------------------------------------------------------*/
// "Cargar valores promedio": rellena cada campo con la media calculada del
// dataset (guardada en features.json). Util para que el usuario tenga un
// punto de partida razonable en lugar de inputs vacios.
document.getElementById("btn-promedio").addEventListener("click", () => {
  featuresInfo.feature_names.forEach((name, idx) => {
    const input = document.getElementById(`f-${name}`);
    input.value = featuresInfo.feature_mean[idx].toFixed(4);
    input.classList.remove("invalido");
  });
});

// "Limpiar": resetea el formulario y oculta el resultado anterior.
document.getElementById("btn-limpiar").addEventListener("click", () => {
  document.getElementById("form-individual").reset();
  document.querySelectorAll("#form-grid input").forEach(i => i.classList.remove("invalido"));
  document.getElementById("resultado-individual").classList.add("oculto");
});


/* -----------------------------------------------------------------------------
 * 7. CARGA DE ARCHIVO PARA PREDICCION POR LOTES (CSV o XLSX)
 *
 * Detecta la extension y usa el parser apropiado:
 *   - .csv -> PapaParse (streaming, muy rapido, dynamicTyping convierte
 *             "12.5" -> 12.5 automaticamente).
 *   - .xlsx/.xls -> SheetJS (XLSX). Lee como ArrayBuffer y convierte la
 *             primera hoja a JSON.
 *
 * En ambos casos terminamos con:
 *   - datosLote: array de objetos {col1: val, col2: val, ...}.
 *   - columnasLote: array con los nombres de columnas en orden.
 * ---------------------------------------------------------------------------*/
document.getElementById("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Reset visual: el usuario subio un archivo nuevo, ocultamos resultados
  // del archivo anterior y limpiamos los estados de error/exito.
  const estado = document.getElementById("estado-csv");
  estado.classList.remove("ok", "error");
  document.getElementById("resultado-lote").classList.add("oculto");
  document.getElementById("btn-descargar").classList.add("oculto");

  const ext = file.name.split(".").pop().toLowerCase();

  if (ext === "csv") {
    Papa.parse(file, {
      header: true,             // Primera fila = nombres de columna.
      skipEmptyLines: true,     // Ignora lineas vacias (comunes al final).
      dynamicTyping: true,      // "1.5" -> 1.5, "true" -> true automaticamente.
      complete: (results) => {
        datosLote = results.data;
        columnasLote = results.meta.fields;
        estado.textContent = `CSV cargado: ${datosLote.length} filas, ${columnasLote.length} columnas.`;
        estado.classList.add("ok");
      },
      error: (err) => {
        estado.textContent = "Error leyendo CSV: " + err.message;
        estado.classList.add("error");
      },
    });
  } else if (ext === "xlsx" || ext === "xls") {
    // SheetJS no soporta File directamente; hay que leer como ArrayBuffer.
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        // type:"array" indica que el input es un ArrayBuffer.
        const wb = XLSX.read(ev.target.result, { type: "array" });
        // Tomamos solo la primera hoja del libro (suficiente para este caso de uso).
        const ws = wb.Sheets[wb.SheetNames[0]];
        // sheet_to_json con defval:null asegura que celdas vacias sean null
        // en vez de undefined (mas predecible al filtrar despues).
        const json = XLSX.utils.sheet_to_json(ws, { defval: null });
        datosLote = json;
        columnasLote = Object.keys(json[0] || {});
        estado.textContent = `XLSX cargado: ${datosLote.length} filas, ${columnasLote.length} columnas.`;
        estado.classList.add("ok");
      } catch (err) {
        estado.textContent = "Error leyendo XLSX: " + err.message;
        estado.classList.add("error");
      }
    };
    reader.readAsArrayBuffer(file);
  } else {
    estado.textContent = "Formato no soportado. Usa .csv, .xlsx o .xls.";
    estado.classList.add("error");
  }
});


/* -----------------------------------------------------------------------------
 * 8. PROCESAMIENTO DEL LOTE
 *
 * Flujo:
 *   a) Validar que el archivo tenga las 8 columnas X1..X8.
 *   b) Detectar si trae target (eficiente directo, o Y1 para reclasificar).
 *   c) Convertir cada fila al orden esperado por el modelo (las columnas
 *      pueden venir en cualquier orden dentro del CSV; las mapeamos por nombre).
 *   d) Filtrar filas con valores invalidos (NaN).
 *   e) Llamar predecirBatch.
 *   f) Mostrar resultados (metricas si hay target, solo predicciones si no).
 * ---------------------------------------------------------------------------*/
document.getElementById("btn-procesar").addEventListener("click", async () => {
  const estado = document.getElementById("estado-csv");
  estado.classList.remove("ok", "error");

  if (!datosLote) {
    estado.textContent = "Primero sube un archivo.";
    estado.classList.add("error");
    return;
  }

  // Validacion: el archivo debe tener todas las features X1..X8.
  // Si falta alguna, no hay forma de hacer una prediccion correcta y mostramos
  // un error claro al usuario indicando que columnas faltan.
  const requeridas = featuresInfo.feature_names;
  const faltantes = requeridas.filter(c => !columnasLote.includes(c));
  if (faltantes.length > 0) {
    estado.textContent = "Faltan columnas: " + faltantes.join(", ") +
      ". El archivo debe tener X1, X2, ..., X8.";
    estado.classList.add("error");
    return;
  }

  // Detectamos el target. Hay dos formas validas de incluirlo:
  //   - Columna "eficiente" con 0/1 (target ya binarizado).
  //   - Columna "Y1" con la carga de calefaccion cruda; aplicamos el mismo
  //     umbral guardado en umbral.json para binarizarla aqui.
  const tieneY1 = columnasLote.includes("Y1");
  const tieneEf = columnasLote.includes("eficiente");

  // Construimos las matrices que usara el modelo:
  //   filas: [N x 8] valores numericos en el orden X1..X8.
  //   yReal: [N] target binario (solo si tieneEf || tieneY1).
  // Filtramos cualquier fila con valores no numericos para evitar NaN
  // en el escalado (lo que produciria predicciones invalidas).
  const filas = [];
  const yReal = [];
  for (const row of datosLote) {
    const f = requeridas.map(c => Number(row[c]));
    if (f.some(v => isNaN(v))) continue;  // Skip fila invalida silenciosamente.
    filas.push(f);
    if (tieneEf) {
      yReal.push(Number(row.eficiente) === 1 ? 1 : 0);
    } else if (tieneY1) {
      yReal.push(Number(row.Y1) < umbralInfo.umbral_y1_mediana ? 1 : 0);
    }
  }
  if (filas.length === 0) {
    estado.textContent = "No hay filas validas para procesar.";
    estado.classList.add("error");
    return;
  }

  try {
    estado.textContent = `Procesando ${filas.length} filas...`;
    const { predicciones, probas, modelo } = await predecirBatch(filas);

    // tieneTarget = true si pudimos construir yReal del mismo tamaño que filas.
    // Solo entonces mostramos metricas y matriz de confusion.
    const tieneTarget = yReal.length === filas.length;
    mostrarResultadoLote(filas, predicciones, probas, yReal, tieneTarget, modelo);
    estado.textContent = `Listo. ${filas.length} predicciones con ${modelo}.`;
    estado.classList.add("ok");

    // Guardamos en cache lo necesario para poder generar el CSV de descarga
    // sin tener que rehacer la inferencia.
    resultadoLoteCache = { filas, predicciones, probas, yReal, tieneTarget };
    document.getElementById("btn-descargar").classList.remove("oculto");
  } catch (err) {
    console.error(err);
    estado.textContent = "Error: " + err.message;
    estado.classList.add("error");
  }
});


/* -----------------------------------------------------------------------------
 * 9. RENDER DE RESULTADOS DEL LOTE
 * ---------------------------------------------------------------------------*/
/**
 * Construye:
 *   - Una tabla con todas las predicciones (con badges visuales por clase).
 *   - Tarjetas de metricas (accuracy, precision, recall, F1) si hay target;
 *     un resumen simple (filas, eficientes, no eficientes) si no.
 *   - La matriz de confusion como heatmap si hay target.
 */
function mostrarResultadoLote(filas, predicciones, probas, yReal, tieneTarget, modeloNombre) {
  document.getElementById("resultado-lote").classList.remove("oculto");

  const thead = document.querySelector("#tabla-predicciones thead");
  const tbody = document.querySelector("#tabla-predicciones tbody");

  // Cabecera dinamica: si no hay target, no mostramos las columnas Real/Acierto.
  const headers = ["#", ...featuresInfo.feature_names, "Prediccion", "Prob. Eficiente"];
  if (tieneTarget) headers.push("Real", "Acierto");
  thead.innerHTML = "<tr>" + headers.map(h => `<th>${h}</th>`).join("") + "</tr>";

  // Construimos las filas en HTML. Para datasets grandes esto podria ser lento;
  // dado que el ejemplo es de ~20 filas y el dataset completo de 768 max,
  // el approach simple (innerHTML) es adecuado.
  tbody.innerHTML = predicciones.map((p, i) => {
    const cells = [`<td>${i + 1}</td>`];
    // Cada feature con 2 decimales para no abrumar al usuario.
    filas[i].forEach(v => cells.push(`<td>${typeof v === "number" ? v.toFixed(2) : v}</td>`));

    // Badge coloreado segun la prediccion (verde eficiente, rojo no eficiente).
    const badge = p === 1
      ? '<span class="badge badge-ef">Eficiente</span>'
      : '<span class="badge badge-no">No eficiente</span>';
    cells.push(`<td>${badge}</td>`);
    cells.push(`<td>${(probas[i][1] * 100).toFixed(1)}%</td>`);

    // Si hay target, agregamos columnas Real y Acierto, y resaltamos la fila.
    let clase = "";
    if (tieneTarget) {
      const real = yReal[i];
      const realBadge = real === 1
        ? '<span class="badge badge-ef">Eficiente</span>'
        : '<span class="badge badge-no">No eficiente</span>';
      cells.push(`<td>${realBadge}</td>`);
      cells.push(`<td>${p === real ? "Si" : "No"}</td>`);
      // Tinte verde para aciertos, rojo para errores (visible en hover y por scan).
      clase = p === real ? "acierto" : "error";
    }
    return `<tr class="${clase}">${cells.join("")}</tr>`;
  }).join("");

  const metricasDiv = document.getElementById("metricas-lote");
  const cmWrapper = document.getElementById("cm-lote-wrapper");

  if (tieneTarget) {
    // Caso 1: hay target -> calcular y mostrar metricas + matriz de confusion.
    const m = calcularMetricas(yReal, predicciones);
    metricasDiv.innerHTML = `
      <div class="metric"><span class="valor" style="font-size:1rem">${modeloNombre}</span><span class="nombre">Modelo</span></div>
      <div class="metric"><span class="valor">${(m.accuracy * 100).toFixed(2)}%</span><span class="nombre">Accuracy</span></div>
      <div class="metric"><span class="valor">${(m.precision * 100).toFixed(2)}%</span><span class="nombre">Precision</span></div>
      <div class="metric"><span class="valor">${(m.recall * 100).toFixed(2)}%</span><span class="nombre">Recall</span></div>
      <div class="metric"><span class="valor">${(m.f1 * 100).toFixed(2)}%</span><span class="nombre">F1-Score</span></div>
    `;
    cmWrapper.classList.remove("oculto");
    renderMatrizConfusion(m.cm);
  } else {
    // Caso 2: solo features -> resumen descriptivo, sin metricas.
    const ef = predicciones.filter(p => p === 1).length;
    metricasDiv.innerHTML = `
      <div class="metric"><span class="valor" style="font-size:1rem">${modeloNombre}</span><span class="nombre">Modelo</span></div>
      <div class="metric"><span class="valor">${predicciones.length}</span><span class="nombre">Filas</span></div>
      <div class="metric"><span class="valor">${ef}</span><span class="nombre">Eficientes</span></div>
      <div class="metric"><span class="valor">${predicciones.length - ef}</span><span class="nombre">No eficientes</span></div>
    `;
    cmWrapper.classList.add("oculto");
  }

  document.getElementById("resultado-lote").scrollIntoView({ behavior: "smooth", block: "nearest" });
}


/* -----------------------------------------------------------------------------
 * 10. CALCULO DE METRICAS Y MATRIZ DE CONFUSION
 *
 * Convencion: clase positiva = 1 = "Eficiente".
 *   TP = real Eficiente y predicho Eficiente.
 *   TN = real No eficiente y predicho No eficiente.
 *   FP = real No eficiente pero predicho Eficiente (falso positivo, costoso
 *        en el contexto de certificaciones de eficiencia energetica).
 *   FN = real Eficiente pero predicho No eficiente (falso negativo).
 *
 * Devolvemos cm en el formato sklearn: [[TN, FP], [FN, TP]].
 * ---------------------------------------------------------------------------*/
function calcularMetricas(yTrue, yPred) {
  let tn = 0, fp = 0, fn = 0, tp = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
  }
  const accuracy  = (tp + tn) / yTrue.length;
  // Salvaguardas contra division por cero (cuando el modelo no predice una
  // clase en absoluto sobre el batch, lo que puede pasar con batches pequeños).
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall    = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1        = (precision + recall) === 0 ? 0
                    : 2 * precision * recall / (precision + recall);
  return { accuracy, precision, recall, f1, cm: [[tn, fp], [fn, tp]] };
}

/**
 * Renderiza la matriz de confusion como un heatmap CSS Grid (no imagen):
 *   - 3x3 celdas (cabeceras + 2x2 valores).
 *   - Cada celda de valor se colorea segun su semantica
 *     (verde para aciertos, rojo para FP, naranja/ambar para FN).
 *   - Numero grande con el conteo y etiqueta TN/FP/FN/TP.
 */
function renderMatrizConfusion(cm) {
  const [[tn, fp], [fn, tp]] = cm;
  document.getElementById("cm-lote").innerHTML = `
    <div class="cm-cell cm-header"></div>
    <div class="cm-cell cm-header">Pred. No eficiente</div>
    <div class="cm-cell cm-header">Pred. Eficiente</div>

    <div class="cm-cell cm-header">Real No eficiente</div>
    <div class="cm-cell cm-value cm-value-tn"><span class="num">${tn}</span><span class="lbl">TN</span></div>
    <div class="cm-cell cm-value cm-value-fp"><span class="num">${fp}</span><span class="lbl">FP</span></div>

    <div class="cm-cell cm-header">Real Eficiente</div>
    <div class="cm-cell cm-value cm-value-fn"><span class="num">${fn}</span><span class="lbl">FN</span></div>
    <div class="cm-cell cm-value cm-value-tp"><span class="num">${tp}</span><span class="lbl">TP</span></div>
  `;
}


/* -----------------------------------------------------------------------------
 * 11. DESCARGA DE RESULTADOS COMO CSV
 *
 * Construye un CSV en memoria con las features originales + prediccion +
 * probabilidad + (si hay target) real y acierto. Lo entrega como Blob via
 * URL.createObjectURL, simulando un click en un <a download>.
 * ---------------------------------------------------------------------------*/
document.getElementById("btn-descargar").addEventListener("click", () => {
  if (!resultadoLoteCache) return;
  const { filas, predicciones, probas, yReal, tieneTarget } = resultadoLoteCache;

  // Cabecera del CSV (siempre incluye features + prediccion + probabilidad).
  const cols = [...featuresInfo.feature_names, "prediccion", "prob_eficiente"];
  if (tieneTarget) cols.push("real", "acierto");
  const lineas = [cols.join(",")];

  // Cuerpo: una linea por prediccion.
  for (let i = 0; i < predicciones.length; i++) {
    const fila = [
      ...filas[i],
      predicciones[i] === 1 ? "Eficiente" : "No eficiente",
      probas[i][1].toFixed(4),  // Probabilidad de eficiente con 4 decimales.
    ];
    if (tieneTarget) {
      fila.push(yReal[i] === 1 ? "Eficiente" : "No eficiente");
      fila.push(predicciones[i] === yReal[i] ? "Si" : "No");
    }
    lineas.push(fila.join(","));
  }

  // Blob -> URL temporal -> trigger de descarga.
  const blob = new Blob([lineas.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "predicciones_eficiencia.csv";
  a.click();
  URL.revokeObjectURL(url);  // Libera la URL temporal (buena practica).
});


/* -----------------------------------------------------------------------------
 * 12. NAVEGACION POR TABS
 *
 * Cada boton .tab tiene data-tab=<id>. Al hacer click:
 *   - Quita .active de todos los botones y paneles.
 *   - Pone .active al boton clicado y al panel #tab-<id>.
 * ---------------------------------------------------------------------------*/
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + target).classList.add("active");
  });
});


/* -----------------------------------------------------------------------------
 * ARRANQUE
 *
 * Disparamos la carga al final del archivo, una vez que todas las funciones
 * y handlers estan definidos. cargarRecursos es asincrona pero no la
 * await-eamos: dejamos que corra en background, los listeners ya estan
 * activos y mostraran un mensaje de error si algo falla.
 * ---------------------------------------------------------------------------*/
cargarRecursos();
