/* Clasificador de Eficiencia Energetica - Inferencia en navegador.
 *
 * Carga modelos ONNX (LogReg y MLP), aplica StandardScaler en JS replicando
 * el de sklearn, y predice en el cliente. CSV via PapaParse, XLSX via SheetJS.
 * Sin backend: 100% estatico para GitHub Pages.
 */

// onnxruntime-web busca sus .wasm junto al script principal por defecto, lo
// que falla en GitHub Pages. Apuntamos al CDN para que coincidan versiones.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

const MODELOS = {
  logreg: { url: "modelos/logreg.onnx", session: null, nombre: "Regresion Logistica" },
  mlp:    { url: "modelos/mlp.onnx",    session: null, nombre: "Red Neuronal" },
};

let scaler = null, featuresInfo = null, metricasGlobal = null, umbralInfo = null;
let datosLote = null, columnasLote = null, resultadoLoteCache = null;


// 1. Carga inicial (JSONs en paralelo + sesiones ONNX en paralelo)
async function cargarRecursos() {
  const estado = document.getElementById("estado-modelo");
  try {
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

    construirFormulario();
    pintarAccuracyEnSelector();
    pintarUmbral();

    await Promise.all(Object.values(MODELOS).map(async m => {
      m.session = await ort.InferenceSession.create(m.url, { executionProviders: ["wasm"] });
    }));

    estado.textContent = "Modelos cargados. Listos para predecir.";
    estado.classList.add("ok");
    document.getElementById("btn-procesar").disabled = false;
  } catch (err) {
    console.error(err);
    estado.textContent = `Error: ${err.message}`;
    estado.classList.add("error");
  }
}

function pintarAccuracyEnSelector() {
  document.querySelectorAll(".modelo-acc").forEach(el => {
    const m = metricasGlobal[el.dataset.modelo];
    if (m) el.textContent = `Accuracy: ${(m.accuracy * 100).toFixed(2)}%`;
  });
}

function pintarUmbral() {
  document.getElementById("info-umbral").textContent =
    `Umbral aplicado: Y1 < ${umbralInfo.umbral_y1_mediana.toFixed(2)} kWh/m² → Eficiente.`;
}


// 2. Formulario dinamico (8 inputs en el orden de feature_names)
function construirFormulario() {
  const grid = document.getElementById("form-grid");
  grid.innerHTML = "";
  featuresInfo.feature_names.forEach((name, idx) => {
    const label = featuresInfo.feature_labels[idx] || name;
    const help = featuresInfo.feature_help?.[idx] || "";
    const min = featuresInfo.feature_min?.[idx];
    const max = featuresInfo.feature_max?.[idx];
    const div = document.createElement("div");
    div.className = "form-field";
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
 * Detecta inputs fuera del dominio observado en entrenamiento. Para features
 * continuas avisa si el valor cae fuera de [min, max]; para cuasi-categoricas
 * avisa si no coincide con ningun valor del dataset (con tolerancia float).
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
      const TOL = 1e-6;
      if (!opciones.some(o => Math.abs(o - v) < TOL)) {
        warnings.push(`${label}: ${v} no es un valor presente en el dataset (validos: ${opciones.join(", ")})`);
      }
    } else {
      const min = featuresInfo.feature_min[idx];
      const max = featuresInfo.feature_max[idx];
      if (v < min || v > max) {
        warnings.push(`${label}: ${v} fuera de [${min.toFixed(2)}, ${max.toFixed(2)}]`);
      }
    }
  });
  return warnings;
}


// 3. Escalado replicando sklearn StandardScaler.transform: z = (x - mean) / scale
function escalar(fila) {
  const out = new Float32Array(fila.length);
  for (let i = 0; i < fila.length; i++) {
    out[i] = (fila[i] - scaler.mean[i]) / scaler.scale[i];
  }
  return out;
}


/**
 * Inferencia ONNX para un batch [N x F]. Tensor en row-major float32.
 * Devuelve { predicciones: int[N], probas: [pBenigno, pMaligno][N] }.
 * Localizamos labels/probabilities por regex porque skl2onnx puede generar
 * distintos nombres de salida segun la version.
 */
async function predecirBatch(filas) {
  const modeloKey = document.querySelector('input[name="modelo"]:checked').value;
  const session = MODELOS[modeloKey].session;
  if (!session) throw new Error("Modelo no cargado.");

  const N = filas.length;
  const F = scaler.mean.length;
  const data = new Float32Array(N * F);
  for (let i = 0; i < N; i++) data.set(escalar(filas[i]), i * F);

  const tensor = new ort.Tensor("float32", data, [N, F]);
  const results = await session.run({ input: tensor });
  const outNames = Object.keys(results);
  const labelOut = results[outNames.find(n => /label/i.test(n))] || results[outNames[0]];
  const probOut  = results[outNames.find(n => /prob/i.test(n))]  || results[outNames[1]];

  // labelOut.data puede ser BigInt64Array; Number() normaliza a numero JS.
  const predicciones = Array.from(labelOut.data, v => Number(v));
  const probas = [];
  for (let i = 0; i < N; i++) probas.push([probOut.data[i * 2], probOut.data[i * 2 + 1]]);
  return { predicciones, probas, modelo: MODELOS[modeloKey].nombre };
}


// 4. Lectura y validacion del formulario individual
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

document.getElementById("form-individual").addEventListener("submit", async (e) => {
  e.preventDefault();
  const valores = leerFormulario();
  if (!valores) {
    alert("Completa todos los campos como numeros validos.");
    return;
  }
  try {
    const { predicciones, probas } = await predecirBatch([valores]);
    mostrarResultadoIndividual(predicciones[0], probas[0], detectarOOD());
  } catch (err) {
    console.error(err);
    alert("Error: " + err.message);
  }
});

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

  document.getElementById("ind-prob-ef").textContent = (proba[1] * 100).toFixed(2) + "%";
  document.getElementById("ind-prob-no").textContent = (proba[0] * 100).toFixed(2) + "%";
  document.getElementById("prob-bar").style.width = (proba[1] * 100).toFixed(2) + "%";

  // Banner de extrapolacion: en zonas OOD, LogReg y MLP pueden divergir.
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

  cont.scrollIntoView({ behavior: "smooth", block: "nearest" });
}


// 5. Botones auxiliares del formulario
document.getElementById("btn-promedio").addEventListener("click", () => {
  featuresInfo.feature_names.forEach((name, idx) => {
    const input = document.getElementById(`f-${name}`);
    input.value = featuresInfo.feature_mean[idx].toFixed(4);
    input.classList.remove("invalido");
  });
});

document.getElementById("btn-limpiar").addEventListener("click", () => {
  document.getElementById("form-individual").reset();
  document.querySelectorAll("#form-grid input").forEach(i => i.classList.remove("invalido"));
  document.getElementById("resultado-individual").classList.add("oculto");
});


// 6. Carga de archivo (CSV via PapaParse, XLSX via SheetJS)
document.getElementById("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const estado = document.getElementById("estado-csv");
  estado.classList.remove("ok", "error");
  document.getElementById("resultado-lote").classList.add("oculto");
  document.getElementById("btn-descargar").classList.add("oculto");

  const ext = file.name.split(".").pop().toLowerCase();
  if (ext === "csv") {
    Papa.parse(file, {
      header: true, skipEmptyLines: true, dynamicTyping: true,
      complete: (results) => {
        datosLote = results.data;
        columnasLote = results.meta.fields;
        estado.textContent = `CSV cargado: ${datosLote.length} filas, ${columnasLote.length} columnas.`;
        estado.classList.add("ok");
      },
      error: (err) => { estado.textContent = "Error leyendo CSV: " + err.message; estado.classList.add("error"); },
    });
  } else if (ext === "xlsx" || ext === "xls") {
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const wb = XLSX.read(ev.target.result, { type: "array" });
        const ws = wb.Sheets[wb.SheetNames[0]];
        const json = XLSX.utils.sheet_to_json(ws, { defval: null });
        datosLote = json;
        columnasLote = Object.keys(json[0] || {});
        estado.textContent = `XLSX cargado: ${datosLote.length} filas, ${columnasLote.length} columnas.`;
        estado.classList.add("ok");
      } catch (err) { estado.textContent = "Error leyendo XLSX: " + err.message; estado.classList.add("error"); }
    };
    reader.readAsArrayBuffer(file);
  } else {
    estado.textContent = "Formato no soportado. Usa .csv, .xlsx o .xls.";
    estado.classList.add("error");
  }
});


// 7. Procesar lote (validacion + inferencia + render)
document.getElementById("btn-procesar").addEventListener("click", async () => {
  const estado = document.getElementById("estado-csv");
  estado.classList.remove("ok", "error");
  if (!datosLote) { estado.textContent = "Primero sube un archivo."; estado.classList.add("error"); return; }

  const requeridas = featuresInfo.feature_names;
  const faltantes = requeridas.filter(c => !columnasLote.includes(c));
  if (faltantes.length > 0) {
    estado.textContent = "Faltan columnas: " + faltantes.join(", ") + ". El archivo debe tener X1, X2, ..., X8.";
    estado.classList.add("error");
    return;
  }

  // Target detectable de dos formas:
  //   - columna "eficiente" (0/1) ya binarizada,
  //   - columna "Y1" cruda (re-aplicamos el umbral guardado).
  const tieneY1 = columnasLote.includes("Y1");
  const tieneEf = columnasLote.includes("eficiente");

  const filas = [], yReal = [];
  for (const row of datosLote) {
    const f = requeridas.map(c => Number(row[c]));
    if (f.some(v => isNaN(v))) continue;
    filas.push(f);
    if (tieneEf) yReal.push(Number(row.eficiente) === 1 ? 1 : 0);
    else if (tieneY1) yReal.push(Number(row.Y1) < umbralInfo.umbral_y1_mediana ? 1 : 0);
  }
  if (filas.length === 0) { estado.textContent = "No hay filas validas."; estado.classList.add("error"); return; }

  try {
    estado.textContent = `Procesando ${filas.length} filas...`;
    const { predicciones, probas, modelo } = await predecirBatch(filas);
    const tieneTarget = yReal.length === filas.length;
    mostrarResultadoLote(filas, predicciones, probas, yReal, tieneTarget, modelo);
    estado.textContent = `Listo. ${filas.length} predicciones con ${modelo}.`;
    estado.classList.add("ok");

    resultadoLoteCache = { filas, predicciones, probas, yReal, tieneTarget };
    document.getElementById("btn-descargar").classList.remove("oculto");
  } catch (err) {
    console.error(err);
    estado.textContent = "Error: " + err.message;
    estado.classList.add("error");
  }
});


// 8. Render del resultado de lote
function mostrarResultadoLote(filas, predicciones, probas, yReal, tieneTarget, modeloNombre) {
  document.getElementById("resultado-lote").classList.remove("oculto");

  const thead = document.querySelector("#tabla-predicciones thead");
  const tbody = document.querySelector("#tabla-predicciones tbody");

  const headers = ["#", ...featuresInfo.feature_names, "Prediccion", "Prob. Eficiente"];
  if (tieneTarget) headers.push("Real", "Acierto");
  thead.innerHTML = "<tr>" + headers.map(h => `<th>${h}</th>`).join("") + "</tr>";

  tbody.innerHTML = predicciones.map((p, i) => {
    const cells = [`<td>${i + 1}</td>`];
    filas[i].forEach(v => cells.push(`<td>${typeof v === "number" ? v.toFixed(2) : v}</td>`));
    const badge = p === 1
      ? '<span class="badge badge-ef">Eficiente</span>'
      : '<span class="badge badge-no">No eficiente</span>';
    cells.push(`<td>${badge}</td>`);
    cells.push(`<td>${(probas[i][1] * 100).toFixed(1)}%</td>`);
    let clase = "";
    if (tieneTarget) {
      const real = yReal[i];
      const realBadge = real === 1
        ? '<span class="badge badge-ef">Eficiente</span>'
        : '<span class="badge badge-no">No eficiente</span>';
      cells.push(`<td>${realBadge}</td>`);
      cells.push(`<td>${p === real ? "Si" : "No"}</td>`);
      clase = p === real ? "acierto" : "error";
    }
    return `<tr class="${clase}">${cells.join("")}</tr>`;
  }).join("");

  const metricasDiv = document.getElementById("metricas-lote");
  const cmWrapper = document.getElementById("cm-lote-wrapper");

  if (tieneTarget) {
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


// 9. Metricas y matriz de confusion (clase positiva = 1 = Eficiente)
//    cm = [[TN, FP], [FN, TP]] (mismo formato que sklearn).
function calcularMetricas(yTrue, yPred) {
  let tn = 0, fp = 0, fn = 0, tp = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
  }
  const accuracy = (tp + tn) / yTrue.length;
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = (precision + recall) === 0 ? 0 : 2 * precision * recall / (precision + recall);
  return { accuracy, precision, recall, f1, cm: [[tn, fp], [fn, tp]] };
}

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


// 10. Descarga de resultados como CSV (Blob -> URL temporal -> click)
document.getElementById("btn-descargar").addEventListener("click", () => {
  if (!resultadoLoteCache) return;
  const { filas, predicciones, probas, yReal, tieneTarget } = resultadoLoteCache;

  const cols = [...featuresInfo.feature_names, "prediccion", "prob_eficiente"];
  if (tieneTarget) cols.push("real", "acierto");
  const lineas = [cols.join(",")];

  for (let i = 0; i < predicciones.length; i++) {
    const fila = [
      ...filas[i],
      predicciones[i] === 1 ? "Eficiente" : "No eficiente",
      probas[i][1].toFixed(4),
    ];
    if (tieneTarget) {
      fila.push(yReal[i] === 1 ? "Eficiente" : "No eficiente");
      fila.push(predicciones[i] === yReal[i] ? "Si" : "No");
    }
    lineas.push(fila.join(","));
  }

  const blob = new Blob([lineas.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "predicciones_eficiencia.csv";
  a.click();
  URL.revokeObjectURL(url);
});


// 11. Tabs
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + target).classList.add("active");
  });
});


cargarRecursos();
