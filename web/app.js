/* Pollen ID frontend. No build step: plain JS + Chart.js (vendored). */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const main = $("#main");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 1) => (v == null || Number.isNaN(v) ? "n/a" : (v * 100).toFixed(d) + "%");
const fix = (v, d = 3) => (v == null || Number.isNaN(v) ? "n/a" : Number(v).toFixed(d));
const sci = (name) => `<span class="sci">${esc(name)}</span>`;
const fmtTime = (s) => (s == null ? "n/a" : s >= 3600 ? `${(s / 3600).toFixed(1)} h` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`);

let charts = [];
let pollTimer = null;
function resetPage() {
  charts.forEach((c) => c.destroy());
  charts = [];
  clearInterval(pollTimer);
}
const COLORS = { ink: "#22261f", red: "#b5573b", green: "#6f8f6b", blue: "#7f93a3", grey: "#a8aa9c", pollen: "#d8a61f" };
function lineChart(canvas, labels, datasets, yMax) {
  const c = new Chart(canvas, {
    type: "line",
    data: { labels, datasets: datasets.map((d) => ({ tension: 0.25, pointRadius: 2, borderWidth: 2, spanGaps: true, ...d })) },
    options: { responsive: true, maintainAspectRatio: false, animation: false,
      scales: { y: { beginAtZero: true, max: yMax }, x: { title: { display: true, text: "Epoch" } } },
      plugins: { legend: { position: "bottom" } } },
  });
  charts.push(c);
  return c;
}

/* The run report is markdown; show it as plain, tidy text. */
const cleanReport = (t) => String(t || "").replace(/^#+\s*/gm, "").replace(/^- /gm, "\u2022 ").replace(/\*\*/g, "");

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* keep statusText */ }
    throw new Error(msg);
  }
  return r.json();
}
const errorBox = (e) => `<div class="notice warn" role="alert"><b>Something went wrong.</b><br>${esc(e.message)}</div>`;

/* Closest-match bars (shared by the confident and the "no confident match" views). */
const top5Bars = (r) => r.top5.map((t) => `<div class="bar"><span>${sci(t.species)}</span><div class="track"><div class="fill" style="width:${(t.probability * 100).toFixed(1)}%"></div></div><span class="num">${pct(t.probability)}</span></div>`).join("");

/* ------------------------------------------------------------------ Identify */
async function pageIdentify() {
  // decorative pollen grain: a spiny sphere drawn as one thin line
  const spikes = Array.from({ length: 20 }, (_, i) => {
    const a = (i / 20) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
    return `M${(60 + 29 * c).toFixed(1)} ${(60 + 29 * s).toFixed(1)}L${(60 + 38 * c).toFixed(1)} ${(60 + 38 * s).toFixed(1)}`;
  }).join("");
  main.innerHTML = `
    <h1>Identify a pollen grain</h1>
    <p class="lede">Drop a light-microscopy image to see the most likely genus and species.</p>
    <div class="predict">
      <div class="stage">
        <div class="field" id="field" tabindex="0" role="button" aria-label="Choose or drop a pollen image">
          <svg class="grain" viewBox="0 0 120 120" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round">
            <circle cx="60" cy="60" r="27"/><circle cx="60" cy="60" r="15" stroke-dasharray="1 5"/><path d="${spikes}"/></g></svg>
          <span class="hint" id="hint">Drop an image<br>or click to choose</span>
          <img id="preview" alt="Your image" hidden>
        </div>
        <input type="file" id="file" accept="image/*" hidden>
        <label class="switch"><input type="checkbox" id="cam"><span>Show attention map</span></label>
      </div>
      <div id="result" class="result" aria-live="polite"><p class="placeholder">The result will appear here.</p></div>
    </div>`;
  const field = $("#field"), file = $("#file"), preview = $("#preview"), camBox = $("#cam");
  let last = null;

  const show = (r) => {
    last = r;
    preview.src = camBox.checked && r.gradcam ? r.gradcam : r.input_view;
    preview.hidden = false; $("#hint").hidden = true;
    const matches = `<div class="matches"><h3>Closest matches</h3>${top5Bars(r)}</div>`;
    if (r.uncertain) {
      // Quiet by design: no banner and no big name, just the closest matches and a small tag.
      const label = /not a pollen/i.test(r.message || "") ? "Does not look like pollen" : "Low confidence";
      $("#result").innerHTML = `<p class="tag">${label} <span class="num">&middot; ${pct(r.species_confidence)}</span></p>${matches}`;
      return;
    }
    $("#result").innerHTML = `
      <div class="result-genus">${esc(r.genus)}</div>
      <p class="result-name">${esc(r.species)}</p>
      <p class="conf"><span class="num">${pct(r.species_confidence)}</span> confidence</p>
      ${matches}`;
  };
  const run = async (f) => {
    $("#result").innerHTML = `<p class="placeholder">Identifying...</p>`;
    preview.src = URL.createObjectURL(f); preview.hidden = false; $("#hint").hidden = true;
    const fd = new FormData(); fd.append("file", f);
    try { show(await api(`/predict?gradcam=true`, { method: "POST", body: fd })); }
    catch (e) { $("#result").innerHTML = errorBox(e); }
  };
  field.addEventListener("click", () => file.click());
  field.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => file.files[0] && run(file.files[0]));
  ["dragenter", "dragover"].forEach((ev) => field.addEventListener(ev, (e) => { e.preventDefault(); field.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => field.addEventListener(ev, (e) => { e.preventDefault(); field.classList.remove("drag"); }));
  field.addEventListener("drop", (e) => e.dataTransfer.files[0] && run(e.dataTransfer.files[0]));
  camBox.addEventListener("change", () => last && show(last));
}

/* ------------------------------------------------------------------ Training */
async function pageTraining() {
  main.innerHTML = `
    <h1>Training</h1>
    <p class="lede">Watch a run learn, or start a new one. Each run is saved so you can compare it with the others.</p>
    <div id="status"></div>
    <div class="grid2"><div><h3>Loss</h3><div class="chart-box"><canvas id="c-loss"></canvas></div></div>
    <div><h3>Accuracy</h3><div class="chart-box"><canvas id="c-acc"></canvas></div></div></div>
    <section class="block"><h2>Start a new run</h2>
      <form class="inline" id="form">
        <label>Backbone<select name="backbone"><option>convnext_tiny</option><option>tf_efficientnetv2_s</option><option>deit_small_patch16_224</option></select></label>
        <label>Image size<input type="number" name="data.image_size" value="384" min="64" step="32"></label>
        <label>Fine-tune epochs<input type="number" name="train.epochs_finetune" value="40" min="1"></label>
        <label>Batch size<input type="number" name="train.batch_size" value="16" min="1"></label>
        <label>Note<input type="text" name="note" placeholder="what is different?" maxlength="120"></label>
        <button class="primary" type="submit">Start training</button>
      </form><div id="start-msg" aria-live="polite"></div></section>`;
  let lossChart, accChart, viewing = null;

  async function refresh() {
    try {
      const st = await api("/train/status");
      if (st.state === "none") { $("#status").innerHTML = `<div class="empty"><b>No runs yet.</b><br>Start one below, or run <code>python scripts/train.py</code>.</div>`; return; }
      viewing = st.run_id;
      const run = await api(`/runs/${encodeURIComponent(viewing)}`);
      const h = run.history, running = ["running", "queued"].includes(st.state);
      const pill = st.state === "finished" ? "good" : st.state === "failed" ? "warn" : "";
      $("#status").innerHTML = `<div class="row">
        <div class="stat"><b>${esc(st.run_id.replace("run_", ""))}</b><span>latest run <span class="pill ${pill}">${esc(st.state)}</span></span></div>
        <div class="stat"><b>${st.epoch ?? 0} / ${st.total_epochs ?? "?"}</b><span>epochs (${esc(st.stage ?? "")})</span></div>
        <div class="stat"><b>${st.lr ? Number(st.lr).toExponential(1) : "n/a"}</b><span>learning rate</span></div>
        <div class="stat"><b>${running ? fmtTime(st.eta_seconds) : fmtTime(st.elapsed_seconds)}</b><span>${running ? "time left" : "total time"}</span></div></div>
        ${st.state === "failed" ? errorBox(new Error((st.error || "Training failed") + " See runs/" + st.run_id + "/error.txt")) : ""}`;
      const labels = h.map((e) => e.epoch);
      const ds = (key, label, color, dash) => ({ label, data: h.map((e) => e[key]), borderColor: color, backgroundColor: color, borderDash: dash });
      const lossData = [ds("train_loss", "Train loss", COLORS.ink), ds("val_loss", "Validation loss", COLORS.red)];
      const accData = [ds("train_species_acc", "Train species", COLORS.ink), ds("val_species_acc", "Validation species", COLORS.green),
        ds("val_genus_acc", "Validation genus", COLORS.blue, [5, 4])];
      if (!lossChart) { lossChart = lineChart($("#c-loss"), labels, lossData); accChart = lineChart($("#c-acc"), labels, accData, 1); }
      else { lossChart.data.labels = labels; lossChart.data.datasets.forEach((d, i) => (d.data = lossData[i].data)); lossChart.update();
        accChart.data.labels = labels; accChart.data.datasets.forEach((d, i) => (d.data = accData[i].data)); accChart.update(); }
      if (!running) clearInterval(pollTimer);
    } catch (e) { $("#status").innerHTML = errorBox(e); }
  }
  $("#form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target), body = { overrides: {}, note: fd.get("note") || "" };
    body.backbone = fd.get("backbone");
    ["data.image_size", "train.epochs_finetune", "train.batch_size"].forEach((k) => (body.overrides[k] = Number(fd.get(k))));
    try {
      const r = await api("/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      $("#start-msg").innerHTML = `<div class="notice good">Started ${esc(r.run_id)}.</div>`;
      lossChart = accChart = null; charts.forEach((c) => c.destroy()); charts = [];
      clearInterval(pollTimer); await refresh(); pollTimer = setInterval(refresh, 3000);
    } catch (e) { $("#start-msg").innerHTML = errorBox(e); }
  });
  await refresh();
  const st = await api("/train/status").catch(() => ({}));
  if (["running", "queued"].includes(st.state)) pollTimer = setInterval(refresh, 3000);
}

/* ------------------------------------------------------------------ Compare */
async function pageCompare() {
  main.innerHTML = `<h1>Compare runs</h1><p class="lede">See how each change affected accuracy. Pick two runs to see exactly what changed.</p><div id="body">Loading...</div>`;
  try {
    const runs = (await api("/runs")).filter((r) => r.state === "finished");
    if (!runs.length) { $("#body").innerHTML = `<div class="empty"><b>No finished runs yet.</b><br>Finish a training run and it will show up here.</div>`; return; }
    const cols = [["id", "Run"], ["backbone", "Backbone"], ["image_size", "Size"], ["epochs", "Epochs"], ["test_genus_top1", "Genus top-1"],
      ["test_species_top1", "Species top-1"], ["test_species_top5", "Species top-5"], ["test_macro_f1", "Macro F1"]];
    let sortKey = "test_species_top1", dir = -1;
    const table = () => {
      const rows = [...runs].sort((a, b) => ((a[sortKey] ?? -1) > (b[sortKey] ?? -1) ? 1 : -1) * dir);
      return `<div class="scroll"><table><thead><tr>${cols.map(([k, l]) => `<th class="sortable ${k.startsWith("test") ? "r" : ""}" data-k="${k}" aria-sort="${k === sortKey ? (dir > 0 ? "ascending" : "descending") : "none"}">${l}</th>`).join("")}</tr></thead><tbody>
        ${rows.map((r) => `<tr><td>${esc(r.id.replace("run_", ""))} ${r.active ? '<span class="pill good">active</span>' : ""}<div class="muted small">${esc(r.note)}</div></td><td>${esc(r.backbone)}</td><td class="r">${r.image_size ?? ""}</td><td class="r">${r.epochs}</td>
        <td class="r">${pct(r.test_genus_top1)}</td><td class="r">${pct(r.test_species_top1)}</td><td class="r">${pct(r.test_species_top5)}</td><td class="r">${fix(r.test_macro_f1)}</td></tr>`).join("")}</tbody></table></div>`;
    };
    $("#body").innerHTML = `<div id="tbl"></div>
      <h2>Accuracy by run</h2><div class="chart-box"><canvas id="c-cmp"></canvas></div>
      <h2>What changed</h2>
      <form class="inline"><label>Earlier run<select id="ra">${runs.map((r) => `<option value="${esc(r.id)}">${esc(r.id.replace("run_", ""))}</option>`).join("")}</select></label>
      <label>Later run<select id="rb">${runs.map((r, i) => `<option value="${esc(r.id)}" ${i === 0 ? "selected" : ""}>${esc(r.id.replace("run_", ""))}</option>`).join("")}</select></label>
      <button type="button" id="go">Compare</button></form><div id="diff"></div>`;
    const draw = () => { $("#tbl").innerHTML = table(); document.querySelectorAll("th.sortable").forEach((th) => th.onclick = () => { dir = th.dataset.k === sortKey ? -dir : -1; sortKey = th.dataset.k; draw(); }); };
    draw();
    const ordered = [...runs].reverse();
    charts.push(new Chart($("#c-cmp"), { type: "bar", data: { labels: ordered.map((r) => `${r.backbone} (${r.id.slice(4, 16)})`),
      datasets: [{ label: "Genus top-1", data: ordered.map((r) => r.test_genus_top1), backgroundColor: COLORS.blue },
        { label: "Species top-1", data: ordered.map((r) => r.test_species_top1), backgroundColor: COLORS.ink },
        { label: "Species top-5", data: ordered.map((r) => r.test_species_top5), backgroundColor: COLORS.green }] },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 1 } }, plugins: { legend: { position: "bottom" } } } }));
    $("#go").onclick = async () => {
      try {
        const d = await api(`/runs/${encodeURIComponent($("#ra").value)}/diff/${encodeURIComponent($("#rb").value)}`);
        const arrow = (x) => (x == null ? "" : x > 0 ? "up" : x < 0 ? "down" : "same");
        $("#diff").innerHTML = `<h3>Settings that differ</h3>${d.config_changes.length ? `<div class="scroll"><table><thead><tr><th>Setting</th><th>Earlier</th><th>Later</th></tr></thead><tbody>${d.config_changes.map((c) => `<tr><td>${esc(c.key)}</td><td>${esc(c.a)}</td><td>${esc(c.b)}</td></tr>`).join("")}</tbody></table></div>` : `<p class="muted">Same settings.</p>`}
          <h3>Metric change</h3><div class="scroll"><table><thead><tr><th>Metric</th><th class="r">Earlier</th><th class="r">Later</th><th class="r">Change</th></tr></thead><tbody>${d.metric_delta.map((m) => `<tr><td>${esc(m.metric)}</td><td class="r">${fix(m.a)}</td><td class="r">${fix(m.b)}</td><td class="r">${m.delta == null ? "n/a" : (m.delta > 0 ? "+" : "") + m.delta.toFixed(3)} ${arrow(m.delta)}</td></tr>`).join("")}</tbody></table></div>`;
      } catch (e) { $("#diff").innerHTML = errorBox(e); }
    };
  } catch (e) { $("#body").innerHTML = errorBox(e); }
}

/* ------------------------------------------------------------------ Evaluation */
async function pageEvaluation() {
  main.innerHTML = `<h1>Evaluation</h1><p class="lede">Results on images the model never trained on. Images from one specimen are never split between training and test.</p><div id="body">Loading...</div>`;
  try {
    const models = await api("/models");
    if (!models.length) { $("#body").innerHTML = `<div class="empty"><b>No trained model yet.</b><br>Train one first.</div>`; return; }
    const pick = models.find((m) => m.active) || models[0];
    $("#body").innerHTML = `<label style="max-width:24rem">Run<select id="run">${models.map((m) => `<option value="${esc(m.id)}" ${m.id === pick.id ? "selected" : ""}>${esc(m.id.replace("run_", ""))}${m.active ? " (active)" : ""}</option>`).join("")}</select></label><div id="ev"></div>`;
    $("#run").onchange = () => load($("#run").value);
    load(pick.id);
  } catch (e) { $("#body").innerHTML = errorBox(e); }

  async function load(id) {
    resetCharts();
    try {
      const run = await api(`/runs/${encodeURIComponent(id)}`);
      const t = run.evaluation.test, m = run.metrics;
      if (!t) { $("#ev").innerHTML = `<div class="notice warn">${esc(cleanReport(run.report) || "This run has no test results.")}</div>`; return; }
      const tg = m.targets, goodG = t.genus_top1 >= tg.genus_top1, goodS = t.species_top1 >= tg.species_top1;
      const at = t.at_threshold;
      $("#ev").innerHTML = `
        <div class="row" style="margin-top:1.5rem">
          <div class="stat ${goodG ? "good" : "bad"}"><b>${pct(t.genus_top1)}</b><span>genus top-1 (target ${pct(tg.genus_top1, 0)})</span></div>
          <div class="stat ${goodS ? "good" : "bad"}"><b>${pct(t.species_top1)}</b><span>species top-1 (target ${pct(tg.species_top1, 0)})</span></div>
          <div class="stat"><b>${pct(t.species_top5)}</b><span>species top-5</span></div>
          <div class="stat"><b>${fix(t.macro_f1_species)}</b><span>macro F1</span></div>
          <div class="stat"><b>${fix(t.ece_species)}</b><span>calibration error (lower is better)</span></div>
          <div class="stat"><b>${t.n_images}</b><span>test images</span></div></div>
        ${at && at.accuracy_on_answered != null ? `<p class="muted">At the abstention threshold of ${fix(at.threshold, 2)} the model answers ${pct(at.coverage, 0)} of images with ${pct(at.accuracy_on_answered)} accuracy and marks ${at.abstained} as uncertain.</p>` : ""}
        <div class="notice ${m.target_met ? "good" : "warn"}"><pre style="white-space:pre-wrap;font:inherit;margin:0">${esc(cleanReport(run.report))}</pre></div>
        <h2>Genus confusion</h2><p class="muted small">Rows are the true genus, columns the predicted one. Darker means more images. Hover a cell for details.</p><div class="scroll"><div id="cm"></div></div>
        <h2>Most confused species pairs</h2><label style="max-width:20rem">Filter by true genus<select id="gf"><option value="">All genera</option>${t.confusion_genus.labels.map((g) => `<option>${esc(g)}</option>`).join("")}</select></label><div id="pairs"></div>
        <h2>Weakest species</h2><div class="scroll tall"><table><thead><tr><th>Species</th><th class="r">Recall</th><th class="r">Precision</th><th class="r">F1</th><th class="r">Test images</th><th class="r">Train images</th></tr></thead><tbody>
          ${t.per_class.slice(0, 80).map((c) => `<tr><td>${sci(c.name)}</td><td class="r">${pct(c.recall, 0)}</td><td class="r">${pct(c.precision, 0)}</td><td class="r">${fix(c.f1, 2)}</td><td class="r">${c.support}</td><td class="r">${c.n_train}</td></tr>`).join("")}</tbody></table></div>`;
      drawMatrix(t.confusion_genus);
      const drawPairs = () => {
        const g = $("#gf").value;
        const rows = t.confusion_species.filter((c) => c.true !== c.pred && (!g || c.true_genus === g)).slice(0, 25);
        $("#pairs").innerHTML = rows.length ? `<div class="scroll"><table><thead><tr><th>True species</th><th>Predicted as</th><th class="r">Images</th></tr></thead><tbody>${rows.map((c) => `<tr><td>${sci(c.true.replace(/_/g, " "))}</td><td>${sci(c.pred.replace(/_/g, " "))}${c.true_genus === c.pred_genus ? ' <span class="pill">same genus</span>' : ""}</td><td class="r">${c.count}</td></tr>`).join("")}</tbody></table></div>` : `<p class="muted">No confusions here.</p>`;
      };
      $("#gf").onchange = drawPairs; drawPairs();
    } catch (e) { $("#ev").innerHTML = errorBox(e); }
  }
  function resetCharts() { charts.forEach((c) => c.destroy()); charts = []; }
  function drawMatrix({ labels, matrix }) {
    const n = labels.length, box = $("#cm"), max = Math.max(1, ...matrix.flat());
    const size = n > 60 ? 9 : n > 30 ? 12 : 18;
    box.className = "cm"; box.style.gridTemplateColumns = `repeat(${n}, ${size}px)`;
    box.innerHTML = matrix.map((row, i) => row.map((v, j) => {
      const a = v ? 0.12 + 0.88 * (v / max) : 0, ok = i === j;
      const bg = v ? (ok ? `rgba(63,125,92,${a})` : `rgba(184,52,75,${a})`) : "#fff";
      return `<div style="width:${size}px;height:${size}px;background:${bg}" data-t="${esc(labels[i])} predicted as ${esc(labels[j])}: ${v}"></div>`;
    }).join("")).join("");
    const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; document.body.appendChild(tip);
    box.onmousemove = (e) => { const t = e.target.dataset?.t; tip.hidden = !t; if (t) { tip.textContent = t; tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY + 12 + "px"; } };
    box.onmouseleave = () => (tip.hidden = true);
    window.addEventListener("hashchange", () => tip.remove(), { once: true });
  }
}

/* ------------------------------------------------------------------ Dataset */
async function pageDataset() {
  main.innerHTML = `<h1>Dataset</h1><p class="lede">What the model learns from. Species with few images or few specimens are flagged because they are the hardest to identify reliably.</p><div id="body">Loading...</div>`;
  try {
    const d = await api("/dataset/stats"), t = d.totals;
    $("#body").innerHTML = `
      <div class="row"><div class="stat"><b>${t.images}</b><span>images</span></div><div class="stat"><b>${t.genera}</b><span>genera</span></div>
        <div class="stat"><b>${t.species}</b><span>species</span></div>
        <div class="stat ${t.low_data_species ? "bad" : ""}"><b>${t.low_data_species}</b><span>species flagged as low data</span></div>
        <div class="stat"><b>${t.split_images.train} / ${t.split_images.val} / ${t.split_images.test}</b><span>train / validation / test images</span></div></div>
      <h2>Images per species</h2><p class="muted small">A long left side means many species have only a few images.</p><div class="chart-box"><canvas id="c-hist"></canvas></div>
      <h2>Species to collect more of</h2><div class="scroll tall"><table><thead><tr><th>Species</th><th class="r">Images</th><th class="r">Specimen records</th><th class="r">Test images</th></tr></thead><tbody>
        ${d.species.filter((s) => s.low_data).slice(0, 100).map((s) => `<tr><td>${sci(s.name)}</td><td class="r">${s.images}</td><td class="r">${s.records}</td><td class="r">${s.test}</td></tr>`).join("") || '<tr><td colspan="4">None. Every species has at least 5 images and 3 specimen records.</td></tr>'}</tbody></table></div>
      <h2>Browse images</h2><form class="inline"><label>Genus<select id="g">${d.genera.map((g) => `<option>${esc(g.genus)}</option>`).join("")}</select></label>
        <label>Species<select id="s"></select></label></form><div id="gal" class="gallery" style="margin-top:1rem"></div>`;
    const buckets = {}; d.species.forEach((s) => { const b = Math.min(s.images, 30); buckets[b] = (buckets[b] || 0) + 1; });
    const keys = Array.from({ length: 30 }, (_, i) => i + 1);
    charts.push(new Chart($("#c-hist"), { type: "bar", data: { labels: keys.map((k) => (k === 30 ? "30+" : k)),
      datasets: [{ label: "Number of species", data: keys.map((k) => buckets[k] || 0), backgroundColor: keys.map((k) => (k < 5 ? COLORS.red : COLORS.ink)) }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "Images in the species" } }, y: { title: { display: true, text: "Species" } } } } }));
    const fillSpecies = () => { const g = $("#g").value; $("#s").innerHTML = d.species.filter((s) => s.genus === g).map((s) => `<option value="${esc(s.species)}">${esc(s.name)}</option>`).join(""); gallery(); };
    const gallery = async () => {
      const sp = $("#s").value; if (!sp) return;
      try { const imgs = await api(`/dataset/gallery/${encodeURIComponent(sp)}`);
        $("#gal").innerHTML = imgs.map((i) => `<figure><img loading="lazy" src="${esc(i.url)}" alt="${esc(i.caption || sp)}"><figcaption>${esc(i.caption || "")} <span class="pill">${esc(i.split)}</span>${i.page_url ? ` <a href="${esc(i.page_url)}" target="_blank" rel="noopener">source</a>` : ""}</figcaption></figure>`).join("");
      } catch (e) { $("#gal").innerHTML = errorBox(e); }
    };
    $("#g").onchange = fillSpecies; $("#s").onchange = gallery; fillSpecies();
  } catch (e) { $("#body").innerHTML = errorBox(e); }
}

/* ------------------------------------------------------------------ Model versions */
async function pageModels() {
  main.innerHTML = `<h1>Model versions</h1><p class="lede">Every trained model is kept. Activate any version to use it for identification, or roll back to an earlier one.</p><div id="body">Loading...</div>`;
  const load = async () => {
    try {
      const models = await api("/models");
      if (!models.length) { $("#body").innerHTML = `<div class="empty"><b>No trained models yet.</b><br>Train one on the Training page.</div>`; return; }
      $("#body").innerHTML = `<div class="scroll"><table><thead><tr><th>Version</th><th>Backbone</th><th class="r">Genus</th><th class="r">Species</th><th>Target</th><th></th></tr></thead><tbody>
        ${models.map((m) => `<tr><td>${esc(m.id.replace("run_", ""))}<div class="muted small">${esc(m.note)}</div></td><td>${esc(m.backbone)}</td><td class="r">${pct(m.test_genus_top1)}</td><td class="r">${pct(m.test_species_top1)}</td>
          <td>${m.target_met ? '<span class="pill good">met</span>' : '<span class="pill warn">not met</span>'}</td>
          <td>${m.active ? '<span class="pill good">active</span>' : `<button data-id="${esc(m.id)}">Use this version</button>`}</td></tr>`).join("")}</tbody></table></div><div id="msg" aria-live="polite"></div>`;
      document.querySelectorAll("button[data-id]").forEach((b) => (b.onclick = async () => {
        try { await api(`/models/${encodeURIComponent(b.dataset.id)}/activate`, { method: "POST" }); await load(); await refreshActive(); }
        catch (e) { $("#msg").innerHTML = errorBox(e); }
      }));
    } catch (e) { $("#body").innerHTML = errorBox(e); }
  };
  load();
}

/* ------------------------------------------------------------------ About */
function pageAbout() {
  main.innerHTML = `<h1>About and credits</h1>
    <p class="lede">Pollen ID names a pollen grain from a light-microscopy image: first the genus, then the species within that genus.</p>
    <h2>Data source</h2><p style="max-width:62ch">Images and species names come from <a href="https://www.paldat.org" target="_blank" rel="noopener">PalDat - Palynological Database</a>, and every image belongs to its photographer and to PalDat. They are used here for personal study only and must not be redistributed. Cite PalDat in anything you present.</p>
    <h2>How to read the results</h2><p style="max-width:62ch">When the model is not confident it says so instead of guessing. An uncertain answer can mean a look-alike species or a species the model has never seen. Check those against a reference before relying on them.</p>`;
}

/* ------------------------------------------------------------------ routing */
const routes = { predict: pageIdentify, training: pageTraining, compare: pageCompare, evaluation: pageEvaluation, dataset: pageDataset, models: pageModels, about: pageAbout };
async function refreshActive() {
  try { const h = await api("/health"); $("#active-model").textContent = h.active_model ? "Model: " + h.active_model.replace("run_", "") : "No model trained yet"; }
  catch (_) { $("#active-model").textContent = "Server not reachable"; }
}
function route() {
  resetPage();
  document.querySelectorAll("[data-route]").forEach((a) => a.removeAttribute("aria-current"));
  const name = (location.hash.replace(/^#\//, "") || "predict");
  const fn = routes[name] || routes.predict;
  const link = document.querySelector(`[data-route="${routes[name] ? name : "predict"}"]`);
  if (link) link.setAttribute("aria-current", "page");
  fn();
  main.focus({ preventScroll: true });
}
window.addEventListener("hashchange", route);
route();
refreshActive();
