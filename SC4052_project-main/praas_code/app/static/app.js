// PraaS frontend — vanilla JS, talks to the FastAPI gateway.

const $ = (sel) => document.querySelector(sel);

const statusEl    = $("#status");
const results     = $("#results");
const analyzeOut  = $("#analyze-out");
const optimizeOut = $("#optimize-out");
const adaptOut    = $("#adapt-out");
const evaluateOut = $("#evaluate-out");
const fpdbPanel   = $("#fpdb-panel");
const fpdbOut     = $("#fpdb-out");

function setStatus(text) {
  statusEl.textContent = text || "";
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function containsMockOutput(ev) {
  if (!ev || !Array.isArray(ev.scores)) return false;
  return ev.scores.some(s => String(s.output || "").includes("[mock-output#"));
}

function renderAnalyze(a) {
  const rows = Object.entries(a.dimensions).map(([name, d]) => `
    <tr>
      <td>${esc(name)}</td>
      <td><span class="badge ${d.status}">${esc(d.status)}</span></td>
      <td>${esc(d.justification)}</td>
    </tr>
  `).join("");

  analyzeOut.innerHTML = `
    <p><strong>Summary:</strong> ${esc(a.summary)}</p>
    <p><strong>Overall quality:</strong>
       ${(a.overall_quality * 100).toFixed(0)}%
       &middot; prompt hash <code>${esc(a.prompt_hash)}</code></p>
    <table class="dim-table">
      <thead>
        <tr>
          <th>Dimension</th>
          <th>Status</th>
          <th>Justification</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderOptimize(o) {
  optimizeOut.innerHTML = `
    <p><strong>Gradient (step 1):</strong></p>
    <div class="code-block">${esc(o.gradient)}</div>

    <p><strong>Optimised prompt (step 2):</strong></p>
    <div class="code-block">${esc(o.optimised_prompt)}</div>

    <p class="service-note">Iterations run: ${esc(o.iterations_run)}</p>
  `;
}

function renderAdapt(ad) {
  adaptOut.innerHTML = ad.variants.map(v => `
    <div class="variant">
      <div class="variant-head">
        <span>${esc(v.family)}</span>
        <span class="score">${esc(String(v.notes || "").slice(0, 60))}...</span>
      </div>
      <div class="variant-body">
        <div class="code-block">${esc(v.prompt)}</div>
      </div>
    </div>
  `).join("");
}

function renderEvaluate(ev) {
  const items = ev.scores.map(s => {
    const isWin = s.family === ev.winner;
    const rb = s.rubric_breakdown || {
      completion: 0,
      compliance: 0,
      structure: 0,
    };

    return `
      <div class="variant ${isWin ? "winner" : ""}">
        <div class="variant-head">
          <span>${esc(s.family)}</span>
          <span class="score">
            ${Number(s.score).toFixed(2)} / 5 &middot;
            completion ${Number(rb.completion).toFixed(1)} &middot;
            compliance ${Number(rb.compliance).toFixed(1)} &middot;
            structure ${Number(rb.structure).toFixed(1)}
          </span>
        </div>
        <div class="variant-body">
          <div class="code-block">${esc(s.output)}</div>
        </div>
      </div>
    `;
  }).join("");

  const backendNote = containsMockOutput(ev)
    ? `<p class="service-note"><strong>Backend note:</strong> mock / controlled backend detected in outputs.</p>`
    : `<p class="service-note"><strong>Backend note:</strong> no mock marker detected in outputs.</p>`;

  evaluateOut.innerHTML = `
    ${backendNote}
    ${items}
    <p><strong>Winner:</strong> ${esc(ev.winner)} —
       <em>${esc(ev.explanation)}</em></p>
  `;
}

async function runPipeline() {
  const prompt = $("#prompt").value.trim();
  const task = $("#task").value.trim();

  if (!prompt) {
    setStatus("Please enter a prompt.");
    return;
  }

  setStatus("Running pipeline... (Analyzer → Optimizer → Adapter → Evaluator)");
  $("#run").disabled = true;
  $("#run-analyze").disabled = true;

  try {
    const r = await fetch("/pipeline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        task_description: task,
      }),
    });

    if (!r.ok) {
      throw new Error(`HTTP ${r.status}`);
    }

    const data = await r.json();

    results.classList.remove("hidden");
    renderAnalyze(data.analyze);
    renderOptimize(data.optimize);
    renderAdapt(data.adapt);
    renderEvaluate(data.evaluate);

    if (containsMockOutput(data.evaluate)) {
      setStatus("Done. Mock / controlled backend still appears to be in use.");
    } else {
      setStatus("Done. Outputs do not show the mock marker.");
    }
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    $("#run").disabled = false;
    $("#run-analyze").disabled = false;
  }
}

async function runAnalyzeOnly() {
  const prompt = $("#prompt").value.trim();
  const task = $("#task").value.trim();

  if (!prompt) {
    setStatus("Please enter a prompt.");
    return;
  }

  setStatus("Analyzing...");

  try {
    const r = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        task_description: task,
      }),
    });

    if (!r.ok) {
      throw new Error(`HTTP ${r.status}`);
    }

    const data = await r.json();

    results.classList.remove("hidden");
    renderAnalyze(data);
    optimizeOut.innerHTML = '<p class="service-note">(Not run — use full pipeline.)</p>';
    adaptOut.innerHTML = '<p class="service-note">(Not run — use full pipeline.)</p>';
    evaluateOut.innerHTML = '<p class="service-note">(Not run — use full pipeline.)</p>';
    setStatus("Done.");
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

async function showFpdb(e) {
  if (e) e.preventDefault();

  setStatus("Fetching FPDB stats...");

  try {
    const r = await fetch("/fpdb/stats");
    if (!r.ok) {
      throw new Error(`HTTP ${r.status}`);
    }

    const s = await r.json();

    const rows = Object.entries(s.missing_dimension_pct || {})
      .sort((a, b) => b[1] - a[1])
      .map(([dim, pct]) => `
        <tr>
          <td>${esc(dim)}</td>
          <td>${(pct * 100).toFixed(1)}%</td>
        </tr>
      `)
      .join("");

    fpdbOut.innerHTML = `
      <p>Total prompts analysed: <strong>${esc(s.total_prompts_analysed)}</strong></p>
      <p>Top weaknesses: <strong>${esc((s.top_weaknesses || []).join(", ") || "—")}</strong></p>
      <table class="dim-table">
        <thead>
          <tr>
            <th>Dimension</th>
            <th>% of prompts missing this</th>
          </tr>
        </thead>
        <tbody>
          ${rows || '<tr><td colspan="2">No data yet — run a few prompts first.</td></tr>'}
        </tbody>
      </table>
    `;

    fpdbPanel.classList.remove("hidden");
    setStatus("");
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

$("#run").addEventListener("click", runPipeline);
$("#run-analyze").addEventListener("click", runAnalyzeOnly);
$("#stats-link").addEventListener("click", showFpdb);
