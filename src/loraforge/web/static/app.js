// LoRAForge Web Control Plane Client
const state = {
  models: {},
  gpus: {},
  activeReport: null,
  lossHistory: [],
  ws: null,
};

async function init() {
  await loadOptions();
  setupEventListeners();
  triggerEstimate();
  setupWebSocket();
}

async function loadOptions() {
  try {
    const [modelsRes, hwRes] = await Promise.all([
      fetch('/api/models'),
      fetch('/api/hardware'),
    ]);

    state.models = await modelsRes.json();
    const hwData = await hwRes.json();
    state.gpus = hwData.presets;

    const modelSelect = document.getElementById('modelSelect');
    modelSelect.innerHTML = '';
    for (const [key, m] of Object.entries(state.models)) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = `${key.split('/').pop()} (${(m.num_parameters / 1e9).toFixed(1)}B)`;
      modelSelect.appendChild(opt);
    }

    const gpuSelect = document.getElementById('gpuSelect');
    gpuSelect.innerHTML = '';
    for (const [key, g] of Object.entries(state.gpus)) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = `${g.name} (${Math.round(g.total_vram_mb / 1024)}GB)`;
      gpuSelect.appendChild(opt);
    }
  } catch (err) {
    console.error('Failed to load options:', err);
  }
}

function setupEventListeners() {
  const inputs = [
    'modelSelect', 'gpuSelect', 'seqLenSlider', 'batchSizeSlider',
    'gradAccSlider', 'rankSlider', 'quantSelect', 'optSelect',
    'gradCkptCheck', 'flashAttnCheck'
  ];

  inputs.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      syncLabels();
      triggerEstimate();
    });
  });

  document.getElementById('btnAutoTune').addEventListener('click', runAutoTune);
  document.getElementById('btnStartTrain').addEventListener('click', launchTrainingJob);
  document.getElementById('btnExportInferOps').addEventListener('click', exportToInferOps);
}

function syncLabels() {
  document.getElementById('seqLenVal').textContent = document.getElementById('seqLenSlider').value;
  document.getElementById('batchSizeVal').textContent = document.getElementById('batchSizeSlider').value;
  document.getElementById('gradAccVal').textContent = document.getElementById('gradAccSlider').value;
  document.getElementById('rankVal').textContent = document.getElementById('rankSlider').value;
}

async function triggerEstimate() {
  const payload = {
    model_name: document.getElementById('modelSelect').value,
    gpu_name: document.getElementById('gpuSelect').value,
    seq_len: parseInt(document.getElementById('seqLenSlider').value),
    batch_size: parseInt(document.getElementById('batchSizeSlider').value),
    grad_acc: parseInt(document.getElementById('gradAccSlider').value),
    rank: parseInt(document.getElementById('rankSlider').value),
    quant: document.getElementById('quantSelect').value,
    optimizer: document.getElementById('optSelect').value,
    gradient_checkpointing: document.getElementById('gradCkptCheck').checked,
    flash_attention: document.getElementById('flashAttnCheck').checked,
  };

  try {
    const res = await fetch('/api/vram/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) return;

    state.activeReport = await res.json();
    renderReport(state.activeReport);
  } catch (err) {
    console.error('Estimate failed:', err);
  }
}

function renderReport(report) {
  // 1. Feasibility badge
  const badge = document.getElementById('statusBadge');
  badge.textContent = report.status;
  badge.className = 'status-tag status-' + report.status.toLowerCase().replace('_', '-');

  document.getElementById('peakVramDisplay').textContent = `${report.peak_vram_mb.toFixed(1)} MB`;
  document.getElementById('totalVramDisplay').textContent = `${report.total_gpu_vram_mb} MB`;
  document.getElementById('headroomDisplay').textContent = `${report.headroom_mb.toFixed(1)} MB (${(100 - report.utilization_percentage).toFixed(1)}% free)`;

  // 2. Stacked Bar Chart
  const bd = report.breakdown;
  const total = report.total_gpu_vram_mb;

  const barSegments = [
    { key: 'cuda', val: bd.cuda_context_mb, color: '#64748b' },
    { key: 'base', val: bd.base_model_weights_mb, color: '#3b82f6' },
    { key: 'qmeta', val: bd.quantization_metadata_mb, color: '#6366f1' },
    { key: 'lora', val: bd.lora_weights_mb, color: '#a855f7' },
    { key: 'grad', val: bd.gradients_mb, color: '#ec4899' },
    { key: 'opt', val: bd.optimizer_states_mb, color: '#f59e0b' },
    { key: 'act', val: bd.activations_mb, color: '#06b6d4' },
    { key: 'frag', val: bd.fragmentation_buffer_mb, color: '#475569' },
  ];

  const barContainer = document.getElementById('stackedBar');
  barContainer.innerHTML = '';
  barSegments.forEach(s => {
    const seg = document.createElement('div');
    seg.className = 'bar-segment';
    const pct = (s.val / total) * 100;
    seg.style.width = `${pct}%`;
    seg.style.backgroundColor = s.color;
    seg.title = `${s.key}: ${s.val.toFixed(1)} MB (${pct.toFixed(1)}%)`;
    barContainer.appendChild(seg);
  });

  // 3. Table Rows
  const tbody = document.getElementById('breakdownTableBody');
  tbody.innerHTML = `
    <tr><td>Base Model Weights</td><td>${report.model_name}</td><td class="numeric">${bd.base_model_weights_mb.toFixed(1)} MB</td></tr>
    <tr><td>Quantization Scales</td><td>Block scale constants</td><td class="numeric">${bd.quantization_metadata_mb.toFixed(1)} MB</td></tr>
    <tr><td>LoRA Adapters</td><td>r=${report.lora_parameters.rank} (${report.lora_parameters.trainable_parameters.toLocaleString()} params)</td><td class="numeric">${bd.lora_weights_mb.toFixed(1)} MB</td></tr>
    <tr><td>Gradient Tensors</td><td>Trainable parameters only</td><td class="numeric">${bd.gradients_mb.toFixed(1)} MB</td></tr>
    <tr><td>Optimizer States</td><td>Moments & Master states</td><td class="numeric">${bd.optimizer_states_mb.toFixed(1)} MB</td></tr>
    <tr><td>Activation Memory</td><td>Forward stashed + intra-block recomputation</td><td class="numeric">${bd.activations_mb.toFixed(1)} MB</td></tr>
    <tr><td>CUDA Context</td><td>Driver baseline overhead</td><td class="numeric">${bd.cuda_context_mb.toFixed(1)} MB</td></tr>
    <tr><td>Allocator Headroom</td><td>PyTorch memory fragmentation</td><td class="numeric">${bd.fragmentation_buffer_mb.toFixed(1)} MB</td></tr>
  `;

  // 4. Recommendations
  const recList = document.getElementById('recommendationList');
  recList.innerHTML = '';
  report.recommendations.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    recList.appendChild(li);
  });
}

async function runAutoTune() {
  const payload = {
    model_name: document.getElementById('modelSelect').value,
    gpu_name: document.getElementById('gpuSelect').value,
    effective_batch: 16,
    target_seq_len: parseInt(document.getElementById('seqLenSlider').value),
  };

  try {
    const res = await fetch('/api/vram/autotune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const rec = await res.json();

    document.getElementById('batchSizeSlider').value = rec.optimal_micro_batch_size;
    document.getElementById('gradAccSlider').value = rec.optimal_grad_accumulation_steps;
    document.getElementById('rankSlider').value = rec.recommended_rank;
    document.getElementById('quantSelect').value = rec.recommended_quantization;
    document.getElementById('optSelect').value = rec.recommended_optimizer;

    syncLabels();
    triggerEstimate();
  } catch (err) {
    console.error('AutoTune failed:', err);
  }
}

async function launchTrainingJob() {
  const btn = document.getElementById('btnStartTrain');
  btn.disabled = true;
  btn.textContent = 'Running Supervisor...';

  state.lossHistory = [];
  drawLossChart();

  const payload = {
    job_name: 'gui-experiment',
    model_name: document.getElementById('modelSelect').value,
    gpu_name: document.getElementById('gpuSelect').value,
    seq_len: parseInt(document.getElementById('seqLenSlider').value),
    batch_size: parseInt(document.getElementById('batchSizeSlider').value),
    grad_acc: parseInt(document.getElementById('gradAccSlider').value),
    rank: parseInt(document.getElementById('rankSlider').value),
    quant: document.getElementById('quantSelect').value,
    optimizer: document.getElementById('optSelect').value,
    total_steps: 40,
  };

  try {
    await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    console.error('Failed to launch job:', err);
  }
}

function setupWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  state.ws = new WebSocket(wsUrl);
  state.ws.onmessage = (event) => {
    const metric = JSON.parse(event.data);
    state.lossHistory.push(metric);
    document.getElementById('trainingStepText').textContent = `Step ${metric.step} | Loss: ${metric.loss.toFixed(4)} | VRAM: ${metric.vram_used_mb.toFixed(1)} MB`;
    drawLossChart();

    if (metric.step >= 40) {
      const btn = document.getElementById('btnStartTrain');
      btn.disabled = false;
      btn.textContent = 'Training Completed (Run Again)';
    }
  };

  state.ws.onclose = () => {
    setTimeout(setupWebSocket, 3000);
  };
}

function drawLossChart() {
  const canvas = document.getElementById('lossChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = 200;

  ctx.clearRect(0, 0, w, h);

  if (state.lossHistory.length < 2) {
    ctx.fillStyle = '#64748b';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for training steps...', w / 2, h / 2);
    return;
  }

  // Draw grid
  ctx.strokeStyle = '#23304e';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(40, 20); ctx.lineTo(40, h - 30);
  ctx.lineTo(w - 20, h - 30);
  ctx.stroke();

  const maxLoss = 3.5;
  const minLoss = 0.0;
  const steps = state.lossHistory.length;

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2.5;
  ctx.beginPath();

  state.lossHistory.forEach((pt, i) => {
    const x = 40 + (i / (steps - 1)) * (w - 70);
    const y = (h - 30) - ((pt.loss - minLoss) / (maxLoss - minLoss)) * (h - 50);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function exportToInferOps() {
  const payload = {
    job_name: 'gui-experiment',
    model_name: document.getElementById('modelSelect').value,
    gpu_name: document.getElementById('gpuSelect').value,
    engine: document.getElementById('inferopsEngineSelect').value,
    output_dir: './exports/inferops-bundle',
  };

  try {
    const res = await fetch('/api/export/inferops', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    document.getElementById('inferopsCliCmd').textContent = data.inferops_command;
    document.getElementById('inferopsExportCard').style.display = 'block';
  } catch (err) {
    console.error('Export failed:', err);
  }
}

window.addEventListener('DOMContentLoaded', init);
