// LoRAForge Web Control Plane Client
const state = {
  models: {},
  gpus: {},
  presets: {},
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
    const [modelsRes, hwRes, presetsRes] = await Promise.all([
      fetch('/api/models'),
      fetch('/api/hardware'),
      fetch('/api/presets'),
    ]);

    state.models = await modelsRes.json();
    const hwData = await hwRes.json();
    state.gpus = hwData.presets;
    state.presets = await presetsRes.json();

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
    'modelSelect', 'gpuSelect', 'gpuCountSelect', 'strategySelect',
    'seqLenSlider', 'batchSizeSlider', 'gradAccSlider', 'rankSlider',
    'quantSelect', 'optSelect', 'gradCkptCheck', 'flashAttnCheck'
  ];

  inputs.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      syncLabels();
      triggerEstimate();
    });
  });

  document.getElementById('presetSelect').addEventListener('change', (e) => {
    const pKey = e.target.value;
    if (state.presets[pKey]) {
      const p = state.presets[pKey];
      if (document.getElementById('modelSelect').querySelector(`option[value="${p.recommended_model}"]`)) {
        document.getElementById('modelSelect').value = p.recommended_model;
      }
      document.getElementById('seqLenSlider').value = p.training.max_seq_length;
      document.getElementById('batchSizeSlider').value = p.training.per_device_train_batch_size;
      document.getElementById('gradAccSlider').value = p.training.gradient_accumulation_steps;
      document.getElementById('rankSlider').value = p.lora.r;
      document.getElementById('quantSelect').value = p.quantization.quant_type;
      document.getElementById('optSelect').value = p.training.optimizer;
      syncLabels();
      triggerEstimate();
    }
  });

  document.getElementById('btnAutoTune').addEventListener('click', runAutoTune);
  document.getElementById('btnStartTrain').addEventListener('click', launchTrainingJob);
  document.getElementById('btnExportInferOps').addEventListener('click', exportToInferOps);
  document.getElementById('btnDownloadYaml').addEventListener('click', downloadYamlConfig);
  document.getElementById('btnCloudExport').addEventListener('click', exportCloudPackage);
  document.getElementById('btnMergeAdapters')?.addEventListener('click', mergeAdapters);
}

function syncLabels() {
  document.getElementById('seqLenVal').textContent = document.getElementById('seqLenSlider').value;
  document.getElementById('batchSizeVal').textContent = document.getElementById('batchSizeSlider').value;
  document.getElementById('gradAccVal').textContent = document.getElementById('gradAccSlider').value;
  document.getElementById('rankVal').textContent = document.getElementById('rankSlider').value;
}

function getPayload() {
  return {
    model_name: document.getElementById('modelSelect').value,
    gpu_name: document.getElementById('gpuSelect').value,
    num_gpus: parseInt(document.getElementById('gpuCountSelect').value),
    distributed_strategy: document.getElementById('strategySelect').value,
    seq_len: parseInt(document.getElementById('seqLenSlider').value),
    batch_size: parseInt(document.getElementById('batchSizeSlider').value),
    grad_acc: parseInt(document.getElementById('gradAccSlider').value),
    rank: parseInt(document.getElementById('rankSlider').value),
    quant: document.getElementById('quantSelect').value,
    optimizer: document.getElementById('optSelect').value,
    gradient_checkpointing: document.getElementById('gradCkptCheck').checked,
    flash_attention: document.getElementById('flashAttnCheck').checked,
  };
}

async function triggerEstimate() {
  const payload = getPayload();
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
  // 1. Feasibility badge & hero
  const badge = document.getElementById('statusBadge');
  badge.textContent = report.status;
  badge.className = 'status-tag status-' + report.status.toLowerCase().replace('_', '-');

  document.getElementById('peakVramDisplay').textContent = `${report.peak_vram_mb.toFixed(1)} MB`;
  document.getElementById('totalVramDisplay').textContent = `${report.total_gpu_vram_mb} MB`;
  document.getElementById('headroomDisplay').textContent = `${report.headroom_mb.toFixed(1)} MB (${(100 - report.utilization_percentage).toFixed(1)}% free)`;

  if (report.num_gpus > 1) {
    document.getElementById('clusterVramDisplay').textContent =
      `Topology: ${report.num_gpus}x GPUs (${report.distributed_strategy.toUpperCase()}) | Cluster VRAM: ${report.total_cluster_vram_mb} MB | Global Batch: ${report.effective_batch_size}`;
  } else {
    document.getElementById('clusterVramDisplay').textContent = `Single GPU | Effective Batch: ${report.effective_batch_size}`;
  }

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
    { key: 'comm', val: bd.communication_buffers_mb, color: '#14b8a6' },
    { key: 'frag', val: bd.fragmentation_buffer_mb, color: '#475569' },
  ];

  const barContainer = document.getElementById('stackedBar');
  barContainer.innerHTML = '';
  barSegments.forEach(s => {
    if (s.val <= 0) return;
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
    <tr><td>Communication Rings</td><td>Multi-GPU AllReduce & gathering buffers</td><td class="numeric">${bd.communication_buffers_mb.toFixed(1)} MB</td></tr>
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

    let statusText = `Step ${metric.step} | Loss: ${metric.loss.toFixed(4)}`;
    if (metric.val_loss !== null && metric.val_loss !== undefined) {
      statusText += ` | Val Loss: ${metric.val_loss.toFixed(4)}`;
    }
    statusText += ` | VRAM: ${metric.vram_used_mb.toFixed(1)} MB`;
    document.getElementById('trainingStepText').textContent = statusText;

    if (metric.perplexity !== null && metric.perplexity !== undefined) {
      const pplBadge = document.getElementById('pplBadge');
      pplBadge.style.display = 'inline-block';
      pplBadge.textContent = `PPL: ${metric.perplexity.toFixed(2)}`;
    }

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

  // 1. Draw Train Loss (Cyan)
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

  // 2. Draw Val Loss (Emerald)
  const valPoints = state.lossHistory.filter(pt => pt.val_loss !== null && pt.val_loss !== undefined);
  if (valPoints.length > 0) {
    ctx.strokeStyle = '#10b981';
    ctx.fillStyle = '#10b981';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    let first = true;
    state.lossHistory.forEach((pt, i) => {
      if (pt.val_loss !== null && pt.val_loss !== undefined) {
        const x = 40 + (i / (steps - 1)) * (w - 70);
        const y = (h - 30) - ((pt.val_loss - minLoss) / (maxLoss - minLoss)) * (h - 50);
        if (first) {
          ctx.moveTo(x, y);
          first = false;
        } else {
          ctx.lineTo(x, y);
        }
      }
    });
    ctx.stroke();

    // Draw dots for validation evaluations
    state.lossHistory.forEach((pt, i) => {
      if (pt.val_loss !== null && pt.val_loss !== undefined) {
        const x = 40 + (i / (steps - 1)) * (w - 70);
        const y = (h - 30) - ((pt.val_loss - minLoss) / (maxLoss - minLoss)) * (h - 50);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }
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

async function downloadYamlConfig() {
  const payload = getPayload();
  try {
    const res = await fetch('/api/config/yaml', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    const blob = new Blob([data.yaml], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'loraforge.yaml';
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Failed to download YAML:', err);
  }
}

async function exportCloudPackage() {
  const payload = getPayload();
  try {
    const res = await fetch('/api/export/cloud', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const pkg = await res.json();
    alert(`Cloud package generated in ${pkg.dockerfile_path}`);
  } catch (err) {
    console.error('Failed to export cloud package:', err);
  }
}

async function mergeAdapters() {
  const inputStr = document.getElementById('mergeAdaptersInput').value;
  const paths = inputStr.split(',').map(s => s.trim()).filter(Boolean);
  if (paths.length === 0) {
    alert('Please enter at least one adapter path (e.g. ./runs/adapter_code, ./runs/adapter_math)');
    return;
  }

  const method = document.getElementById('mergeMethodSelect').value;
  const density = parseFloat(document.getElementById('mergeDensityInput').value) || 0.2;

  const btn = document.getElementById('btnMergeAdapters');
  btn.disabled = true;
  btn.textContent = 'Merging...';

  try {
    const res = await fetch('/api/export/merge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        adapter_paths: paths,
        method: method,
        density: density,
        drop_rate: Math.max(0.0, 1.0 - density),
        target_rank: 16,
        output_dir: './merged_adapters',
      }),
    });
    const report = await res.json();
    if (!res.ok) {
      alert(`Merge error: ${report.detail || 'Failed to merge'}`);
      return;
    }
    document.getElementById('mergeResultSummary').textContent = report.summary;
    document.getElementById('mergeCliCmd').textContent =
      `# Ready for InferOps serving!\ninferops serve --manifest ${report.output_file} --engine vllm --port 8000`;
    document.getElementById('mergeResultCard').style.display = 'block';
  } catch (err) {
    console.error('Failed to merge adapters:', err);
    alert('Merge request failed.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Merge Adapters';
  }
}

window.addEventListener('DOMContentLoaded', init);
