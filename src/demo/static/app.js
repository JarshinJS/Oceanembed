/**
 * OceanEmbed Interactive Demo Frontend Application
 * Problem Statement ID: 26066
 */

(function () {
  'use strict';

  // =========================================================================
  // CONSTANTS & CONFIGURATION
  // =========================================================================
  const DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];
  const DEFAULT_DATE = '2020-03-20';
  const DEFAULT_DEPTH = 100;
  const DEFAULT_LAT = 14.625;
  const DEFAULT_LON = 86.875;

  const LATS = [];
  for (let i = 0; i < 24; i++) LATS.push(12.125 + i * 0.25);
  const LONS = [];
  for (let j = 0; j < 32; j++) LONS.push(85.125 + j * 0.25);

  // Colormap definition (Ocean thermal: dark blue -> cyan -> yellow -> deep red)
  const COLORMAP_STOPS = [
    { t: 0.00, r: 10,  g: 40,  b: 95 },   // Deep blue
    { t: 0.20, r: 25,  g: 100, b: 180 },  // Blue
    { t: 0.40, r: 0,   g: 200, b: 220 },  // Cyan
    { t: 0.60, r: 80,  g: 220, b: 120 },  // Green
    { t: 0.75, r: 250, g: 210, b: 50 },   // Yellow
    { t: 0.90, r: 240, g: 100, b: 30 },   // Orange
    { t: 1.00, r: 180, g: 15,  b: 35 },   // Deep Red
  ];

  // =========================================================================
  // STATE
  // =========================================================================
  const state = {
    dates: [],
    currentDate: DEFAULT_DATE,
    currentDepth: DEFAULT_DEPTH,
    currentChannel: 'sst',
    probeLat: DEFAULT_LAT,
    probeLon: DEFAULT_LON,
    prediction: null,     // [15, 24, 32]
    surface: null,        // 7 channels
    profile: null,        // vertical profile data
    metrics: null,
    predictionsCache: {}, // date -> prediction object
    surfaceCache: {},     // date -> surface object
    loading: false,
  };

  // =========================================================================
  // DOM ELEMENTS
  // =========================================================================
  const dom = {
    dateSelect: document.getElementById('date-select'),
    dateSlider: document.getElementById('date-slider'),
    splitPill: document.getElementById('split-pill'),
    argoPill: document.getElementById('argo-pill'),
    depthDisplay: document.getElementById('depth-display'),
    regimePill: document.getElementById('regime-pill'),
    depthBtnGroup: document.getElementById('depth-btn-group'),
    runInferenceBtn: document.getElementById('run-inference-btn'),
    latencyVal: document.getElementById('latency-val'),
    cacheVal: document.getElementById('cache-val'),
    mapSubtitle: document.getElementById('map-subtitle'),
    cellProbeBadge: document.getElementById('cell-probe-badge'),
    cellTempBadge: document.getElementById('cell-temp-badge'),
    subsurfaceCanvas: document.getElementById('subsurface-canvas'),
    mapCrosshair: document.getElementById('map-crosshair'),
    mapTooltip: document.getElementById('map-tooltip'),
    cbarMin: document.getElementById('cbar-min'),
    cbarMax: document.getElementById('cbar-max'),
    surfaceTabs: document.getElementById('surface-tabs'),
    surfaceCanvas: document.getElementById('surface-canvas'),
    surfaceCoverageVal: document.getElementById('surface-coverage-val'),
    surfaceCbarMin: document.getElementById('surface-cbar-min'),
    surfaceCbarMax: document.getElementById('surface-cbar-max'),
    surfaceCbarUnits: document.getElementById('surface-cbar-units'),
    profileSubtitle: document.getElementById('profile-subtitle'),
    profileCanvas: document.getElementById('profile-canvas'),
    profileTooltip: document.getElementById('profile-tooltip'),
    argoCollocationStatus: document.getElementById('argo-collocation-status'),
    argoLegendRow: document.getElementById('argo-legend-row'),
    modelStatusText: document.getElementById('model-status-text'),
  };

  // =========================================================================
  // UTILITIES
  // =========================================================================
  function getColorFromScale(value, min, max, stops = COLORMAP_STOPS) {
    if (value === null || value === undefined || isNaN(value)) {
      return '#1e293b'; // Land / masked cell color
    }
    let norm = (value - min) / Math.max(max - min, 1e-5);
    norm = Math.max(0, Math.min(1, norm));

    let i = 0;
    while (i < stops.length - 1 && stops[i + 1].t < norm) i++;
    const s0 = stops[i];
    const s1 = stops[Math.min(i + 1, stops.length - 1)];

    const span = s1.t - s0.t || 1e-5;
    const factor = (norm - s0.t) / span;

    const r = Math.round(s0.r + factor * (s1.r - s0.r));
    const g = Math.round(s0.g + factor * (s1.g - s0.g));
    const b = Math.round(s0.b + factor * (s1.b - s0.b));
    return `rgb(${r}, ${g}, ${b})`;
  }

  function getRegimeName(depth) {
    if (depth <= 20) return 'MIXED LAYER (0–20m)';
    if (depth <= 200) return 'THERMOCLINE (50–200m)';
    return 'DEEP OCEAN (>200m)';
  }

  // =========================================================================
  // INITIALIZATION
  // =========================================================================
  async function init() {
    buildDepthButtons();
    setupEventListeners();

    try {
      // 1. Load dates
      const datesRes = await fetch('/api/dates');
      state.dates = await datesRes.json();
      populateDateControls();

      // 2. Load health & metrics
      const [healthRes, metricsRes] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/metrics'),
      ]);
      const health = await healthRes.json();
      state.metrics = await metricsRes.json();
      if (health.model_name) {
        dom.modelStatusText.textContent = `${health.model_name} (${(health.parameter_count / 1000).toFixed(0)}k params)`;
      }

      // 3. Trigger initial inference and surface data fetch for default date
      await runInference(state.currentDate);
      await loadSurfaceData(state.currentDate);
      await loadProfileData(state.currentDate, state.probeLat, state.probeLon);

    } catch (err) {
      console.error('Initialization error:', err);
    }
  }

  function buildDepthButtons() {
    dom.depthBtnGroup.innerHTML = '';
    DEPTHS.forEach(d => {
      const btn = document.createElement('button');
      btn.className = `depth-btn ${d === state.currentDepth ? 'active' : ''}`;
      btn.textContent = `${d}m`;
      btn.dataset.depth = d;
      btn.addEventListener('click', () => {
        state.currentDepth = d;
        document.querySelectorAll('.depth-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        dom.depthDisplay.textContent = `${d} m`;
        dom.regimePill.textContent = getRegimeName(d);
        renderSubsurfaceMap();
      });
      dom.depthBtnGroup.appendChild(btn);
    });
  }

  function populateDateControls() {
    dom.dateSelect.innerHTML = '';
    state.dates.forEach((item, idx) => {
      const opt = document.createElement('option');
      opt.value = item.date;
      opt.textContent = `${item.date} [${item.split.toUpperCase()}]${item.has_argo ? ' ★ ARGO' : ''}`;
      if (item.date === state.currentDate) opt.selected = true;
      dom.dateSelect.appendChild(opt);
    });

    dom.dateSlider.max = state.dates.length - 1;
    const defaultIdx = state.dates.findIndex(d => d.date === state.currentDate);
    if (defaultIdx >= 0) dom.dateSlider.value = defaultIdx;

    updateDatePills();
  }

  function updateDatePills() {
    const curInfo = state.dates.find(d => d.date === state.currentDate);
    if (!curInfo) return;

    dom.splitPill.textContent = `${curInfo.split.toUpperCase()} PERIOD`;
    if (curInfo.has_argo) {
      dom.argoPill.classList.remove('hidden');
      dom.argoPill.textContent = `● ARGO IN SITU FLOATS AVAILABLE (${curInfo.argo_count})`;
    } else {
      dom.argoPill.classList.add('hidden');
    }
  }

  function setupEventListeners() {
    // Date select
    dom.dateSelect.addEventListener('change', async (e) => {
      state.currentDate = e.target.value;
      const idx = state.dates.findIndex(d => d.date === state.currentDate);
      if (idx >= 0) dom.dateSlider.value = idx;
      updateDatePills();
      await runInference(state.currentDate);
      await loadSurfaceData(state.currentDate);
      await loadProfileData(state.currentDate, state.probeLat, state.probeLon);
    });

    // Date slider
    dom.dateSlider.addEventListener('input', async (e) => {
      const idx = parseInt(e.target.value, 10);
      if (state.dates[idx]) {
        state.currentDate = state.dates[idx].date;
        dom.dateSelect.value = state.currentDate;
        updateDatePills();
        await runInference(state.currentDate);
        await loadSurfaceData(state.currentDate);
        await loadProfileData(state.currentDate, state.probeLat, state.probeLon);
      }
    });

    // Run inference button
    dom.runInferenceBtn.addEventListener('click', async () => {
      await runInference(state.currentDate, true);
    });

    // Surface tabs
    dom.surfaceTabs.addEventListener('click', (e) => {
      const btn = e.target.closest('.tab-btn');
      if (!btn) return;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentChannel = btn.dataset.channel;
      renderSurfaceMap();
    });

    // Subsurface Canvas click & hover
    dom.subsurfaceCanvas.addEventListener('click', onMapClick);
    dom.subsurfaceCanvas.addEventListener('mousemove', onMapMouseMove);
    dom.subsurfaceCanvas.addEventListener('mouseleave', () => {
      dom.mapTooltip.classList.add('hidden');
    });
  }

  // =========================================================================
  // API CALLS & INFERENCE
  // =========================================================================
  async function runInference(dateStr, force = false) {
    if (!force && state.predictionsCache[dateStr]) {
      state.prediction = state.predictionsCache[dateStr];
      dom.latencyVal.textContent = `Latency: ${state.prediction.inference_time_ms || 46} ms`;
      dom.cacheVal.textContent = 'Cached (In-Memory)';
      renderSubsurfaceMap();
      return;
    }

    try {
      dom.runInferenceBtn.disabled = true;
      dom.runInferenceBtn.querySelector('.btn-text').textContent = 'Inferring...';

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date: dateStr }),
      });
      const data = await res.json();
      state.prediction = data;
      state.predictionsCache[dateStr] = data;

      dom.latencyVal.textContent = `Latency: ${data.inference_time_ms} ms`;
      dom.cacheVal.textContent = data.cached ? 'Cached' : 'Direct Forward Pass';

      renderSubsurfaceMap();
    } catch (err) {
      console.error('Error running inference:', err);
    } finally {
      dom.runInferenceBtn.disabled = false;
      dom.runInferenceBtn.querySelector('.btn-text').textContent = 'Run OceanEmbed';
    }
  }

  async function loadSurfaceData(dateStr) {
    if (state.surfaceCache[dateStr]) {
      state.surface = state.surfaceCache[dateStr];
      renderSurfaceMap();
      return;
    }
    try {
      const res = await fetch(`/api/surface?date=${dateStr}`);
      const data = await res.json();
      state.surface = data;
      state.surfaceCache[dateStr] = data;
      renderSurfaceMap();
    } catch (err) {
      console.error('Error fetching surface data:', err);
    }
  }

  async function loadProfileData(dateStr, lat, lon) {
    try {
      const res = await fetch(`/api/profile?date=${dateStr}&lat=${lat}&lon=${lon}`);
      const data = await res.json();
      state.profile = data;
      renderProfileChart();
    } catch (err) {
      console.error('Error fetching profile data:', err);
    }
  }

  // =========================================================================
  // RENDERING: PRIMARY SUBSURFACE MAP (CANVAS)
  // =========================================================================
  function renderSubsurfaceMap() {
    if (!state.prediction) return;

    const canvas = dom.subsurfaceCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const depthIdx = DEPTHS.indexOf(state.currentDepth);
    const grid = state.prediction.prediction[depthIdx >= 0 ? depthIdx : 7]; // [24, 32]
    const oceanMask = state.prediction.ocean_mask;

    // Determine temperature min/max for active depth level
    let minT = 1e9, maxT = -1e9;
    for (let r = 0; r < 24; r++) {
      for (let c = 0; c < 32; c++) {
        const val = grid[r][c];
        if (val !== null && !isNaN(val)) {
          if (val < minT) minT = val;
          if (val > maxT) maxT = val;
        }
      }
    }
    if (minT === 1e9) { minT = 10; maxT = 30; }

    dom.cbarMin.textContent = `${minT.toFixed(1)} °C`;
    dom.cbarMax.textContent = `${maxT.toFixed(1)} °C`;
    dom.mapSubtitle.textContent = `Depth: ${state.currentDepth} m • Date: ${state.currentDate} • 24 × 32 Grid (0.25°)`;

    const cellW = width / 32;
    const cellH = height / 24;

    // Render cells (row 0 = north, row 23 = south)
    for (let r = 0; r < 24; r++) {
      const canvasRow = 23 - r; // Invert row so North (high lat) is at top
      for (let c = 0; c < 32; c++) {
        const val = grid[r][c];
        const isOcean = oceanMask[r][c] === 1;

        if (!isOcean || val === null) {
          // Masked land / bathymetry
          ctx.fillStyle = '#111c2e';
          ctx.fillRect(c * cellW, canvasRow * cellH, cellW + 0.5, cellH + 0.5);

          // Diagonal hatch line
          ctx.strokeStyle = '#1e3250';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(c * cellW, canvasRow * cellH);
          ctx.lineTo((c + 1) * cellW, (canvasRow + 1) * cellH);
          ctx.stroke();
        } else {
          ctx.fillStyle = getColorFromScale(val, minT, maxT);
          ctx.fillRect(c * cellW, canvasRow * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
    }

    // Grid lines (subtle)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 0.5;
    for (let c = 0; c <= 32; c += 4) {
      ctx.beginPath();
      ctx.moveTo(c * cellW, 0);
      ctx.lineTo(c * cellW, height);
      ctx.stroke();
    }
    for (let r = 0; r <= 24; r += 4) {
      ctx.beginPath();
      ctx.moveTo(0, r * cellH);
      ctx.lineTo(width, r * cellH);
      ctx.stroke();
    }

    // Latitude & Longitude border labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('17.88°N', 6, 4);
    ctx.fillText('12.13°N', 6, height - 14);
    ctx.textAlign = 'right';
    ctx.fillText('85.13°E', width - 60, height - 14);
    ctx.fillText('92.88°E', width - 6, height - 14);

    // ARGO Float Markers (if available for this date)
    const argoFloats = state.prediction.argo_floats || [];
    argoFloats.forEach((fl) => {
      const c = (fl.lon - 85.125) / 0.25;
      const r = (fl.lat - 12.125) / 0.25;
      const x = (c + 0.5) * cellW;
      const y = (23 - r + 0.5) * cellH;

      ctx.save();
      ctx.translate(x, y);

      // Glow halo
      ctx.fillStyle = 'rgba(245, 158, 11, 0.3)';
      ctx.beginPath();
      ctx.arc(0, 0, 9, 0, Math.PI * 2);
      ctx.fill();

      // Diamond marker
      ctx.fillStyle = '#f59e0b';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, -6);
      ctx.lineTo(6, 0);
      ctx.lineTo(0, 6);
      ctx.lineTo(-6, 0);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.restore();
    });

    // Update Probed Coordinate Crosshair
    updateCrosshair();
  }

  function updateCrosshair() {
    const canvas = dom.subsurfaceCanvas;
    const rect = canvas.getBoundingClientRect();
    const cellW = rect.width / 32;
    const cellH = rect.height / 24;

    const c = (state.probeLon - 85.125) / 0.25;
    const r = (state.probeLat - 12.125) / 0.25;
    const x = (c + 0.5) * cellW;
    const y = (23 - r + 0.5) * cellH;

    dom.mapCrosshair.classList.remove('hidden');
    dom.mapCrosshair.style.left = `${x}px`;
    dom.mapCrosshair.style.top = `${y}px`;

    // Update probe badges
    dom.cellProbeBadge.textContent = `Probe: ${state.probeLat.toFixed(2)}°N, ${state.probeLon.toFixed(2)}°E`;
    if (state.profile && state.profile.oceanembed_profile) {
      const dIdx = DEPTHS.indexOf(state.currentDepth);
      const tVal = state.profile.oceanembed_profile[dIdx >= 0 ? dIdx : 7];
      dom.cellTempBadge.textContent = tVal !== null ? `${tVal.toFixed(2)} °C` : 'Land / Masked';
    }
  }

  function onMapClick(e) {
    const rect = dom.subsurfaceCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellW = rect.width / 32;
    const cellH = rect.height / 24;

    const c = Math.floor(x / cellW);
    const canvasRow = Math.floor(y / cellH);
    const r = 23 - canvasRow;

    if (r >= 0 && r < 24 && c >= 0 && c < 32) {
      state.probeLat = LATS[r];
      state.probeLon = LONS[c];
      updateCrosshair();
      loadProfileData(state.currentDate, state.probeLat, state.probeLon);
    }
  }

  function onMapMouseMove(e) {
    const rect = dom.subsurfaceCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cellW = rect.width / 32;
    const cellH = rect.height / 24;

    const c = Math.floor(x / cellW);
    const canvasRow = Math.floor(y / cellH);
    const r = 23 - canvasRow;

    if (r >= 0 && r < 24 && c >= 0 && c < 32 && state.prediction) {
      const lat = LATS[r];
      const lon = LONS[c];
      const dIdx = DEPTHS.indexOf(state.currentDepth);
      const val = state.prediction.prediction[dIdx >= 0 ? dIdx : 7][r][c];
      const isOcean = state.prediction.ocean_mask[r][c] === 1;

      dom.mapTooltip.classList.remove('hidden');
      dom.mapTooltip.style.left = `${Math.min(x + 12, rect.width - 150)}px`;
      dom.mapTooltip.style.top = `${Math.max(y - 35, 10)}px`;

      if (!isOcean || val === null) {
        dom.mapTooltip.innerHTML = `<strong>${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E</strong><br><span style="color:#94a3b8">Masked Land / Coast</span>`;
      } else {
        dom.mapTooltip.innerHTML = `<strong>${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E</strong><br>Temp: <span style="color:#00e5ff;font-weight:700">${val.toFixed(2)} °C</span>`;
      }
    }
  }

  // =========================================================================
  // RENDERING: 7 SURFACE INPUTS MAP
  // =========================================================================
  function renderSurfaceMap() {
    if (!state.surface || !state.surface.channels) return;

    const channelMeta = state.surface.channels[state.currentChannel];
    if (!channelMeta) return;

    const canvas = dom.surfaceCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const grid = channelMeta.data; // [24, 32]
    const oceanMask = state.surface.ocean_mask;

    // Calculate min/max for this channel
    let minV = 1e9, maxV = -1e9;
    for (let r = 0; r < 24; r++) {
      for (let c = 0; c < 32; c++) {
        const val = grid[r][c];
        if (val !== null && !isNaN(val)) {
          if (val < minV) minV = val;
          if (val > maxV) maxV = val;
        }
      }
    }
    if (minV === 1e9) { minV = 0; maxV = 1; }

    dom.surfaceCoverageVal.textContent = `${channelMeta.coverage_pct}% valid ocean`;
    dom.surfaceCbarMin.textContent = minV.toFixed(2);
    dom.surfaceCbarMax.textContent = maxV.toFixed(2);
    dom.surfaceCbarUnits.textContent = channelMeta.units;

    const cellW = width / 32;
    const cellH = height / 24;

    for (let r = 0; r < 24; r++) {
      const canvasRow = 23 - r;
      for (let c = 0; c < 32; c++) {
        const val = grid[r][c];
        const isOcean = oceanMask[r][c] === 1;

        if (!isOcean || val === null) {
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(c * cellW, canvasRow * cellH, cellW + 0.5, cellH + 0.5);
        } else {
          ctx.fillStyle = getColorFromScale(val, minV, maxV);
          ctx.fillRect(c * cellW, canvasRow * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
    }
  }

  // =========================================================================
  // RENDERING: VERTICAL TEMPERATURE PROFILE (CANVAS CHART)
  // =========================================================================
  function renderProfileChart() {
    if (!state.profile) return;

    const canvas = dom.profileCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    dom.profileSubtitle.textContent = `Location: ${state.probeLat.toFixed(2)}°N, ${state.probeLon.toFixed(2)}°E • Date: ${state.currentDate}`;

    // ARGO badge status
    const argo = state.profile.argo_profile;
    if (argo && argo.depths && argo.depths.length > 0) {
      dom.argoCollocationStatus.classList.remove('not-collocated');
      dom.argoCollocationStatus.textContent = `Collocated ARGO Float WMO #${argo.platform_number} (${argo.distance_km} km away)`;
      dom.argoLegendRow.classList.remove('hidden');
    } else {
      dom.argoCollocationStatus.classList.add('not-collocated');
      dom.argoCollocationStatus.textContent = 'No ARGO Float on this date at coordinate';
      dom.argoLegendRow.classList.add('hidden');
    }

    if (!state.profile.valid_ocean) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px "Outfit", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Selected coordinate is Land or Masked Bathymetry', width / 2, height / 2);
      return;
    }

    // Chart margins
    const margin = { left: 55, right: 30, top: 30, bottom: 45 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    // Gather temperature bounds
    const pModel = state.profile.oceanembed_profile || [];
    const pClim = state.profile.climatology_profile || [];
    const pGlorys = state.profile.glorys_profile || [];
    const pArgo = argo ? argo.temperatures : [];

    const allTemps = [...pModel, ...pClim, ...pGlorys, ...pArgo].filter(t => t !== null && !isNaN(t));
    let tMin = allTemps.length ? Math.min(...allTemps) : 5;
    let tMax = allTemps.length ? Math.max(...allTemps) : 32;
    tMin = Math.max(4, Math.floor(tMin - 1));
    tMax = Math.min(35, Math.ceil(tMax + 1));

    // Non-linear depth mapping (square-root scaling ensures 0–200m thermocline is detailed)
    const maxDepth = 1000;
    function depthToY(d) {
      const frac = Math.sqrt(Math.max(0, Math.min(d, maxDepth))) / Math.sqrt(maxDepth);
      return margin.top + frac * chartH;
    }

    function tempToX(t) {
      const frac = (t - tMin) / (tMax - tMin);
      return margin.left + frac * chartW;
    }

    // 1. Draw Regime Shading
    // Mixed layer (0–20m)
    const y0 = depthToY(0);
    const y20 = depthToY(20);
    ctx.fillStyle = 'rgba(0, 229, 255, 0.05)';
    ctx.fillRect(margin.left, y0, chartW, y20 - y0);

    // Thermocline (50–200m)
    const y50 = depthToY(50);
    const y200 = depthToY(200);
    ctx.fillStyle = 'rgba(245, 158, 11, 0.06)';
    ctx.fillRect(margin.left, y50, chartW, y200 - y50);

    // Regime labels on chart
    ctx.fillStyle = 'rgba(0, 229, 255, 0.5)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Mixed Layer (0–20m)', width - margin.right - 8, y20 - 4);

    ctx.fillStyle = 'rgba(245, 158, 11, 0.5)';
    ctx.fillText('Thermocline (50–200m)', width - margin.right - 8, y200 - 8);

    // 2. Draw Depth Grid Lines & Ticks (Y-axis)
    const depthTicks = [0, 20, 50, 100, 200, 300, 500, 700, 1000];
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.fillStyle = '#64748b';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    depthTicks.forEach(d => {
      const y = depthToY(d);
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(width - margin.right, y);
      ctx.stroke();

      ctx.fillText(`${d}m`, margin.left - 8, y);
    });

    // 3. Draw Temperature Ticks (X-axis)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const tempStep = (tMax - tMin > 15) ? 5 : 2;
    for (let t = Math.ceil(tMin / tempStep) * tempStep; t <= tMax; t += tempStep) {
      const x = tempToX(t);
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, height - margin.bottom);
      ctx.stroke();

      ctx.fillText(`${t}°C`, x, height - margin.bottom + 8);
    }

    // Axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px "Outfit", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Temperature (°C)', margin.left + chartW / 2, height - 12);

    ctx.save();
    ctx.translate(14, margin.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Depth (m, inverted)', 0, 0);
    ctx.restore();

    // 4. Series 1: Training Spatial Climatology (dashed slate)
    drawSeries(ctx, DEPTHS, pClim, '#94a3b8', { isDashed: true, lineWidth: 1.5, dotRadius: 2.5 });

    // 5. Series 2: GLORYS Reanalysis Reference (emerald green)
    drawSeries(ctx, DEPTHS, pGlorys, '#10b981', { isDashed: false, lineWidth: 2.0, dotRadius: 3.5 });

    // 6. Series 3: Independent ARGO Float (amber diamonds)
    if (argo && argo.depths && argo.depths.length > 0) {
      drawArgoSeries(ctx, argo.depths, argo.temperatures, '#f59e0b');
    }

    // 7. Series 4: OceanEmbed v1-Local Prediction (cyan with glow)
    drawSeries(ctx, DEPTHS, pModel, '#00e5ff', { isDashed: false, lineWidth: 2.8, dotRadius: 4.5, glow: true });
  }

  function drawSeries(ctx, depths, temps, color, options = {}) {
    const margin = { left: 55, right: 30, top: 30, bottom: 45 };
    const chartW = ctx.canvas.width - margin.left - margin.right;
    const chartH = ctx.canvas.height - margin.top - margin.bottom;

    const allTemps = temps.filter(t => t !== null && !isNaN(t));
    if (allTemps.length === 0) return;

    let tMin = 4, tMax = 32;
    if (state.profile) {
      const pModel = state.profile.oceanembed_profile || [];
      const pClim = state.profile.climatology_profile || [];
      const pGlorys = state.profile.glorys_profile || [];
      const pArgo = state.profile.argo_profile ? state.profile.argo_profile.temperatures : [];
      const combined = [...pModel, ...pClim, ...pGlorys, ...pArgo].filter(t => t !== null && !isNaN(t));
      if (combined.length) {
        tMin = Math.max(4, Math.floor(Math.min(...combined) - 1));
        tMax = Math.min(35, Math.ceil(Math.max(...combined) + 1));
      }
    }

    function depthToY(d) {
      const frac = Math.sqrt(Math.max(0, Math.min(d, 1000))) / Math.sqrt(1000);
      return margin.top + frac * chartH;
    }
    function tempToX(t) {
      return margin.left + ((t - tMin) / (tMax - tMin)) * chartW;
    }

    ctx.save();
    if (options.glow) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = options.lineWidth || 2;
    if (options.isDashed) ctx.setLineDash([5, 4]);

    ctx.beginPath();
    let started = false;
    depths.forEach((d, i) => {
      const t = temps[i];
      if (t !== null && !isNaN(t)) {
        const x = tempToX(t);
        const y = depthToY(d);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
    });
    ctx.stroke();
    ctx.restore();

    // Markers
    ctx.fillStyle = color;
    depths.forEach((d, i) => {
      const t = temps[i];
      if (t !== null && !isNaN(t)) {
        const x = tempToX(t);
        const y = depthToY(d);
        ctx.beginPath();
        ctx.arc(x, y, options.dotRadius || 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  function drawArgoSeries(ctx, depths, temps, color) {
    const margin = { left: 55, right: 30, top: 30, bottom: 45 };
    const chartW = ctx.canvas.width - margin.left - margin.right;
    const chartH = ctx.canvas.height - margin.top - margin.bottom;

    let tMin = 4, tMax = 32;
    if (state.profile) {
      const pModel = state.profile.oceanembed_profile || [];
      const pClim = state.profile.climatology_profile || [];
      const pGlorys = state.profile.glorys_profile || [];
      const pArgo = temps;
      const combined = [...pModel, ...pClim, ...pGlorys, ...pArgo].filter(t => t !== null && !isNaN(t));
      if (combined.length) {
        tMin = Math.max(4, Math.floor(Math.min(...combined) - 1));
        tMax = Math.min(35, Math.ceil(Math.max(...combined) + 1));
      }
    }

    function depthToY(d) {
      const frac = Math.sqrt(Math.max(0, Math.min(d, 1000))) / Math.sqrt(1000);
      return margin.top + frac * chartH;
    }
    function tempToX(t) {
      return margin.left + ((t - tMin) / (tMax - tMin)) * chartW;
    }

    // Line through in situ points
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    depths.forEach((d, i) => {
      const t = temps[i];
      if (t !== null && !isNaN(t)) {
        const x = tempToX(t);
        const y = depthToY(d);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
    });
    ctx.stroke();

    // Diamond markers
    ctx.fillStyle = color;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    depths.forEach((d, i) => {
      const t = temps[i];
      if (t !== null && !isNaN(t)) {
        const x = tempToX(t);
        const y = depthToY(d);
        ctx.beginPath();
        ctx.moveTo(x, y - 4);
        ctx.lineTo(x + 4, y);
        ctx.lineTo(x, y + 4);
        ctx.lineTo(x - 4, y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    });
  }

  // Start app on DOMContentLoaded
  document.addEventListener('DOMContentLoaded', init);
})();
