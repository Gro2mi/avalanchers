import init, { WasmSimulation, decode_blosc_chunk, decode_zstd_chunk } from "./pkg/avalanchers.js";

window.dem = new Dem();
window.sim = null;
window.wasm = null;
// sources.js is a classic script and cannot import the module directly.
window.wasmDecoders = null;

// ---------------------------------------------------------------------------
// Element references
// ---------------------------------------------------------------------------

const statusEl = document.getElementById('status');
const loadingScreen = document.getElementById('loadingScreen');
const loadingStatus = document.getElementById('loadingStatus');
const loadingDetail = document.getElementById('loadingDetail');

const demDropdown = document.getElementById('demDropdown');
const runShortcutButton = document.getElementById('runShortcut');
const demReleaseGroup = document.getElementById('demReleaseGroup');

const demDirButton = document.getElementById('demDirButton');
const demFileInput = document.getElementById('demFileInput');
const demDirInput = document.getElementById('demDirInput');
const demDropZone = document.getElementById('demDropZone');
const demStatus = document.getElementById('demStatus');
const zarrSiteRow = document.getElementById('zarrSiteRow');
const zarrSiteDropdown = document.getElementById('zarrSiteDropdown');
const zoomLevelRow = document.getElementById('zoomLevelRow');
const zoomLevelSlider = document.getElementById('zoomLevelSlider');

const releaseCard = document.getElementById('releaseCard');
const calculateReleaseButton = document.getElementById('calculateReleaseButton');
const releaseDirButton = document.getElementById('releaseDirButton');
const releaseFileInput = document.getElementById('releaseFileInput');
const releaseDirInput = document.getElementById('releaseDirInput');
const releaseDropZone = document.getElementById('releaseDropZone');
const releaseStatus = document.getElementById('releaseStatus');
const zarrScenarioRow = document.getElementById('zarrScenarioRow');
const zarrScenarioDropdown = document.getElementById('zarrScenarioDropdown');

const simModelDropdown = document.getElementById('simModelDropdown');
const frictionModelDropdown = document.getElementById('frictionModelDropdown');
const densitySlider = document.getElementById('densitySlider');
const slabThicknessSlider = document.getElementById('slabThicknessSlider');
const minSlopeAngleSlider = document.getElementById('minSlopeAngleSlider');
const maxSlopeAngleSlider = document.getElementById('maxSlopeAngleSlider');
const releaseMinElevationSlider = document.getElementById('releaseMinElevationSlider');
const releaseMaxElevationSlider = document.getElementById('releaseMaxElevationSlider');
const roughnessThresholdSlider = document.getElementById('roughnessThresholdSlider');
const stepSlider = document.getElementById('stepSlider');
const cflSlider = document.getElementById('cflSlider');
const frictionCoefficientSlider = document.getElementById('frictionCoefficientSlider');
const dragCoefficientSlider = document.getElementById('dragCoefficientSlider');
const releasedParticlesPerCellSlider = document.getElementById('releasedParticlesPerCellSlider');
const internalFrictionAngleSlider = document.getElementById('internalFrictionAngleSlider');

const runButton = document.getElementById('runSimulation');
const runButtonLabel = document.getElementById('runButtonLabel');
const runSpinner = document.getElementById('runSpinner');
const prepareButton = document.getElementById('prepareSimulation');
const runStatus = document.getElementById('runStatus');
const runStatusDot = document.getElementById('runStatusDot');
const saveResultsButton = document.getElementById('saveResults');
const saveStatus = document.getElementById('saveStatus');
const loadedDemLabel = document.getElementById('loadedDemLabel');
const loadedReleaseLabel = document.getElementById('loadedReleaseLabel');

const plotVariable = document.getElementById('plotVariable');
const plotVariableAnchor = document.getElementById('plotVariableAnchor');
const resultPlots = document.getElementById('resultPlots');
const demPlotElement = document.getElementById('demPlot');
const demPlotContainer = document.getElementById('demPlotContainer');
const demPlotStickyHost = document.getElementById('demPlotStickyHost');
const demPlotFlowHost = document.getElementById('demPlotFlowHost');
const desktopLayout = window.matchMedia('(min-width: 1200px)');

// ---------------------------------------------------------------------------
// Workflow state
// ---------------------------------------------------------------------------

/**
 * demSource:     { kind: 'example', name }
 *              | { kind: 'gpx', name, gpx }
 *              | { kind: 'raster', name, bytes, ext }
 *              | { kind: 'zarr', name, store, site, dem }
 * releaseSource: null (engine derives them)
 *              | { kind: 'example' }
 *              | { kind: 'calculated' }
 *              | { kind: 'raster', name, bytes, ext }
 *              | { kind: 'zarr', name, scenario, data }
 */
const state = {
    engineReady: false,
    demSource: null,
    releaseSource: null,
    hasResults: false,
    busy: false,
};

const isMobileDevice = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

function setRunStatus(message, kind = 'idle') {
    runStatus.textContent = message;
    runStatusDot.className = 'status-dot';
    if (kind === 'ready') runStatusDot.classList.add('is-ready');
    if (kind === 'running') runStatusDot.classList.add('is-running');
    if (kind === 'error') runStatusDot.classList.add('is-error');
}

function setLoadingScreenStatus(engineMessage, detailMessage) {
    if (loadingStatus && typeof engineMessage === 'string') {
        loadingStatus.textContent = engineMessage;
    }
    if (loadingDetail && typeof detailMessage === 'string') {
        loadingDetail.textContent = detailMessage;
    }
}

function hideLoadingScreen() {
    if (!loadingScreen) return;
    loadingScreen.classList.add('is-hidden');
    loadingScreen.setAttribute('aria-busy', 'false');
}

function setCardStatus(element, message, kind = 'info') {
    element.textContent = message;
    element.className = 'source-summary ' +
        (kind === 'error' ? 'text-danger' : kind === 'ok' ? 'text-success' : 'text-secondary');
}

function resizePlotIfRendered(plotId) {
    const element = document.getElementById(plotId);
    if (element?.classList.contains('js-plotly-plot')) {
        Plotly.Plots.resize(element);
    }
}

function resizeVisibleResultPlots() {
    ['demPlot', 'histogramPlot', 'outputPlot', 'timerPlot', 'debugPlot']
        .forEach(resizePlotIfRendered);
}

function updateResultPlotsVisibility() {
    if (!resultPlots) return;
    const wasHidden = resultPlots.classList.contains('d-none');
    resultPlots.classList.toggle('d-none', !state.hasResults);

    if (wasHidden && state.hasResults) {
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                resizeVisibleResultPlots();
            });
        });
    }
}

function describeDemSource(source) {
    switch (source?.kind) {
        case 'example': return `Example case "${source.name}" (DEM and release areas).`;
        case 'gpx': return `GPX track "${source.name}" — elevation tiles at zoom ${zoomLevelSlider.value}.`;
        case 'raster': return `Raster "${source.name}".`;
        case 'zarr': return source.site
            ? `Zarr store "${source.name}", site "${source.site}".`
            : `Zarr store "${source.name}" — select a site.`;
        default: return 'No DEM selected.';
    }
}

function describeReleaseSource(source) {
    switch (source?.kind) {
        case 'example': return 'Release areas from the selected example case.';
        case 'calculated': return 'Release areas derived from the terrain.';
        case 'raster': return `Release areas from "${source.name}".`;
        case 'zarr': return `Release areas from scenario "${source.scenario}".`;
        default: return 'No release areas selected — the engine will derive them.';
    }
}

/** True once a DEM is loaded and usable by the engine. */
function hasUsableDem() {
    const source = state.demSource;
    if (!source) return false;
    if (source.kind === 'zarr') return !!source.dem;
    return true;
}

/** Enables or disables interactive controls based on engine and workflow state. */
function setBusy(busy) {
    state.busy = busy;
    const ready = state.engineReady;
    const hasDem = hasUsableDem();

    runShortcutButton.disabled = busy || !ready;
    demDropdown.disabled = busy || !ready;
    demDirButton.disabled = busy || !ready;
    zarrSiteDropdown.disabled = busy || !ready;
    zoomLevelSlider.disabled = busy || !ready;

    calculateReleaseButton.disabled = busy || !ready || !hasDem;
    releaseDirButton.disabled = busy || !ready || !hasDem;
    zarrScenarioDropdown.disabled = busy || !ready || !hasDem;
    releaseCard.classList.toggle('is-locked', !hasDem);

    document.querySelectorAll('#simSettingsDiv input, #simSettingsDiv select')
        .forEach(el => { el.disabled = busy; });

    runButton.disabled = busy || !ready || !hasDem;
    prepareButton.disabled = busy || !ready || !hasDem;
    plotVariable.disabled = busy || !ready;
    saveResultsButton.disabled = busy || !ready || !state.hasResults;

    runSpinner.classList.toggle('d-none', !busy);
    runButtonLabel.textContent = busy ? 'Running…' : 'Run simulation';

    if (!busy) updateFrictionControlsFromModel();
}

function refreshWorkflowState() {
    const demDescription = describeDemSource(state.demSource);
    const releaseDescription = describeReleaseSource(state.releaseSource);

    setCardStatus(demStatus, demDescription, state.demSource ? 'ok' : 'info');
    setCardStatus(releaseStatus, releaseDescription,
        state.releaseSource ? 'ok' : 'info');
    loadedDemLabel.textContent = demDescription;
    loadedReleaseLabel.textContent = releaseDescription;
    updateResultPlotsVisibility();
    setBusy(state.busy);
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

function bindSlider(slider, valueId, format) {
    const valueField = document.getElementById(valueId);
    const update = () => {
        const formatted = format ? format(slider.value) : slider.value;
        if (valueField instanceof HTMLInputElement) {
            valueField.value = formatted;
            valueField.min = slider.min;
            valueField.max = slider.max;
            valueField.step = slider.step || 'any';
            return;
        }
        valueField.textContent = formatted;
    };
    slider.addEventListener('input', update);
    valueField.addEventListener('change', () => {
        if (!(valueField instanceof HTMLInputElement)) return;
        const value = Number(valueField.value);
        if (!Number.isFinite(value)) return;
        const min = Number(slider.min);
        const max = Number(slider.max);
        const clamped = Math.min(Math.max(value, min), max);
        slider.value = String(clamped);
        update();
    });
    update();
}

bindSlider(densitySlider, 'densitySliderValue');
bindSlider(slabThicknessSlider, 'slabThicknessSliderValue', v => parseFloat(v).toFixed(2));
bindSlider(minSlopeAngleSlider, 'minSlopeAngleSliderValue', v => parseFloat(v).toFixed(1));
bindSlider(maxSlopeAngleSlider, 'maxSlopeAngleSliderValue', v => parseFloat(v).toFixed(1));
bindSlider(releaseMinElevationSlider, 'releaseMinElevationSliderValue');
bindSlider(releaseMaxElevationSlider, 'releaseMaxElevationSliderValue');
bindSlider(roughnessThresholdSlider, 'roughnessThresholdSliderValue', v => parseFloat(v).toFixed(3));
bindSlider(stepSlider, 'stepSliderValue');
bindSlider(cflSlider, 'cflSliderValue', v => parseFloat(v).toFixed(2));
bindSlider(frictionCoefficientSlider, 'frictionCoefficientValue', v => parseFloat(v).toFixed(4));
bindSlider(dragCoefficientSlider, 'dragCoefficientValue');
bindSlider(releasedParticlesPerCellSlider, 'releasedParticlesPerCellValue');
bindSlider(internalFrictionAngleSlider, 'internalFrictionAngleValue', v => parseFloat(v).toFixed(1));
bindSlider(zoomLevelSlider, 'zoomLevelValue');

function updateFrictionControlsFromModel() {
    const selectedModel = frictionModelDropdown.selectedOptions[0].text;
    dragCoefficientSlider.disabled = selectedModel === 'Coulomb' || selectedModel === 'samosAT';
}

function changeFrictionModel() {
    const selectedModel = frictionModelDropdown.selectedOptions[0].text;
    frictionCoefficientSlider.value = selectedModel === 'Coulomb' ? 0.4663 : 0.2;
    const frictionValueInput = document.getElementById('frictionCoefficientValue');
    if (frictionValueInput instanceof HTMLInputElement) {
        frictionValueInput.value = frictionCoefficientSlider.value;
    } else {
        frictionValueInput.textContent = frictionCoefficientSlider.value;
    }
    updateFrictionControlsFromModel();
}

function getSettings() {
    const base = window.location.pathname.replace(/\/[^/]+\.[^/]+$/, "/") + "data/avaframe/";
    const simSettings = {
        dem_path: base + demDropdown.value + ".png",
        release_areas_path: base + demDropdown.value + "releaseTexture.png",
        max_steps: parseInt(stepSlider.value),
        sim_model: simModelDropdown.value,
        friction_model: frictionModelDropdown.selectedIndex,
        density: parseFloat(densitySlider.value),
        slab_thickness_factor: parseFloat(slabThicknessSlider.value),
        min_slope_angle: parseFloat(minSlopeAngleSlider.value),
        max_slope_angle: parseFloat(maxSlopeAngleSlider.value),
        release_min_elevation: parseFloat(releaseMinElevationSlider.value),
        release_max_elevation: parseFloat(releaseMaxElevationSlider.value),
        roughness_threshold: parseFloat(roughnessThresholdSlider.value),
        friction_coefficient: parseFloat(frictionCoefficientSlider.value),
        drag_coefficient: parseInt(dragCoefficientSlider.value),
        cfl: parseFloat(cflSlider.value),
        released_particles_per_cell: parseInt(releasedParticlesPerCellSlider.value),
        enable_curvature: document.getElementById('enable_curvature').checked,
        enable_particle_interaction: document.getElementById('enable_particle_interaction').checked,
        enable_earth_pressure_coefficient: document.getElementById('enable_earth_pressure_coefficient').checked,
        internal_friction_angle: parseFloat(internalFrictionAngleSlider.value),
    };
    return simSettings;
}

// ---------------------------------------------------------------------------
// Applying the selected sources to the simulation
// ---------------------------------------------------------------------------

/**
 * Rebuilds the simulation from the current settings and re-applies the selected
 * sources. `create` resets the engine, so the DEM and release areas have to be
 * re-sent whenever the settings change.
 */
async function applySources() {
    const settings = getSettings();
    // Rebuilding the engine discards any results from a previous run.
    state.hasResults = false;

    if (state.demSource?.kind === 'example') {
        await sim.create(settings);
        return;
    }

    delete settings.dem_path;
    delete settings.release_areas_path;
    await sim.create(settings);
    await applyDem();
    await applyReleaseAreas();
}

async function applyDem() {
    const source = state.demSource;
    if (!source) throw new Error('No DEM selected.');

    if (source.kind === 'gpx') {
        await sim.set_dem(
            dem.data1d, dem.width, dem.height, dem.cellSize,
            dem.bounds.xmin, dem.bounds.xmax, dem.bounds.ymin, dem.bounds.ymax,
            dem.mapFactor);
        return;
    }

    if (source.kind === 'raster') {
        await sim.load_dem_bytes(source.bytes, source.ext, source.name);
        return;
    }

    if (source.kind === 'zarr') {
        const { data, width, height, cellSize, bounds } = source.dem;
        await sim.set_dem(
            data, width, height, cellSize,
            bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax, 1.0);
    }
}

async function applyReleaseAreas() {
    const source = state.releaseSource;
    // 'calculated', 'example' and null all let the engine derive release areas.
    if (!source || source.kind === 'calculated' || source.kind === 'example') return;

    if (source.kind === 'raster') {
        await sim.load_release_areas_bytes(source.bytes, source.ext);
        return;
    }

    if (source.kind === 'zarr') {
        await sim.set_release_areas(source.data);
    }
}

// ---------------------------------------------------------------------------
// Step 1: DEM selection
// ---------------------------------------------------------------------------

/** Clears everything that depends on the current DEM. */
function resetDependentState() {
    state.releaseSource = null;
    zarrScenarioRow.classList.add('d-none');
    zarrScenarioDropdown.innerHTML = '';
    resetPlots();
}

function setZoomRowVisible(visible) {
    zoomLevelRow.classList.toggle('d-none', !visible);
    zoomLevelRow.classList.toggle('d-flex', visible);
}

async function loadDemFromEntries(entries) {
    if (looksLikeZarr(entries)) {
        await loadDemFromZarr(entries);
        return;
    }

    const supported = entries.filter(e => DEM_FILE_EXTENSIONS.includes(fileExtension(e.path)));
    if (!supported.length) {
        setCardStatus(demStatus,
            `Unsupported file. Use ${DEM_FILE_EXTENSIONS.map(e => '.' + e).join(', ')} or a Zarr folder.`,
            'error');
        return;
    }

    const file = supported[0].file;
    const ext = fileExtension(file.name);
    if (ext === 'gpx') {
        await loadDemFromGpx(file);
    } else {
        await loadDemFromRaster(file, ext);
    }
}

async function loadDemFromGpx(file) {
    await withBusy(async () => {
        setCardStatus(demStatus, `Loading elevation tiles for "${file.name}"…`);
        const gpx = parseGPX(await file.text());
        await dem.loadTiles(gpx, zoomLevelSlider.value);

        resetDependentState();
        state.demSource = { kind: 'gpx', name: file.name, gpx };
        zarrSiteRow.classList.add('d-none');
        setZoomRowVisible(true);

        await applySources();
        plotDem(sim);
        plotGpx(gpx, dem);
        setRunStatus('DEM ready.', 'ready');
    }, demStatus);
}

async function loadDemFromRaster(file, ext) {
    await withBusy(async () => {
        setCardStatus(demStatus, `Reading "${file.name}"…`);
        const bytes = new Uint8Array(await file.arrayBuffer());

        resetDependentState();
        state.demSource = { kind: 'raster', name: file.name, bytes, ext };
        zarrSiteRow.classList.add('d-none');
        setZoomRowVisible(false);

        await applySources();
        plotDem(sim);
        setRunStatus('DEM ready.', 'ready');
    }, demStatus);
}

async function loadDemFromZarr(entries) {
    await withBusy(async () => {
        setCardStatus(demStatus, 'Reading Zarr store…');
        const store = await ZarrStore.fromEntries(entries);
        const sites = store.sites;
        if (!sites.length) throw new Error('This Zarr store contains no sites.');

        resetDependentState();
        state.demSource = { kind: 'zarr', name: store.rootName, store, site: null, dem: null };

        zarrSiteDropdown.innerHTML = sites
            .map(site => `<option value="${site}">${site}</option>`).join('');
        zarrSiteRow.classList.remove('d-none');
        setZoomRowVisible(false);
    }, demStatus);

    if (state.demSource?.kind === 'zarr') {
        await selectZarrSite(zarrSiteDropdown.value);
    }
}

async function selectZarrSite(site) {
    const source = state.demSource;
    if (source?.kind !== 'zarr' || !site) return;

    await withBusy(async () => {
        // Record the structure first so the scenario dropdown stays usable even
        // if the DEM array itself cannot be decoded.
        source.site = site;
        source.dem = null;
        state.releaseSource = null;
        populateScenarioDropdown(source.store, site);

        setCardStatus(demStatus, `Loading DEM for site "${site}"…`);
        const raw = await source.store.readSiteDem(site);
        source.dem = { ...raw, ...deriveGeoreference(raw) };

        await applySources();
        plotDem(sim);
        setRunStatus('DEM ready.', 'ready');
    }, demStatus);
}

/** Derives cell size and bounds from the Zarr coordinate arrays. */
function deriveGeoreference({ width, height, x, y }) {
    const spacing = coords => (coords && coords.length > 1)
        ? Math.abs(coords[1] - coords[0])
        : null;
    const cellSize = spacing(x) ?? spacing(y) ?? 1.0;

    const xmin = x ? Math.min(x[0], x[x.length - 1]) : 0;
    const xmax = x ? Math.max(x[0], x[x.length - 1]) : width * cellSize;
    const ymin = y ? Math.min(y[0], y[y.length - 1]) : 0;
    const ymax = y ? Math.max(y[0], y[y.length - 1]) : height * cellSize;

    return { cellSize, bounds: { xmin, xmax, ymin, ymax } };
}

function populateScenarioDropdown(store, site) {
    const scenarios = store.scenariosOf(site);
    if (!scenarios.length) {
        zarrScenarioRow.classList.add('d-none');
        return;
    }
    zarrScenarioDropdown.innerHTML =
        '<option value="">— none (calculate instead) —</option>' +
        scenarios.map(s => `<option value="${s}">${s}</option>`).join('');
    zarrScenarioRow.classList.remove('d-none');
}

// ---------------------------------------------------------------------------
// Step 2: Release areas
// ---------------------------------------------------------------------------

async function loadReleaseFromEntries(entries) {
    if (!hasUsableDem()) {
        setCardStatus(releaseStatus, 'Select a DEM in step 1 first.', 'error');
        return;
    }

    if (looksLikeZarr(entries)) {
        setCardStatus(releaseStatus,
            'Zarr release areas are chosen with the scenario dropdown after selecting a site in step 1.',
            'error');
        return;
    }

    const supported = entries.filter(e => RELEASE_FILE_EXTENSIONS.includes(fileExtension(e.path)));
    if (!supported.length) {
        setCardStatus(releaseStatus,
            `Unsupported file. Use ${RELEASE_FILE_EXTENSIONS.map(e => '.' + e).join(', ')}.`,
            'error');
        return;
    }

    const file = supported[0].file;
    const ext = fileExtension(file.name);
    if (ext === 'gpx') {
        setCardStatus(releaseStatus,
            'GPX files define the DEM extent, not release areas. Use "Calculate release areas" instead.',
            'error');
        return;
    }

    await withBusy(async () => {
        setCardStatus(releaseStatus, `Reading "${file.name}"…`);
        const bytes = new Uint8Array(await file.arrayBuffer());
        state.releaseSource = { kind: 'raster', name: file.name, bytes, ext };
        zarrScenarioDropdown.value = '';

        await applySources();
        await sim.prepare();
        showReleaseAreas();
        setRunStatus('Release areas loaded. Ready to run.', 'ready');
    }, releaseStatus);
}

async function selectZarrScenario(scenario) {
    const source = state.demSource;
    if (source?.kind !== 'zarr') return;

    if (!scenario) {
        state.releaseSource = null;
        refreshWorkflowState();
        return;
    }

    await withBusy(async () => {
        setCardStatus(releaseStatus, `Loading scenario "${scenario}"…`);
        const { data, width, height } =
            await source.store.readScenarioReleaseAreas(source.site, scenario);

        if (width !== source.dem.width || height !== source.dem.height) {
            throw new Error(
                `Scenario grid ${width}x${height} does not match the DEM ` +
                `${source.dem.width}x${source.dem.height}.`);
        }

        state.releaseSource = { kind: 'zarr', name: source.name, scenario, data };
        await applySources();
        await sim.prepare();
        showReleaseAreas();
        setRunStatus('Release areas loaded. Ready to run.', 'ready');
    }, releaseStatus);
}

async function calculateReleaseAreas() {
    if (!hasUsableDem()) return;
    await withBusy(async () => {
        setCardStatus(releaseStatus, 'Calculating release areas…');
        state.releaseSource = { kind: 'calculated' };
        zarrScenarioDropdown.value = '';

        await applySources();
        await sim.prepare();
        showReleaseAreas();
        setRunStatus('Release areas calculated. Ready to run.', 'ready');
    }, releaseStatus);
}

function showReleaseAreas() {
    return showVariable('release_areas');
}

/**
 * Selects a plot variable and awaits the update. The engine rejects overlapping
 * calls, so plot refreshes must never run concurrently with other WASM work.
 */
async function showVariable(name) {
    plotVariable.value = name;
    if (!sim || !hasUsableDem()) return;
    await updatePlots(sim, name);
}

// ---------------------------------------------------------------------------
// Step 4: Running, and the shortcut card
// ---------------------------------------------------------------------------

/** Runs an async action with the UI locked and failures surfaced in a card. */
async function withBusy(action, statusElement) {
    if (state.busy) return;
    setBusy(true);
    try {
        await action();
    } catch (error) {
        console.error(error);
        const message = error?.message ?? String(error);
        if (statusElement) setCardStatus(statusElement, message, 'error');
        setRunStatus(message, 'error');
    } finally {
        setBusy(false);
        refreshWorkflowState();
    }
}

async function runSimulation() {
    if (!hasUsableDem()) {
        setRunStatus('Select a DEM in step 1 first.', 'error');
        return;
    }

    await withBusy(async () => {
        setRunStatus('Running simulation…', 'running');
        await applySources();

        simTimer = new Timer('AvalancheSimulation');
        await sim.run();
        simTimer.checkpoint('simulation');

        await sim.fetch_results();
        simTimer.checkpoint('fetching data');

        // Make containers visible before plotting so Plotly computes full widths.
        state.hasResults = true;
        updateResultPlotsVisibility();

        const timestepData = await sim.get_timestep_data();
        await plotTimestepData(timestepData);
        await plotTrajectory(timestepData);
        plotTimer();

        await showVariable('peak_velocity');
        resizeVisibleResultPlots();
        setRunStatus('Simulation finished.', 'ready');
    }, null);
}

/**
 * Writes the results as a Zarr store into a folder chosen by the user. The
 * picker has to run inside the click handler to keep the user activation.
 */
async function saveResults() {
    if (!state.hasResults || state.busy) return;

    if (typeof window.showDirectoryPicker !== 'function') {
        setCardStatus(saveStatus,
            'This browser cannot write folders. Use Chrome or Edge over HTTPS or localhost.', 'error');
        return;
    }

    let target;
    try {
        target = await window.showDirectoryPicker({ mode: 'readwrite' });
    } catch (error) {
        if (error?.name !== 'AbortError') {
            setCardStatus(saveStatus, error?.message ?? String(error), 'error');
        }
        return;
    }

    await withBusy(async () => {
        setCardStatus(saveStatus, 'Writing Zarr store…');
        const files = await sim.save_results_zarr();

        for (const file of files) {
            await writeStoreFile(target, file.path, file.bytes);
        }
        const targetName = target?.name ?? 'selected folder';
        setCardStatus(saveStatus, `Saved ${files.length} files to "${targetName}".`, 'ok');
    }, saveStatus);
}

async function writeStoreFile(root, path, bytes) {
    const parts = path.split('/');
    let directory = root;
    for (const part of parts.slice(0, -1)) {
        directory = await directory.getDirectoryHandle(part, { create: true });
    }
    const handle = await directory.getFileHandle(parts[parts.length - 1], { create: true });
    const writable = await handle.createWritable();
    await writable.write(bytes);
    await writable.close();
}

async function prepareSimulation() {
    if (!hasUsableDem()) return;
    await withBusy(async () => {
        setRunStatus('Preparing simulation…', 'running');
        await applySources();
        await sim.prepare();
        await showVariable('slope_aspect');
        setRunStatus('Prepared. Ready to run.', 'ready');
    }, null);
}

function scrollToSimulationSection() {
    const runCard = document.getElementById('runCard');
    const plotTarget = document.getElementById('demPlot');
    const target = runCard || plotTarget;
    if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function collapseDemReleaseGroup() {
    if (demReleaseGroup instanceof HTMLDetailsElement) {
        demReleaseGroup.open = false;
    }
}

function moveDemPlotTo(host) {
    if (!host || !demPlotContainer || demPlotContainer.parentElement === host) return;
    host.appendChild(demPlotContainer);
    if (demPlotElement?.classList.contains('js-plotly-plot')) {
        requestAnimationFrame(() => Plotly.Plots.resize(demPlotElement));
    }
}

function updateDemPlotPlacement() {
    if (!plotVariable || !demPlotStickyHost || !demPlotFlowHost) return;

    if (!state.hasResults) {
        moveDemPlotTo(demPlotStickyHost);
        return;
    }

    const placementAnchor = plotVariableAnchor || plotVariable;
    const plotControlsReached = placementAnchor.getBoundingClientRect().top < window.innerHeight;
    moveDemPlotTo(!desktopLayout.matches || plotControlsReached
        ? demPlotFlowHost
        : demPlotStickyHost);
}

async function loadExampleCase(run) {
    await withBusy(async () => {
        const name = demDropdown.value;
        localStorage.setItem('demDropdown', name);

        if (!run) {
            setLoadingScreenStatus('Engine ready.', `Loading example "${name}"…`);
        }

        resetDependentState();
        state.demSource = { kind: 'example', name };
        state.releaseSource = { kind: 'example' };
        zarrSiteRow.classList.add('d-none');
        setZoomRowVisible(false);

        await applySources();
        plotDem(sim);
        setRunStatus('Example loaded.', 'ready');

        if (!run) {
            setLoadingScreenStatus('Engine ready.', `Example "${name}" loaded.`);
            hideLoadingScreen();
        }
    }, demStatus);

    if (run && state.demSource?.kind === 'example') {
        scrollToSimulationSection();
        await runSimulation();
    }
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

runShortcutButton.addEventListener('click', async () => {
    collapseDemReleaseGroup();
    scrollToSimulationSection();
    await loadExampleCase(true);
});
// demDropdown.addEventListener('change', () => loadExampleCase(!isMobileDevice));
frictionModelDropdown.addEventListener('change', changeFrictionModel);

demDirButton.addEventListener('click', () => demDirInput.click());
demFileInput.addEventListener('change', async event => {
    const entries = entriesFromFileList(event.target.files);
    event.target.value = '';
    if (entries.length) await loadDemFromEntries(entries);
});
demDirInput.addEventListener('change', async event => {
    const entries = entriesFromFileList(event.target.files);
    event.target.value = '';
    if (entries.length) await loadDemFromEntries(entries);
});
setupDropZone(demDropZone, loadDemFromEntries, () => state.engineReady && !state.busy);
demDropZone.addEventListener('click', () => demFileInput.click());

zarrSiteDropdown.addEventListener('change', event => selectZarrSite(event.target.value));

releaseDirButton.addEventListener('click', () => releaseDirInput.click());
releaseFileInput.addEventListener('change', async event => {
    const entries = entriesFromFileList(event.target.files);
    event.target.value = '';
    if (entries.length) await loadReleaseFromEntries(entries);
});
releaseDirInput.addEventListener('change', async event => {
    const entries = entriesFromFileList(event.target.files);
    event.target.value = '';
    if (entries.length) await loadReleaseFromEntries(entries);
});
setupDropZone(releaseDropZone, loadReleaseFromEntries,
    () => state.engineReady && !state.busy && hasUsableDem());
releaseDropZone.addEventListener('click', () => {
    if (hasUsableDem()) releaseFileInput.click();
});

zarrScenarioDropdown.addEventListener('change', event => selectZarrScenario(event.target.value));
calculateReleaseButton.addEventListener('click', calculateReleaseAreas);

runButton.addEventListener('click', runSimulation);
prepareButton.addEventListener('click', prepareSimulation);
saveResultsButton.addEventListener('click', saveResults);

zoomLevelSlider.addEventListener('change', async () => {
    if (state.demSource?.kind !== 'gpx') return;
    const gpx = state.demSource.gpx;
    await withBusy(async () => {
        await dem.loadTiles(gpx, zoomLevelSlider.value);
        await applySources();
        plotDem(sim);
        plotGpx(gpx, dem);
    }, demStatus);
});

plotVariable.addEventListener('change', event => {
    if (!sim || !hasUsableDem()) return;
    updatePlots(sim, event.target.value).catch(console.error);
});

document.addEventListener('keydown', async event => {
    if (event.key === 'r' && !state.busy) await runSimulation();
});

let plotPlacementFrame = null;
function scheduleDemPlotPlacement() {
    if (plotPlacementFrame !== null) return;
    plotPlacementFrame = requestAnimationFrame(() => {
        plotPlacementFrame = null;
        updateDemPlotPlacement();
    });
}

window.addEventListener('scroll', scheduleDemPlotPlacement, { passive: true });
window.addEventListener('resize', scheduleDemPlotPlacement);
desktopLayout.addEventListener('change', updateDemPlotPlacement);
updateDemPlotPlacement();

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

function withTimeout(promise, ms, label = "operation") {
    let timer;
    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    });
    return Promise.race([promise.finally(() => clearTimeout(timer)), timeout]);
}

function checkWebGPU() {
    if (!navigator.gpu) {
        alert("WebGPU is not supported in this browser. Please use a compatible browser like Chrome or Edge with WebGPU enabled.");
        throw new Error("WebGPU not supported");
    }
}

async function loadEngine() {
    try {
        setLoadingScreenStatus('Loading engine…', 'Waiting for WASM runtime.');
        statusEl.textContent = "Loading Engine...";
        window.wasm = await init();
        window.wasmDecoders = { blosc: decode_blosc_chunk, zstd: decode_zstd_chunk };

        setLoadingScreenStatus('Creating simulation…', 'Allocating engine resources.');
        statusEl.textContent = "Creating Simulation...";
        window.sim = await withTimeout(WasmSimulation.new(), 5000, "WasmSimulation.new");

        setLoadingScreenStatus('Engine ready.', 'Loading example…');
        statusEl.textContent = "Engine ready.";
        state.engineReady = true;
    } catch (err) {
        console.error("WASM init failed:", err);
        const msg = err instanceof Error ? err.message : String(err);
        statusEl.textContent = `Engine load failed: ${msg}, check console for details.`;
        statusEl.style.backgroundColor = "rgba(255, 0, 0, 0.8)";
        setLoadingScreenStatus('Engine failed to load.', msg);
        setRunStatus('Engine failed to load.', 'error');
        throw err;
    }
}

async function main() {
    const savedCase = localStorage.getItem('demDropdown');
    if (savedCase) demDropdown.value = savedCase;

    changeFrictionModel();
    refreshWorkflowState();

    checkWebGPU();
    await loadEngine();

    setRunStatus('Engine ready. Load an example or select a DEM.', 'ready');
    await loadExampleCase(false);
}

main().catch(console.error);
