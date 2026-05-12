import init, { WasmSimulation } from "./pkg/avalanchers.js";


var release_points;
var release_point;
var simSettings;
var gpx;
var isExample = true;
var tiles = [];
window.dem = new Dem();
window.sim = null;
window.wasm = null;

const wasm = await init();
const demDropdown = document.getElementById('demDropdown');
const frictionModelDropdown = document.getElementById('frictionModelDropdown');

const stepSlider = document.getElementById('stepSlider');
const stepSliderValue = document.getElementById('stepSliderValue');
stepSlider.addEventListener('input', () => {
    stepSliderValue.textContent = stepSlider.value;
});

const cflSlider = document.getElementById('cflSlider');
const cflSliderValue = document.getElementById('cflSliderValue');
cflSlider.addEventListener('input', () => {
    cflSliderValue.textContent = cflSlider.value;
});

const frictionCoefficientSlider = document.getElementById('frictionCoefficientSlider');
const frictionCoefficientValue = document.getElementById('frictionCoefficientValue');
frictionCoefficientSlider.addEventListener('input', () => {
    frictionCoefficientValue.textContent = frictionCoefficientSlider.value;
});
const dragCoefficientSlider = document.getElementById('dragCoefficientSlider');
const dragCoefficientValue = document.getElementById('dragCoefficientValue');
dragCoefficientSlider.addEventListener('input', () => {
    dragCoefficientValue.textContent = dragCoefficientSlider.value;
});
const releasedParticlesPerCellSlider = document.getElementById('releasedParticlesPerCellSlider');
const releasedParticlesPerCellValue = document.getElementById('releasedParticlesPerCellValue');
releasedParticlesPerCellSlider.addEventListener('input', () => {
    releasedParticlesPerCellValue.textContent = releasedParticlesPerCellSlider.value;
});

const zoomLevelSlider = document.getElementById('zoomLevelSlider');
const zoomLevelValue = document.getElementById('zoomLevelValue');
zoomLevelValue.textContent = zoomLevelSlider.value + ' Resolution: ' + pixelWidthMeters(zoomLevelSlider.value, 47.2).toFixed(2) + ' m';
zoomLevelSlider.addEventListener('change', () => {
    zoomLevelValue.textContent = zoomLevelSlider.value + ' Resolution: ' + pixelWidthMeters(zoomLevelSlider.value, 47.2).toFixed(2) + ' m';
    dem.loadTiles(gpx, zoomLevelSlider.value).then(() => {
        plotDem(dem);
        plotGpx(gpx)
        simSettings.setDem(dem);
    });

});
zoomLevelSlider.addEventListener('input', () => {
    zoomLevelValue.textContent = zoomLevelSlider.value + ' Resolution: ' + pixelWidthMeters(zoomLevelSlider.value, 47.2).toFixed(2) + ' m';
});

demDropdown.addEventListener('change', async (event) => {
    predefinedReleasePoints = true;
    const selectedFile = event.target.value;
    localStorage.setItem('demDropdown', selectedFile);
    simSettings = getSettings();
    await sim.create(simSettings);
    isExample = true;
    plotDem(sim);
    if (!isMobileDevice) {
        runAndPlot();
    }
});

frictionModelDropdown.addEventListener('change', (event) => {
    changeFrictionModel();
});

function changeFrictionModel() {
    const selectedModel = frictionModelDropdown.selectedOptions[0].text;
    if (selectedModel == 'Coulomb') {
        frictionCoefficientSlider.value = 0.4663;
        frictionCoefficientValue.textContent = frictionCoefficientSlider.value;
    } else {
        frictionCoefficientSlider.value = 0.155;
        frictionCoefficientValue.textContent = frictionCoefficientSlider.value;
    }
    if (selectedModel == 'Coulomb' || selectedModel == 'samosAT') {
        dragCoefficientSlider.disabled = true;
    } else {
        dragCoefficientSlider.disabled = false;
    }
}
function setSettingsDisabled(flag) {
    const controls = document.querySelectorAll('#simSettingsDiv input, #simSettingsDiv select, #simSettingsDiv textarea, #simSettingsDiv button');
    controls.forEach(el => el.disabled = flag);
    runButton.disabled = flag;
    prepareButton.disabled = flag;
    changeFrictionModel();
    if (flag) {
        runButton.textContent = "Running...";
    } else {
        runButton.textContent = "Run Simulation";
    }
}

// const exportResultsCheckbox = document.getElementById('exportResults');
// exportResultsCheckbox.addEventListener('change', async (event) => {
//     if (!event.target.checked) return; // Only act on 'checked' state
//     try {
//         if (!directoryHandle) {
//             await setExportDirectory();
//         }
//         await exportResults();
//     } catch (error) {
//         alert("Failed to export results:", error);
//     }
// });


// Enable all
const simSettingsDiv = document.getElementById('simSettingsDiv')
const runButton = document.getElementById('runSimulation')
const prepareButton = document.getElementById('prepareSimulation')
runButton.addEventListener('click', async () => {
    await runAndPlot();
});
prepareButton.addEventListener('click', async () => {
    await run(simSettings, dem, release_point, predefinedReleasePoints);
    plotVariable.value = 'slopeAspect';
    plotVariable.dispatchEvent(new Event('change'));
});

async function runAndPlot() {
    console.log('Run simulation');
    setSettingsDisabled(true);
    try {
        simSettings = getSettings();
        if (!isExample) {
            delete simSettings.dem_path;
            delete simSettings.release_areas_path;
            await sim.create(simSettings);
            await sim.set_dem(dem.data1d,
                dem.width,
                dem.height,
                dem.cellSize,
                dem.bounds.xmin, dem.bounds.xmax, dem.bounds.ymin, dem.bounds.ymax,
                dem.mapFactor);
        }
        else {
            await sim.create(simSettings);
        }
        simTimer = new Timer('AvalancheSimulation');
        await sim.run();
        simTimer.checkpoint('simulation');
        await sim.get_timestep_data();
        // await sim.fetch_peak_flow_thickness();
        await sim.fetch_cell_count();
        simTimer.checkpoint('fetching data');
        plotTimestepData(sim);
        plotTrajectory(sim);
        // plotVariable.value = 'peak_flow_thickness';
        plotVariable.value = 'cell_count';
        plotVariable.dispatchEvent(new Event('change'));
        plotTimer();
        // if (exportResultsCheckbox.checked) {
        //     await exportResults();
        // }
    } catch (error) {
        console.error('Error during simulation:', error);
    }
    setSettingsDisabled(false);
}

const plotVariable = document.getElementById('plotVariable');
plotVariable.addEventListener('change', async (event) => {
    const selectedVariable = event.target.value;
    updatePlots(sim, selectedVariable)
});

document.addEventListener('keydown', async function (event) {
    console.log('Key pressed:', event.key);

    if (event.key === 'Enter') {
        console.log('Enter was pressed!');
    }

    if (event.key === 'r') {
        await runAndPlot();
    }
});

window.addEventListener('DOMContentLoaded', () => {
    const savedFile = localStorage.getItem('demDropdown');
    if (savedFile) {
        demDropdown.value = savedFile;
    }
});

function getSettings() {
    const simSettings = {
        dem_path: window.location.pathname.replace(/\/[^\/]+\.[^\/]+$/, "/") + "data/avaframe/" + demDropdown.value + ".png",
        release_areas_path: window.location.pathname.replace(/\/[^\/]+\.[^\/]+$/, "/") + "data/avaframe/" + demDropdown.value + "releaseTexture.png",
        max_steps: parseInt(stepSlider.value),
        sim_model: 0,
        friction_model: frictionModelDropdown.selectedIndex,
        density: 200,
        slab_thickness: 1,
        friction_coefficient: parseFloat(frictionCoefficientSlider.value),
        drag_coefficient: parseInt(dragCoefficientSlider.value),
        cfl: parseFloat(cflSlider.value),
        released_particles_per_cell: parseInt(releasedParticlesPerCellSlider.value),
    };
    return simSettings;
}

async function saveFilePersistent() {
    try {
        // Create a Blob containing the data you want to save
        const textToSave = "Hello, world! This is my data.";
        const blob = new Blob([textToSave], { type: 'text/plain' });

        // Options for the save file picker
        const options = {
            suggestedName: 'my-data.txt',
            types: [
                {
                    description: 'Text Files',
                    accept: {
                        'text/plain': ['.txt'],
                    },
                },
            ],
        };

        // Show the save file picker and get a FileSystemFileHandle
        // This is where the user interacts and potentially grants persistent permission
        const fileHandle = await window.showSaveFilePicker(options);

        // Create a writable stream to write data to the file
        const writableStream = await fileHandle.createWritable();

        // Write the blob to the file
        await writableStream.write(blob);

        // Close the stream
        await writableStream.close();

        console.log('File saved successfully!');

    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('User cancelled the save operation.');
        } else {
            console.error('Error saving file:', error);
        }
    }
}

async function savePngFile(pngBlob) {
    let fileHandle = await getStoredHandle(); // Your own function to retrieve stored handle

    if (!fileHandle) {
        fileHandle = await window.showSaveFilePicker({
            suggestedName: 'image.png',
            types: [{
                description: 'PNG Image',
                accept: { 'image/png': ['.png'] }
            }]
        });
        await storeHandle(fileHandle); // Your own function to persist the handle
    }

    // Check or request permission
    const permission = await fileHandle.queryPermission({ mode: 'readwrite' }) ||
        await fileHandle.requestPermission({ mode: 'readwrite' });

    if (permission !== 'granted') {
        throw new Error('Permission to write file denied.');
    }

    const writable = await fileHandle.createWritable();
    await writable.write(pngBlob);
    await writable.close();
}

// document.getElementById('exportResults').addEventListener('click', savePNG);

async function main() {
    changeFrictionModel();
    const settings = getSettings();
    await sim.create(settings);

    // const gpxString = await fetch('data/gpx/Nockspitze.gpx').then(response => response.text());
    // gpx = parseGPX(gpxString);
    // await dem.loadTiles(gpx, zoomLevelSlider.value)
    // await sim.set_dem(dem.data1d,
    //     dem.width,
    //     dem.height, 
    //     dem.cellSize, 
    //     dem.bounds.xmin, dem.bounds.xmax, dem.bounds.ymin, dem.bounds.ymax,
    //     dem.mapFactor);    

    plotDem(sim);
    // plotGpx(gpx, dem);
    if (!isMobileDevice) {
        runAndPlot();
    }
}

document.getElementById("gpxfile").addEventListener("change", async (e) => {
    isExample = false;
    predefinedReleasePoints = false;
    const file = e.target.files[0];
    if (!file) return;

    const gpxString = await file.text();
    tiles = [];
    gpx = parseGPX(gpxString);
    await dem.loadTiles(gpx, zoomLevelSlider.value)
    // simSettings.setDem(dem);
    await sim.set_dem(dem.data1d,
        dem.width,
        dem.height,
        dem.cell_size,
        dem.bounds.xmin, dem.bounds.xmax, dem.bounds.ymin, dem.bounds.ymax,
        dem.mapFactor);
    resetPlots();
    plotDem(sim);
    plotGpx(gpx, dem);
    if (!isMobileDevice) {
        runAndPlot();
    }
});

function withTimeout(promise, ms, label = "operation") {
    let timer;

    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => {
            reject(new Error(`${label} timed out after ${ms}ms`));
        }, ms);
    });

    return Promise.race([
        promise.finally(() => clearTimeout(timer)),
        timeout
    ]);
}

function checkWebGPU() {
    if (!navigator.gpu) {
        alert("WebGPU is not supported in this browser. Please use a compatible browser like Chrome or Edge with WebGPU enabled.");
        throw new Error("WebGPU not supported");
    }
}

var debug = false;
let predefinedReleasePoints = true;
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get("debug") === "vscode") {
    debug = true;
    console.log("Running in VS Code debug session");
}
var isMobileDevice = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);


checkWebGPU();
await loadEngine().catch(console.error);
main();
async function loadEngine() {
    const statusEl = document.getElementById("status");
    try {
        statusEl.textContent = "Loading Engine...";

        window.wasm = await init();

        statusEl.textContent = "Creating Simulation...";
        sim = await withTimeout(
                WasmSimulation.new(),
                5000,
                "WasmSimulation.new"
            );

    statusEl.textContent = "Engine Ready!";
    } catch (err) {
        console.error("WASM init failed:", err);

        let msg = "Unknown error";

        if (err instanceof Error) {
            msg = err.message;
        } else if (typeof err === "string") {
            msg = err;
        }

        statusEl.textContent = `Engine load failed: ${msg}, check console for details.`;
        statusEl.style.backgroundColor = "rgba(255, 0, 0, 0.8)";
    }
}
