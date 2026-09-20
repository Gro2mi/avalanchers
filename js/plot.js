const outputPlot = document.getElementById('outputPlot');
const histogramPlot = document.getElementById('histogramPlot');

const mobilePlotMedia = window.matchMedia('(max-width: 991.98px)');

function resetPlots() {
    ['outputPlot', 'histogramPlot', 'timerPlot'].forEach(id => Plotly.purge(id));
}

/** Loads the cached data a variable needs, tolerating stages where it does not exist yet. */
async function ensureVariableFetched(sim, variable) {
    const fetchers = {
        release_areas: () => sim.fetch_release_areas(),
        slope_angle: () => sim.fetch_slope_angle(),
        slope_aspect: () => sim.fetch_slope_aspect(),
        roughness: () => sim.fetch_roughness(),
        peak_velocity: () => sim.fetch_peak_velocity(),
        peak_flow_thickness: () => sim.fetch_peak_flow_thickness(),
    };
    const fetcher = fetchers[variable];
    if (!fetcher) return true;
    try {
        await fetcher();
        return true;
    } catch (error) {
        console.warn(`"${variable}" is not available yet.`, error);
        return false;
    }
}

async function updatePlots(sim, selectedVariable) {
    if (!await ensureVariableFetched(sim, selectedVariable)) return;

    const values = new Float32Array(sim[selectedVariable]);
    if (values.length !== sim.width * sim.height) {
        console.warn(`"${selectedVariable}" has no data to plot yet.`);
        return;
    }

    // Restrict the histogram to cells that carry terrain; flow variables only
    // become meaningful above a noise floor.
    const demValues = new Float32Array(sim.dem);
    let histogramValues = values.filter((val, index) => demValues[index] > 0);
    if (selectedVariable === 'peak_velocity' || selectedVariable === 'peak_flow_thickness') {
        histogramValues = histogramValues.filter(val => val > 1e-5);
    }

    const layoutHist = {
        title: `Histogram of ${selectedVariable}`,
        template: plotly_dark,
    };

    Plotly.react(histogramPlot, [{ type: 'histogram', x: histogramValues, autobinx: true }], layoutHist);
}

async function plotTimestepData(timestepData) {

    let x = new Float32Array(timestepData.time);
    let n = timestepData.time.length;
    const friction = {
        type: 'scatter',
        mode: 'lines',
        x: x.slice(1, n),
        y: new Float32Array(timestepData.accelerationFrictionMagnitude).slice(1, n),
        name: 'Friction Acceleration',
        visible: 'legendonly',
    };
    const tangential = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: new Float32Array(timestepData.accelerationTangentialMagnitude),
        name: 'Tangential Acceleration',
        visible: 'legendonly',
    };
    const dt = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: new Float32Array(timestepData.timestep),
        name: 'Timestep',
        visible: 'legendonly',
    };
    const traceCfl = {
        type: 'scatter',
        mode: 'lines',
        // first element is zero due to velocity being zero at the start
        x: x.slice(1, n - 2),
        y: new Float32Array(timestepData.cfl).slice(1, n - 2),
        name: 'CFL',
        visible: 'legendonly',
    };
    const traceVelocityMagnitude = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: new Float32Array(timestepData.velocityMagnitude),
        name: 'Velocity Magnitude',
        visible: 'legendonly',
    };

    const tracePositionZ = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: new Float32Array(timestepData.position.z),
        name: 'Position Z',
        visible: 'legendonly',
    };
    const elevation = new Float32Array(timestepData.position.filter((_, i) => i % 3 === 2));
    const traceElevation = {
        type: 'scatter',
        mode: 'lines',
        x: x.slice(0, n - 3),
        y: elevation,
        name: 'Elevation',
        visible: 'legendonly',
    };
    const positionZError = new Float32Array(n);
    for (let i = 1; i < n; i++) {
        positionZError[i] = elevation[i] - timestepData.position[i * 3 + 2];
    }
    const tracePositionZError = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: positionZError,
        name: 'Position Z Error',
        visible: 'legendonly',
    };

    const diffElevation = new Float32Array(n);
    for (let i = 1; i < n; i++) {
        diffElevation[i] = elevation[i] - elevation[i - 1];
    }
    const traceDiffElevation = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: diffElevation,

        name: 'Diff Elevation',
        visible: 'legendonly',
    };
    const diffZ = new Float32Array(n);
    for (let i = 1; i < n; i++) {
        diffZ[i] = timestepData.position[i * 3 + 2] - timestepData.position[(i - 1) * 3 + 2];
    }
    const traceDiffZ = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: diffZ,

        name: 'Diff Position Z',
        visible: 'legendonly',
    };
    const traceStepDistance = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: new Float32Array(timestepData.stepDistance),

        name: 'Step Distance',
        visible: 'legendonly',
    };
    const traceGEff = {
        type: 'scatter',
        mode: 'lines',
        x: x.slice(1, n),
        y: new Float32Array(timestepData.gEff).slice(1, n),

        name: 'g_eff',
        // visible: 'legendonly',
    };
    let layout = {
        template: plotly_dark,
        updatemenus: [{
            buttons: [
                {
                    method: 'restyle',
                    args: ['x', [new Float32Array(timestepData.travelDistance)]],
                    label: 'Travel Distance [m]'
                },
                {
                    method: 'restyle',
                    args: ['x', [new Float32Array(timestepData.time)]],
                    label: 'Time [s]'
                },
                {
                    method: 'restyle',
                    args: ['x', [new Float32Array(Array.from({ length: n }, (_, i) => i))]],
                    label: 'Timestep [#]'
                }
            ],
            direction: 'up',
            showactive: true,
            x: 1,
            xanchor: 'right',
            y: 0,
            yanchor: 'top',
        }]
    };

    const traces = [
        friction,
        tangential,
        dt,
        traceCfl,
        traceVelocityMagnitude,
        tracePositionZ,
        traceElevation,
        tracePositionZError,
        traceDiffElevation,
        traceDiffZ,
        traceStepDistance,
        traceGEff,
    ]

    Plotly.newPlot('outputPlot', traces, layout).then(() => {
        // Restore visibility AFTER plot is rendered
        restoreTraceVisibility(outputPlot, traces);

        // Attach listener to save visibility changes
        if (!outputPlot._restyleListenerAdded) {
            outputPlot.on('plotly_restyle', () => {
                const visibility = outputPlot.data.map(trace => trace.visible ?? true);
                localStorage.setItem('traceVisibility', JSON.stringify(visibility));
                outputPlot._restyleListenerAdded = true;
            });
        }
    });
}

function restoreTraceVisibility(plotElement, traces) {
    const saved = localStorage.getItem('traceVisibility');
    if (!saved) return;

    const visibility = JSON.parse(saved);
    // Only apply if the number of visibilities matches the number of traces
    if (Array.isArray(visibility) && visibility.length === traces.length) {
        const update = { visible: visibility };
        Plotly.restyle(plotElement, update);
    } else {
        // Optionally clear invalid saved visibility
        localStorage.removeItem('traceVisibility');
        console.warn('Saved trace visibility does not match number of traces. Skipping restore.');
    }
}

function plotTimer() {
    const checkpoints = simTimer.getCheckpoints();

    const x = checkpoints.map(cp => cp.name);
    const y = checkpoints.map(cp => parseFloat(cp.delta));
    const data = [{
        type: "waterfall",
        x: x,
        y: y,
        textposition: "outside",
        text: y.map(v => v.toFixed(2) + " ms"),
        connector: {
            line: {
                color: "rgb(63, 63, 63)"
            }
        }
    }];

    const layout = {
        title: "Timer Checkpoints Waterfall",
        yaxis: {
            title: "Milliseconds",
            zeroline: false
        },
        template: plotly_dark,
    };

    Plotly.newPlot("timerPlot", data, layout);
}
