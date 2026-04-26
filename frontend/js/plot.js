const outputPlot = document.getElementById('outputPlot');
const demPlot = document.getElementById('demPlot');
const histogramPlot = document.getElementById('histogramPlot');

const resetLighting = {
    ambient: 0.8,
    diffuse: 0.8,
    specular: 0.05,
    roughness: 0.5,
    fresnel: 0.2,
}

function resetPlots() {
    Plotly.purge('outputPlot');
    Plotly.purge('demPlot');
    Plotly.purge('histogramPlot');
}

function plotDem(sim) {
    // DEM  coordinates have to be copied. Otherwise detached ArrayBuffer issues arise when restyling
    try {
        const surfaceDem = {
            x: new Float32Array(sim.x),
            y: new Float32Array(sim.y),
            z: to2D(new Float32Array(sim.dem), sim.width, sim.height),
            type: 'surface',
            colorscale: [[0, '#a5a5a5'], [1, '#a5a5a5']],
            showscale: false,

            lighting: {
                ambient: 0.6,   // Base brightness
                diffuse: 0.5,   // Defines the shape/shadows
                specular: 0.05, // Very low "shininess"
                roughness: 0.9
            },
            // cmin: 0,
            // cmax: 3000,
            contours: {
                z: {
                    show: true,
                    start: 0,
                    end: 4000,
                    size: 100,                 // Contours at 0, 100, 200, ..., 4000
                    color: 'white',
                    project: { z: false }
                },
            },
        };
        const data = [surfaceDem];

        const layout = {
            template: plotly_dark,
            scene: {
                aspectmode: 'data',
            }
        };
        Plotly.newPlot('demPlot', data, layout);
        demPlot.on('plotly_click', function (eventData) {
            const point = eventData.points[0];
            const i = point.pointNumber[0]; // column index (x)
            const j = point.pointNumber[1]; // row index (y)
            const x = point.x;
            const y = point.y;
            const z = point.z;

            console.log(`Clicked surface point at i=${i}, j=${j}, x=${x}, y=${y}, z=${z}`);
        });
    } catch (error) {
        console.error('Error loading or plotting data:', error);
    }
}
const cyclicAspectColorscale = [
    [0.0, 'blue'],     // 0° North
    [0.25, 'green'],    // 90° East
    [0.5, 'red'],      // 180° South
    [0.75, 'yellow'],   // 270° West
    [1.0, 'blue']      // 360° North again to close the loop
];

function createRandomMatrix2D(width, height) {
    const matrix = new Array(height);
    for (let i = 0; i < height; i++) {
        const row = new Array(width);
        for (let j = 0; j < width; j++) {
            row[j] = Math.random();
        }
        matrix[i] = row;
    }
    return matrix;
}

async function updatePlots(sim, selectedVariable) {
    // await sim.fetch_max_velocity();
    await sim.fetch_cell_count();
    sim.fetch_results();

    if (selectedVariable === 'elevation') {
        Plotly.restyle(demPlot, {
            surfacecolor: [to2D(new Float32Array(sim.dem), sim.width, sim.height)],
            colorscale: 'Earth',
            cmin: [0],
            cmax: [4000],
            colorbar: {
                title: 'Elevation (m)'
            },
            lighting: resetLighting,
        });
        return;
    }
    var traceHist = {
        type: 'histogram',
        x: sim[selectedVariable],
        autobinx: true, // or set fixed bin settings
    };

    const layoutHist = {
        title: `Histogram of ${selectedVariable}`,
        template: plotly_dark,
    };

    var plotOptions = {
        surfacecolor: [to2D(sim[selectedVariable], sim.width, sim.height)],
        showscale: true,
        colorscale: ['Portland'],
        cmin: [null],
        cmax: [null],
        colorbar: {
            title: selectedVariable,
        },
    };
    traceHist.x = sim[selectedVariable].filter((val, index) => (sim.dem[index] > 0));
    if (selectedVariable === 'cell_count') {
        const cellCountLog = new Float32Array(sim.cell_count).map(val => Math.log10(val));
        plotOptions.surfacecolor = [to2D(cellCountLog, sim.width, sim.height)];
        traceHist.x = cellCountLog.filter(val => val > 0);
    } else if (selectedVariable === 'max_velocity') {
        traceHist.x = traceHist.x.filter(val => val > 1e-5)
    }
    Plotly.restyle(demPlot, plotOptions, [0]);
    Plotly.react(histogramPlot, [traceHist], layoutHist);
}

function plotGpx(gpx, dem) {
    const webMercatorCoords = gpx.map(pt => latLonToWebMercator(pt.lat, pt.lon)).map(pt => dem.interpolateElevation(pt));

    dem.interpolateElevation(webMercatorCoords[0])
    const lineTrace = {
        type: 'scatter3d',
        mode: 'lines+markers',
        x: webMercatorCoords.map(pt => pt.x),
        y: webMercatorCoords.map(pt => pt.y),
        z: webMercatorCoords.map(pt => pt.z + 1 || 3000),
        marker: {
            size: 2,
        },
        line: {
            width: 4,
        },
        name: 'Route'
    };

    Plotly.addTraces(demPlot, [lineTrace]);
}

async function plotTrajectory(sim) {
    timestepData = await sim.get_timestep_data();
    const [xminBounds, yminBounds, mapFactor] = sim.demTrajectoryInfo;
    const lineTrace = {
        type: 'scatter3d',
        mode: 'line+markers',
        x: timestepData.position.filter((_, i) => i % 3 === 0).map(val => val * mapFactor + xminBounds),
        y: timestepData.position.filter((_, i) => i % 3 === 1).map(val => val * mapFactor + yminBounds),
        // Offset elevation by 5 units to visually separate the trajectory from the DEM surface
        z: timestepData.elevation.map((val) => (val + 5)),
        marker: {
            size: 3,
            color: timestepData.velocityMagnitude,
            colorscale: 'Bluered',
            cmin: Math.min(...timestepData.velocityMagnitude),
            cmax: Math.max(...timestepData.velocityMagnitude),
        },
        name: 'Trajectory'
    };
    try {
        if (demPlot.data) {
            const index = demPlot.data.findIndex(trace => trace.name === 'Trajectory');
            if (index !== -1) {
                // If the trace exists, remove it
                Plotly.deleteTraces(demPlot, index);
            }
        }
    } catch (TypeError) {
        // If the plotDiv.data is undefined, we can skip the deletion
        console.warn('demPlot.data is undefined, skipping trace deletion.');
    }

    Plotly.addTraces(demPlot, [lineTrace]);
}

async function plotTimestepData(sim) {
    const timestepData = await sim.get_timestep_data();
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
        y: new Float32Array(timestepData.cfl.slice(1, n - 2)),
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

    const traceElevation = {
        type: 'scatter',
        mode: 'lines',
        x: x.slice(0, n - 3),
        // last elevation point is outside the domain
        y: new Float32Array(timestepData.elevation),
        name: 'Elevation',
        visible: 'legendonly',
    };
    const positionZError = new Float32Array(n);
    for (let i = 1; i < n; i++) {
        positionZError[i] = timestepData.elevation[i] - timestepData.position[i * 3 + 2];
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
        diffElevation[i] = timestepData.elevation[i] - timestepData.elevation[i - 1];
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
    // // const traceNormalX = {
    // //     type: 'scatter',
    // //     mode: 'lines',
    // //     x: x,
    // //     y: timestepData.normal.x,

    // //     name: 'Normal X',
    // //     visible: 'legendonly',
    // // };
    // // const traceNormalY = {
    // //     type: 'scatter',
    // //     mode: 'lines',
    // //     x: x,
    // //     y: timestepData.normal.y,

    // //     name: 'Normal Y',
    // //     visible: 'legendonly',
    // // };
    // // const traceNormalZ = {
    // //     type: 'scatter',
    // //     mode: 'lines',
    // //     x: x,
    // //     y: timestepData.normal.z,

    // //     name: 'Normal Z',
    // //     visible: 'legendonly',
    // // };
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
        // // traceNormalX,
        // // traceNormalY,
        // // traceNormalZ,
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


function plotReleasePointsRGBA(releasePoints, width, height) {
    const slabMap = [];

    for (let y = 0; y < height; y++) {
        const row = [];
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const alpha = releasePoints[idx + 3]; // A channel
            row.push(alpha / 255); // Normalize to 0–1 if needed
        }
        slabMap.push(row);
    }

    const data = [{
        z: slabMap,
        type: 'heatmap',
        colorscale: 'Viridis', // or 'Jet', 'Greys', etc.
        colorbar: { title: 'Slab thickness (norm)' }
    }];

    const layout = {
        title: 'Release Points (Slab Thickness)',
        xaxis: { title: 'X' },
        yaxis: { title: 'Y', autorange: 'reversed' }, // flip vertically for image-like view
    };

    Plotly.newPlot('releasePointsPlot', data, layout);
}

function plotTimer() {
    // const x = ['Load Data', 'Process Data', 'Render UI', 'Finish'];
    //   const y = [12.34, 18.22, 7.89, 4.56]; // delta times in ms
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

function plotHistogram() {
    const data = [{
        x: simData.roughness.flat().filter(v => v > 0), // Flatten and filter out zero values
        type: 'histogram',
        // xbins: {
        //   size: 1  // Optional: bin width
        // }
    }];

    var layout = {
        // ...layout2d,
        title: 'Histogram Example',
        xaxis: { title: 'Value' },
        yaxis: { title: 'Count' },
        template: plotly_dark,
    };

    Plotly.newPlot('histogramPlot', data, layout);
}

function plotDebug(gpx) {

    const webMercatorCoords = gpx.map(pt => latLonToWebMercator(pt.lat, pt.lon)).map(pt => dem.interpolateElevation(pt));

    const trace = {
        x: webMercatorCoords.map(pt => pt.x),
        y: webMercatorCoords.map(pt => pt.y),
        mode: 'markers', // or 'lines' or 'lines+markers'
        type: 'scatter',
        marker: {
            color: 'red',
            size: 8
        },
        name: 'Data Points'
    };

    const layout = {
        title: '2D Scatter Plot',
        xaxis: {
            title: 'X Axis',
            autorange: 'reversed',
            scaleanchor: 'y'
        },
        yaxis: {
            title: 'Y Axis',
            autorange: 'reversed',
            scaleratio: 1,
        }
    };
    const bbox = {
        x: [dem.bounds.xmin, dem.bounds.xmax, dem.bounds.xmax, dem.bounds.xmin, dem.bounds.xmin],
        y: [dem.bounds.ymin, dem.bounds.ymin, dem.bounds.ymax, dem.bounds.ymax, dem.bounds.ymin],
        mode: 'lines', // or 'lines' or 'lines+markers'
        type: 'scatter',

    }
    const xmin = {
        x: [dem.bounds.xmin, dem.bounds.xmin],
        y: [dem.bounds.ymin, dem.bounds.ymax],
        mode: 'markers',
        type: 'scatter',
        marker: {
            color: 'blue',
            size: 8
        },
        name: 'xmin'
    }

    Plotly.newPlot('debugPlot', [trace, bbox, xmin], layout);
}
// plotDebug();
