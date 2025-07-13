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

async function plotDem(dem) {
    try {
        const surfaceDem = {
            x: dem.x,
            y: dem.y,
            z: dem.z,
            type: 'surface',
            colorscale: 'Earth',
            cmin: 0,
            cmax: 3000,
            // lighting: {
            //     ambient: 0.1,      // less ambient = darker shadows
            //     diffuse: 0.4,      // more diffuse = softer shadows
            //     specular: 1.0,     // strong highlights
            //     roughness: 0.7,    // lower = shinier
            //     fresnel: 0.0       // optional for reflectivity
            // },
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
function updatePlots(selectedVariable) {
    if (selectedVariable === 'elevation') {
        Plotly.update(demPlot, {
            surfacecolor: [dem.data],
            colorscale: 'Earth',
            cmin: [0],
            cmax: [3000],
            colorbar: {
                title: 'Elevation (m)'
            },
            lighting: resetLighting,
        });
        return;
    }
    var traceHist = {
        type: 'histogram',
        x: simData[selectedVariable].flat(),
        autobinx: true, // or set fixed bin settings
    };

    const layoutHist = {
        title: `Histogram of ${selectedVariable}`,
        template: plotly_dark,
    };
    Plotly.update(demPlot, {
        surfacecolor: [simData[selectedVariable]],  // new data
        cmin: [null],                               // reset min
        cmax: [null],                               // reset max
        colorscale: ['Portland'],
        colorbar: {
            title: plotVariable.options[plotVariable.selectedIndex].text
        },
        lighting: resetLighting,

    });
    if (selectedVariable === 'slopeAspect') {
        Plotly.update(demPlot, {
            surfacecolor: [simData[selectedVariable]],
            // colorscale: cyclicAspectColorscale,
            cmin: [null],
            cmax: [null],
            colorbar: {
                title: 'Aspect (°)'
            },
            lighting: [resetLighting],
        }, [0]);

        traceHist.x = simData[selectedVariable].flat().filter((val, index) => (Math.abs(val) < 1) && (dem.data1d[index] > 0));
    } else if (selectedVariable === 'cellCount') {
        const transformedSurfaceColor = simData[selectedVariable].map(row =>
            row.map(val => Math.log10(val))
        );
        Plotly.update(demPlot, {
            surfacecolor: [transformedSurfaceColor],
            colorscale: ['Portland'],
            cmin: [null],                               // reset min
            cmax: [null],                               // reset max
            colorbar: {
                title: plotVariable.options[plotVariable.selectedIndex].text
            },
            lighting: [resetLighting],
        }, {
            'scene.colorbar.title.text': 'Log10(Cell Count)',
            'scene.colorbar.title.font.color': 'red'
        }[0]);

        traceHist.x = simData[selectedVariable].flat().filter(val => val > 0).map(val => Math.log10(val));
    } else if (selectedVariable === 'velocityField') {
        traceHist.x = simData[selectedVariable].flat().filter(val => val > 0)
    }

    Plotly.react(histogramPlot, [traceHist], layoutHist);
}

function plotGpx(gpx) {
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

function plotTrajectory(xminBounds, yminBounds, mapFactor) {
    const lineTrace = {
        type: 'scatter3d',
        mode: 'line+markers',
        x: simData.position.x.map(val => val * mapFactor + xminBounds),
        y: simData.position.y.map(val => val * mapFactor + yminBounds),
        // Offset elevation by 5 units to visually separate the trajectory from the DEM surface
        z: simData.elevation.map((val) => (val + 5)),
        marker: {
            size: 3,
            color: simData.velocityMagnitude,
            colorscale: 'Bluered',
            cmin: Math.min(...simData.velocityMagnitude),
            cmax: Math.max(...simData.velocityMagnitude),
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

function plotOutput() {
    let n = simData.timestep.length;
    let x = simData.time;
    const friction = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.accelerationFrictionMagnitude,
        name: 'Friction Acceleration',
        visible: 'legendonly',
    };
    const tangential = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.accelerationTangentialMagnitude,
        name: 'Tangential Acceleration',
        visible: 'legendonly',
    };
    const dt = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.timestep,
        name: 'Timestep',
        visible: 'legendonly',
    };
    const traceCfl = {
        type: 'scatter',
        mode: 'lines',
        // first element is zero due to velocity being zero at the start
        x: x.slice(1, n - 2),
        y: simData.cfl.slice(1, n - 2),
        name: 'CFL',
        visible: 'legendonly',
    };
    const traceVelocityMagnitude = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.velocityMagnitude,
        name: 'Velocity Magnitude',
        visible: 'legendonly',
    };

    const tracePositionZ = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.position.z,
        name: 'Position Z',
        visible: 'legendonly',
    };

    const traceElevation = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        // last elevation point is outside the domain
        y: simData.elevation.slice(0, n - 1),
        name: 'Elevation',
        visible: 'legendonly',
    };
    const tracePositionZError = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: subtractArr(simData.elevation, simData.position.z).slice(0, n - 1),
        name: 'Position Z Error',
        visible: 'legendonly',
    };
    const traceDiffElevation = {
        type: 'scatter',
        mode: 'lines',
        x: x.slice(0, n - 2),
        y: diff(simData.elevation).slice(0, n - 1),

        name: 'Diff Elevation',
        visible: 'legendonly',
    };
    const traceDiffZ = {
        type: 'scatter',
        mode: 'lines',
        x: x.slice(0, n - 2),
        y: diff(simData.position.z).slice(0, n - 1),

        name: 'Diff Position Z',
        visible: 'legendonly',
    };
    const traceNormalX = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.normal.x,

        name: 'Normal X',
        visible: 'legendonly',
    };
    const traceNormalY = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.normal.y,

        name: 'Normal Y',
        visible: 'legendonly',
    };
    const traceNormalZ = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.normal.z,

        name: 'Normal Z',
        visible: 'legendonly',
    };
    const traceStepDistance = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.stepDistance,

        name: 'Step Distance',
        visible: 'legendonly',
    };
    const traceGEff = {
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: simData.gEff,

        name: 'g_eff',
        // visible: 'legendonly',
    };
    let layout = {
        template: plotly_dark,
        updatemenus: [{
            buttons: [
                {
                    method: 'restyle',
                    args: ['x', [simData.travelDistance]],
                    label: 'Travel Distance [m]'
                },
                {
                    method: 'restyle',
                    args: ['x', [simData.time]],
                    label: 'Time [s]'
                },
                {
                    method: 'restyle',
                    args: ['x', [Array.from({ length: n }, (_, i) => i)]],
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
        traceNormalX,
        traceNormalY,
        traceNormalZ,
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
