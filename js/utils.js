function to2D(flatArray, width, height) {
    const matrix = [];
    for (let i = 0; i < height; i++) {
        matrix.push(flatArray.subarray(i * width, (i + 1) * width));
    }
    return matrix;
}


function linspace(start, end, num) {
    if (num === 1) return [start];
    const step = (end - start) / (num - 1);
    return Array.from({ length: num }, (_, i) => start + i * step);
}


function bilinearInterpolate(x, y, grid) {
    const x0 = Math.floor(x);
    const x1 = Math.ceil(x);
    const y0 = Math.floor(y);
    const y1 = Math.ceil(y);

    if (x0 < 0 || x1 >= grid[0].length || y0 < 0 || y1 >= grid.length) return null;

    const q11 = grid[y0][x0];
    const q21 = grid[y0][x1];
    const q12 = grid[y1][x0];
    const q22 = grid[y1][x1];

    const fx = x - x0;
    const fy = y - y0;

    const r1 = q11 * (1 - fx) + q21 * fx;
    const r2 = q12 * (1 - fx) + q22 * fx;

    return r1 * (1 - fy) + r2 * fy;
}

class Dem {
    constructor() {
        this.data1d = null;
        this.width = 0;
        this.height = 0;
        this.bounds = null;
        this.data = null;
        this.x = [];
        this.y = [];
        this.z = [];
        this.cellSize = null; // meters per pixel
        this.mapFactor = null;
    }

    async loadTiles(gpx, zoom = 16, boundingBoxMargin = 100) {
        const bbox = await getGPXBoundingBoxWithMargin(gpx, boundingBoxMargin); // 500m margin
        console.log("GPX Bounding Box:", bbox);
        console.log("Distance:", haversineDistance(bbox.maxLat, bbox.maxLon, bbox.minLat, bbox.minLon));
        console.log("Distance X:", haversineDistance(bbox.maxLat, bbox.maxLon, bbox.maxLat, bbox.minLon));
        console.log("Distance Y:", haversineDistance(bbox.maxLat, bbox.maxLon, bbox.minLat, bbox.maxLon));
        console.log("Bounding Box:", bbox);
        const { tiles, nTilesX, nTilesY, bounds } = await fetchAndCacheTiles([bbox.minLat, bbox.minLon, bbox.maxLat, bbox.maxLon], zoom);
        const { data1d, width, height } = await stitchTilesCropped(tiles, 64, nTilesX, nTilesY);
        this.bounds = bounds;
        this.data1d = data1d;
        this.width = width;
        this.height = height;
        // approximate latitude for correction, assuming it to be constant
        this.cellSize = pixelWidthMeters(zoom, (bbox.maxLat - bbox.minLat) / 2 + bbox.minLat);
        this.mapFactor = pixelWidthMeters(zoom, 0) / this.cellSize;
        this.create2DData();
        this.x = linspace(this.bounds.xmin, this.bounds.xmax, this.width)//.reverse();
        this.y = linspace(this.bounds.ymin, this.bounds.ymax, this.height)//.reverse();
    }

    create2DData() {
        this.data = to2D(this.data1d, this.width, this.height);
        this.z = this.data.map(row => row.map(val => (val < 1 ? null : val)));
    }

    boundsFloat32() {
        return new Float32Array([
            this.bounds.xmin, this.bounds.ymin,
            this.bounds.xmax, this.bounds.ymax,
        ]);
    }

    getIndex(pt) {
        const dx = (pt.x - this.bounds.xmin) / (this.cellSize * this.mapFactor);
        const dy = (pt.y - this.bounds.ymin) / (this.cellSize * this.mapFactor);
        return { x: dx, y: dy };
    }

    interpolateElevation(pt) {
        const { x, y } = this.getIndex(pt);
        const z = bilinearInterpolate(x, y, this.data);
        return { ...pt, z };
    }
}

class RegionBounds {
    constructor(xmin, ymin, xmax, ymax) {
        if (Array.isArray(xmin)) {
            if (xmin.length !== 4) {
                throw new Error("RegionBounds expects an array of 4 numbers: [xmin, ymin, xmax, ymax]");
            }
            [this.xmin, this.ymin, this.xmax, this.ymax] = xmin;
        }
        else {
            this.xmin = xmin
            this.ymin = ymin;
            this.xmax = xmax;
            this.ymax = ymax;
        }
        this.width = this.xmax - this.xmin;
        this.height = this.ymax - this.ymin;
    }
}

class Timer {
    constructor(label) {
        this.label = label;
        this.start = performance.now();
        this.last = this.start;
        this.checkpoints = []; // array of { name, time, delta }
    }

    checkpoint(name, log = false) {
        const now = performance.now();
        const delta = now - this.last;
        this.checkpoints.push({ name, time: now, delta });
        if (log) {
            console.log(`${this.label} - ${name}: ${delta.toFixed(2)} ms`);
        }
        this.last = now;
    }

    getCheckpoints() {
        return this.checkpoints.map(cp => ({
            name: cp.name,
            timeSinceStart: (cp.time - this.start).toFixed(2),
            delta: cp.delta.toFixed(2),
        }));
    }

    printSummary() {
        console.log(`Timer "${this.label}" Summary:`);
        for (const cp of this.getCheckpoints()) {
            console.log(`  ${cp.name}: +${cp.delta} ms (total ${cp.timeSinceStart} ms)`);
        }
    }
}

var simTimer = new Timer('AvalancheSimulation');