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
        this.world_resolution = null; // meters per pixel
        this.map_resolution = null;
    }

    async loadPNGAsFloat32(casename) {
        const { rgba, width, height } = await loadPNG('avaframe/' + casename + '.png');
        this.width = width;
        this.height = height;
        this.data1d = new Float32Array(this.width * this.height);

        const temp = new ArrayBuffer(4);
        const view = new DataView(temp);
        for (let i = 0; i < this.width * this.height; i++) {
            const offset = i * 4;
            view.setUint8(0, rgba[offset]);
            view.setUint8(1, rgba[offset + 1]);
            view.setUint8(2, rgba[offset + 2]);
            view.setUint8(3, rgba[offset + 3]);
            this.data1d[i] = view.getFloat32(0, true); // little endian
        }

        this.create2DData();
        this.bounds = await fetchBounds(casename);
        this.world_resolution = (this.bounds.xmax - this.bounds.xmin) / (this.width - 1);
        this.x = linspace(this.bounds.xmin, this.bounds.xmax, this.width);
        this.y = linspace(this.bounds.ymin, this.bounds.ymax, this.height);
        console.log("Loaded PNG ", casename, ":", this.width, "x", this.height);
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
        this.world_resolution = pixelWidthMeters(zoom, (bbox.maxLat - bbox.minLat) / 2 + bbox.minLat);
        this.map_resolution = pixelWidthMeters(zoom, 0);
        this.create2DData();
        this.x = linspace(this.bounds.xmin, this.bounds.xmax, this.width)//.reverse();
        this.y = linspace(this.bounds.ymin, this.bounds.ymax, this.height)//.reverse();
    }

    create2DData() {
        this.data = to2DArray(this.data1d, this.width, this.height);
        this.z = this.data.map(row => row.map(val => (val < 1 ? null : val)));
    }

    boundsFloat32() {
        return new Float32Array([
            this.bounds.xmin, this.bounds.ymin,
            this.bounds.xmax, this.bounds.ymax,
        ]);
    }

    getIndex(pt) {
        const dx = (pt.x - this.bounds.xmin) / this.map_resolution;
        const dy = (pt.y - this.bounds.ymin) / this.map_resolution; // Y is flipped in images
        return { x: dx, y: dy };
    }

    interpolateElevation(pt) {
        const { x, y } = this.getIndex(pt);
        const z = bilinearInterpolate(x, y, this.data);
        return { ...pt, z };
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

class SimInfo {
    static byteSize = 2 * 4;
    constructor(buffer) {
        const view = new DataView(buffer);
        this.stepCount = view.getUint32(0, true);
        this.dxyMin = view.getFloat32(4, true);
    }
}

class Particle {
    static byteSize = 12 * 4;
    constructor() {
    this.position = { x: 0, y: 0, z: 0 };
    this.mass = 0;
    this.velocity = { x: 0, y: 0, z: 0 };
    this.C = { xx: 0, xy: 0, yy: 0, xx: 0 }; // Assuming C is a 2x2 matrix stored as a flat array
    }

}

class SimSettings {
    static byteSize = 15 * 4;
    constructor() {
    }

    async set(casename, maxSteps, simModel, frictionModel, density, slabThickness, frictionCoefficient, dragCoefficient, cfl, releasedParticlesPerCell) {
        this.casename = casename;

        this.maxSteps = maxSteps;
        this.simModel = simModel;
        this.frictionModel = frictionModel;
        this.density = density;
        this.slabThickness = slabThickness;
        this.frictionCoefficient = frictionCoefficient;
        this.dragCoefficient = dragCoefficient;
        this.cfl = cfl;
        this.cellSize = dem.world_resolution;
        this.releasedParticlesPerCell = releasedParticlesPerCell;
        
        this.minSlopeAngle = 35;
        this.maxSlopeAngle = 45;
        this.minElevation = 1500;
        this.velocityThreshold = 1e-6;
        this.roughnessThreshold = 0.01;  // TODO is this high enough?
    }

    createBuffer() {
        this.cellSize = dem.world_resolution;
        let settingsU32 = new Uint32Array([
            this.maxSteps,
            this.simModel,
            this.frictionModel,
            this.releasedParticlesPerCell,
        ]);
        let settingsF32 = new Float32Array([
            this.density,
            this.slabThickness,
            this.frictionCoefficient,
            this.dragCoefficient,
            this.cfl,
            this.cellSize,
            this.minSlopeAngle,
            this.maxSlopeAngle,
            this.minElevation,
            this.velocityThreshold,
            this.roughnessThreshold
        ]);
        const settingsBufferData = new ArrayBuffer(SimSettings.byteSize);
        const settingsBufferU32 = new Uint32Array(settingsBufferData);
        const settingsBufferF32 = new Float32Array(settingsBufferData);
        settingsBufferU32.set(settingsU32);
        settingsBufferF32.set(settingsF32, settingsU32.length);
        return settingsBufferData;
    }
}

class SimData {
    static timeStepByteSize = 96;
    constructor(dxyMin) {
        this.timestep = [];
        this.time = [];
        this.accelerationFrictionMagnitude = [];
        this.accelerationTangentialMagnitude = [];
        this.velocityMagnitude = [];
        this.position = { x: [], y: [], z: [] };
        this.normal = { x: [], y: [], z: [] };
        this.accelerationTangential = { x: [], y: [], z: [] };
        this.velocity = { x: [], y: [], z: [] };
        this.elevation = [];
        this.uv = { x: [], y: [] };
        this.stepDistance = [];
        this.travelDistance = [];
        this.cfl = [];
        this.dxyMin = dxyMin;
        this.texture;
        this.velocityTexture;
        this.releaseThickness = [];
        this.slopeAspect = [];
        this.roughness = [];
        this.slopeAngle = [];
        this.cellCount = [];
        this.gpxArea = null;
        this.releasePredictor = null;
        this.velocityField = [];
        this.gEff = [];
    }

    cleanFloatArray(data) {
        return nullifyDomainBorder(data.map((row, y) =>
            row.map((value, x) => (dem.data[y][x] > 0.1 ? value : null))
        ), dem.height, dem.width).map(row => Array.from(row));;
    }

    //   releasePoints = transposeAndTo2DArray(releasePoints, dem.width, dem.height);
    parseSlopeTexture(slopeAngle, slopeAspect, windShelterIndex) {
        this.slopeAngle = this.cleanFloatArray(slopeAngle);
        this.slopeAspect = this.cleanFloatArray(slopeAspect);
        this.roughness = this.cleanFloatArray(windShelterIndex);
    }
    parseReleaseTexture(slabThickness, gpxArea, releasePredictor) {
        this.releaseSlabThickness = this.cleanFloatArray(slabThickness);
        this.gpxArea = this.cleanFloatArray(gpxArea);
        this.releasePredictor = this.cleanFloatArray(releasePredictor);
    }
    parseRoughnessTexture(roughness, forest) {
        this.roughness = this.cleanFloatArray(roughness);
        this.forest = this.cleanFloatArray(forest);
    }
    parseVelocityTexture(velocityTexture) {
        this.velocityField = to2DArray(velocityTexture, dem.width, dem.height).map(typedArr => Array.from(typedArr));
    }
    parseCellCountTexture(texture) {
        this.cellCount = to2DArray(texture.map(value => value), dem.width, dem.height).map(typedArr => Array.from(typedArr));
    }

    addDataExplicit(timestep, time, accelerationFrictionMagnitude, accelerationTangentialMagnitude, velocityMagnitude, position, elevation) {
        this.timestep.push(timestep);
        this.time.push(time);
        this.accelerationFrictionMagnitude.push(accelerationFrictionMagnitude);
        this.accelerationTangentialMagnitude.push(accelerationTangentialMagnitude);
        this.velocityMagnitude.push(velocityMagnitude);
        this.position.x.push(position.x);
        this.position.y.push(position.y);
        this.position.z.push(position.z);
        this.elevation.push(elevation);
        this.uv.x.push(uv.x);
        this.uv.y.push(uv.y);
    }
    parse(bufferData, nTimesteps, numberTrackedTrajectories) {
        for (let i = 0; i < nTimesteps; i++) {
            let baseOffset = numberTrackedTrajectories * i * SimData.timeStepByteSize / 4;
            this.addData(bufferData, baseOffset);
        }
    }

    addData(bufferData, baseOffset = 0) {
        this.velocity.x.push(bufferData[baseOffset + 0]);
        this.velocity.y.push(bufferData[baseOffset + 1]);
        this.velocity.z.push(bufferData[baseOffset + 2]);

        this.timestep.push(bufferData[baseOffset + 3]);

        this.accelerationTangential.x.push(bufferData[baseOffset + 4]);
        this.accelerationTangential.y.push(bufferData[baseOffset + 5]);
        this.accelerationTangential.z.push(bufferData[baseOffset + 6]);

        this.accelerationFrictionMagnitude.push(bufferData[baseOffset + 7]);

        this.position.x.push(bufferData[baseOffset + 8]);
        this.position.y.push(bufferData[baseOffset + 9]);
        this.position.z.push(bufferData[baseOffset + 10]);

        this.elevation.push(bufferData[baseOffset + 11]);

        this.normal.x.push(bufferData[baseOffset + 12]);
        this.normal.y.push(bufferData[baseOffset + 13]);
        this.normal.z.push(bufferData[baseOffset + 14]);

        this.gEff.push(bufferData[baseOffset + 22]);

        this.uv.x.push(bufferData[baseOffset + 16]);
        this.uv.y.push(bufferData[baseOffset + 17]);
        let n = this.timestep.length - 1
        this.accelerationTangentialMagnitude.push(magnitudeVecArr(this.accelerationTangential, n));
        this.velocityMagnitude.push(magnitudeVecArr(this.velocity, n));
        if (n === 0) {
            this.time.push(0);
            this.stepDistance.push(0);
            this.travelDistance.push(0);
            this.cfl.push(0);
        } else {
            this.time.push(this.time[n - 1] + this.timestep[n]);
            this.stepDistance.push(magnitude(subtract(this.position, n, this.position, n - 1)));
            this.travelDistance.push(this.travelDistance[n - 1] + this.stepDistance[n]);
            this.cfl.push(this.velocityMagnitude[n] * this.timestep[n] / this.dxyMin);
        }
    }
}

function countLines(str) {
    if (!str) return 0;
    // Split by newline, count resulting array length
    return str.split('\n').length;
}

function max(arr) {
    arr = arr.flatMap(row => Array.from(row));
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > max && arr[i] !== null && arr[i] !== undefined && arr[i] !== NaN) {
            max = arr[i];
        }
    }
    return max;
}

function min(arr) {
    arr = arr.flatMap(row => Array.from(row));
    let min = Infinity;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] < min && arr[i] !== null && arr[i] !== undefined && arr[i] !== NaN) {
            min = arr[i];
        }
    }
    return min;
}

function mean(arr) {
    arr = arr.flatMap(row => Array.from(row));
    if (arr.length === 0) {
        throw new Error("Cannot calculate mean of an empty array");
    }
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] !== null || arr[i] !== undefined && arr[i] !== NaN) {
            sum += arr[i];
        }
    }
    return sum / arr.length;
}

function minPositiveValue(floatArray) {
    let min = Infinity;

    for (let i = 0; i < floatArray.length; i++) {
        const val = floatArray[i];
        if (val > 0 && val < min) {
            min = val;
        }
    }
    if (min === Infinity) {
        error("No positive values found in the array.");
    }
    return min; // return null if no positive values
}

async function loadDemBinary(url, width, height) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();

    const data = new Float32Array(buffer); // Assumes little-endian (true on most systems)
    if (data.length !== width * height) {
        throw new Error(`Size mismatch: expected ${width * height} floats, got ${data.length}`);
    }

    // Optional: convert to 2D array
    const heightmap2D = [];
    for (let y = 0; y < height; y++) {
        heightmap2D.push(data.slice(y * width, (y + 1) * width));
    }

    return heightmap2D;
}

async function loadDemJson(casename) {
    const response = await fetch('avaframe/' + casename + '.json');  // Path to your JSON file
    const jsonData = await response.json();
    return jsonData;
}

async function loadReleasePoints(casename) {
    const response = await fetch('avaframe/' + casename + '.rp');  // Path to your JSON file
    const jsonData = await response.json();
    return jsonData;
}

async function loadPNG(url) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();

    // Decode PNG to raw RGBA bytes
    const img = UPNG.decode(buffer);
    const width = img.width;
    const height = img.height;
    const rgba = new Uint8Array(UPNG.toRGBA8(img)[0]);
    return { rgba, width, height };
}
function to2DArray(arr1D, width, height) {
    if (arr1D.length !== width * height) {
        throw new Error("1D array length does not match width * height");
    }
    const arr2D = [];
    for (let row = 0; row < height; row++) {
        const start = row * width;
        const end = start + width;
        arr2D.push([...arr1D.slice(start, end)]);
    }
    return arr2D;
}

function linspace(start, end, num) {
    if (num === 1) return [start];
    const step = (end - start) / (num - 1);
    return Array.from({ length: num }, (_, i) => start + i * step);
}

async function fetchBounds(casename) {
    const res = await fetch('avaframe/' + casename + '.aabb');
    const text = await res.text();

    // Split by line, filter non-empty, parse as float
    const numbers = text
        .split(/\r?\n/)
        .filter(line => line.trim() !== "")
        .map(line => parseFloat(line));

    return new RegionBounds([...numbers]);
}

function subtract(vec1, n, vec2, m) {
    return {
        x: vec1.x[n] - vec2.x[m],
        y: vec1.y[n] - vec2.y[m],
        z: vec1.z[n] - vec2.z[m]
    };
}

function subtractArr(arr1, arr2) {
    if (!Array.isArray(arr1) || !Array.isArray(arr2)) {
        throw new Error("Both inputs must be arrays");
    }
    if (arr1.length !== arr2.length) {
        throw new Error("Arrays must be of the same length");
    }
    return arr1.map((value, index) => value - arr2[index]);
}

function s(arr1, arr2) {
    subtract(arr1, arr2);
}

function cumulativeSum(arr) {
    if (arr instanceof Float32Array) {
        arr = [...arr];
    }
    if (!Array.isArray(arr)) {
        throw new Error("Input must be an array");
    }
    const result = new Float32Array(arr.length);
    result[0] = arr[0];
    for (let i = 1; i < arr.length; i++) {
        result[i] = result[i - 1] + arr[i];
    }
    return result;
}

function add(arr1, arr2) {
    if (!Array.isArray(arr1) || !Array.isArray(arr2)) {
        throw new Error("Both inputs must be arrays");
    }
    if (arr1.length !== arr2.length) {
        throw new Error("Arrays must be of the same length");
    }
    return arr1.map((value, index) => value + arr2[index]);
}

function multiply(arr, scalar) {
    if (!Array.isArray(arr)) {
        throw new Error("Input must be an array");
    }
    return arr.map(value => value * scalar);
}

function divide(arr, scalar) {
    if (!Array.isArray(arr)) {
        throw new Error("Input must be an array");
    }
    if (scalar === 0) {
        throw new Error("Division by zero is not allowed");
    }
    return arr.map(value => value / scalar);
}

function magnitudeArr(vec3arr) {
    const count = vec3arr.x.length;
    const magnitude = new Float32Array(count);

    for (let i = 0; i < count; i++) {
        const x = vec3arr.x[i];
        const y = vec3arr.y[i];
        const z = vec3arr.z[i];
        magnitude[i] = magnitude([x, y, z]);
    }
    return magnitude;
}
function magnitudeVecArr(vec3arr, n) {
    const x = vec3arr.x[n];
    const y = vec3arr.y[n];
    const z = vec3arr.z[n];
    return magnitude({ x, y, z });
}

function magnitude(vec) {
    return Math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

function diff(obj) {
    if (Array.isArray(obj)) {
        const diffArr = [];
        for (let i = 1; i < obj.length; i++) {
            diffArr.push(obj[i] - obj[i - 1]);
        }
        return diffArr;
    } else if (obj.hasOwnProperty('x') && obj.hasOwnProperty('y') && obj.hasOwnProperty('z')) {
        const diffObj = { x: [], y: [], z: [] };
        const count = obj.x.length;
        for (let i = 1; i < count; i++) {
            diffObj.x.push(obj.x[i] - obj.x[i - 1]);
            diffObj.y.push(obj.y[i] - obj.y[i - 1]);
            diffObj.z.push(obj.z[i] - obj.z[i - 1]);
        }
        return diffObj;
    }
    else {
        throw new Error("Input must be an array or an object with x, y, z properties");
    }

}

function nullifyBorders(array2D) {
    const height = array2D.length;
    const width = array2D[0].length;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (y === 0 || y === height - 1 || x === 0 || x === width - 1) {
                array2D[y][x] = null;
            }
        }
    }

    return array2D;
}
function nullifyDomainBorder(array2D) {
    const height = array2D.length;
    const width = array2D[0].length;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (y === 0 || y === height - 1 || x === 0 || x === width - 1) {
                array2D[y][x] = null;
            }
            if (dem.data[y][x] === null) {
                array2D[y][x] = null;
                if (y > 0) {
                    array2D[y - 1][x] = null;
                }
                if (y < height - 1) {
                    array2D[y + 1][x] = null;
                }
                if (x > 0) {
                    array2D[y][x - 1] = null;
                }
                if (x < width - 1) {
                    array2D[y][x + 1] = null;
                }
            }

        }
    }

    return array2D;
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

function decodeFloat16(bits) {
    const s = (bits & 0x8000) >> 15;       // sign
    const e = (bits & 0x7C00) >> 10;       // exponent
    const f = bits & 0x03FF;               // fraction

    if (e === 0) {
        // Subnormal
        return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
    }
    if (e === 0x1F) {
        // Inf or NaN
        return f === 0 ? (s ? -Infinity : Infinity) : NaN;
    }

    // Normalized
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / Math.pow(2, 10));
}

function flipFlatArrayInY(src, width, height) {
    const rowSize = width * 4 * src.constructor.BYTES_PER_ELEMENT; // 4 bytes per pixel (RGBA)
    const flipped = new src.constructor(src.length);

    for (let y = 0; y < height; y++) {
        const srcOffset = y * rowSize;
        const dstOffset = (height - 1 - y) * rowSize;
        flipped.set(src.subarray(srcOffset, srcOffset + rowSize), dstOffset);
    }
    return flipped;
}