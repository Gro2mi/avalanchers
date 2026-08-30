/**
 * File ingestion helpers for the simulation frontend.
 *
 * Handles drag & drop, file/directory pickers, and reading Zarr v3 stores that
 * are laid out as <store>/<site>/<scenario>, where the DEM lives at the site
 * level and the release areas live inside each scenario.
 */

const DEM_FILE_EXTENSIONS = ['gpx', 'tif', 'tiff', 'asc'];
const RELEASE_FILE_EXTENSIONS = ['gpx', 'tif', 'tiff', 'asc'];

/** Name of the DEM array stored at the site level of a Zarr store. */
const ZARR_DEM_ARRAY = 'dem';
/** Name of the release area array stored inside a Zarr scenario. */
const ZARR_RELEASE_ARRAY = 'release_area';

function fileExtension(name) {
    const match = /\.([^./\\]+)$/.exec(name || '');
    return match ? match[1].toLowerCase() : '';
}

function relativePathOf(file) {
    return file.webkitRelativePath || file.relativePath || file.name;
}

/**
 * Recursively collects files from a dropped FileSystemEntry so that dropping a
 * whole `.zarr` folder works and not just individual files.
 */
async function readEntryRecursive(entry, prefix, out) {
    if (!entry) return;
    if (entry.isFile) {
        const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
        out.push({ file, path: prefix + entry.name });
        return;
    }
    if (!entry.isDirectory) return;

    const reader = entry.createReader();
    // readEntries only returns a partial batch, so it must be drained in a loop.
    while (true) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) {
            await readEntryRecursive(child, prefix + entry.name + '/', out);
        }
    }
}

/** Normalizes a drop event into a flat list of `{ file, path }` entries. */
async function entriesFromDataTransfer(dataTransfer) {
    const items = Array.from(dataTransfer.items || []);
    const canUseEntries = items.length > 0 && typeof items[0].webkitGetAsEntry === 'function';

    if (canUseEntries) {
        const roots = items
            .filter(item => item.kind === 'file')
            .map(item => item.webkitGetAsEntry());
        const out = [];
        for (const entry of roots) {
            await readEntryRecursive(entry, '', out);
        }
        if (out.length) return out;
    }

    return Array.from(dataTransfer.files || []).map(file => ({ file, path: relativePathOf(file) }));
}

/** Normalizes a `<input type="file">` FileList into `{ file, path }` entries. */
function entriesFromFileList(fileList) {
    return Array.from(fileList || []).map(file => ({ file, path: relativePathOf(file) }));
}

function looksLikeZarr(entries) {
    return entries.some(e => e.path.split('/').pop() === 'zarr.json');
}

/**
 * Wires a drop zone. `onEntries` receives the normalized `{ file, path }` list.
 */
function setupDropZone(element, onEntries, isEnabled) {
    if (!element) return;

    const setDragState = active => element.classList.toggle('is-dragover', active);

    ['dragenter', 'dragover'].forEach(type => {
        element.addEventListener(type, event => {
            event.preventDefault();
            event.stopPropagation();
            if (isEnabled && !isEnabled()) {
                event.dataTransfer.dropEffect = 'none';
                return;
            }
            event.dataTransfer.dropEffect = 'copy';
            setDragState(true);
        });
    });

    ['dragleave', 'dragend'].forEach(type => {
        element.addEventListener(type, event => {
            event.preventDefault();
            event.stopPropagation();
            setDragState(false);
        });
    });

    element.addEventListener('drop', async event => {
        event.preventDefault();
        event.stopPropagation();
        setDragState(false);
        if (isEnabled && !isEnabled()) return;
        const entries = await entriesFromDataTransfer(event.dataTransfer);
        if (entries.length) await onEntries(entries);
    });
}

// ---------------------------------------------------------------------------
// Zarr v3 reading
// ---------------------------------------------------------------------------

const ZARR_DTYPES = {
    bool: { ctor: Uint8Array, bytes: 1 },
    int8: { ctor: Int8Array, bytes: 1 },
    int16: { ctor: Int16Array, bytes: 2 },
    int32: { ctor: Int32Array, bytes: 4 },
    int64: { ctor: BigInt64Array, bytes: 8 },
    uint8: { ctor: Uint8Array, bytes: 1 },
    uint16: { ctor: Uint16Array, bytes: 2 },
    uint32: { ctor: Uint32Array, bytes: 4 },
    uint64: { ctor: BigUint64Array, bytes: 8 },
    // Half precision is widened to f32, which is what the engine consumes.
    float16: { ctor: Float32Array, bytes: 2 },
    float32: { ctor: Float32Array, bytes: 4 },
    float64: { ctor: Float64Array, bytes: 8 },
};

function float16ToFloat32(bits) {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits & 0x7C00) >> 10;
    const fraction = bits & 0x03FF;
    if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
    if (exponent === 0x1F) return fraction ? NaN : sign * Infinity;
    return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

async function decompressBytes(bytes, format) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream(format));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Blosc and zstd are decoded by the WASM engine, which `main.js` exposes here. */
function wasmDecoder(name) {
    const decoder = window.wasmDecoders?.[name];
    if (typeof decoder !== 'function') {
        throw new Error('The engine is still loading, so this chunk cannot be decoded yet.');
    }
    return decoder;
}

async function decodeChunkBytes(bytes, codecs) {
    let data = bytes;
    let endian = 'little';

    // Zarr v3 lists codecs in encode order, so decoding runs back to front.
    for (let i = codecs.length - 1; i >= 0; i--) {
        const codec = codecs[i];
        const config = codec.configuration || {};
        switch (codec.name) {
            case 'bytes':
                endian = config.endian || 'little';
                break;
            case 'blosc':
                data = wasmDecoder('blosc')(data);
                break;
            case 'zstd':
                data = wasmDecoder('zstd')(data);
                break;
            case 'gzip':
                data = await decompressBytes(data, 'gzip');
                break;
            case 'zlib':
                data = await decompressBytes(data, 'deflate');
                break;
            case 'crc32c':
                data = data.subarray(0, data.length - 4);
                break;
            case 'transpose':
                throw new Error('Zarr "transpose" codec is not supported by this frontend.');
            default:
                throw new Error(`Unsupported Zarr codec: "${codec.name}".`);
        }
    }
    return { data, endian };
}

function bytesToTypedArray(bytes, dtype, endian) {
    const spec = ZARR_DTYPES[dtype];
    if (!spec) throw new Error(`Unsupported Zarr data type: "${dtype}".`);

    const count = Math.floor(bytes.length / spec.bytes);
    const littleEndian = endian !== 'big';

    if (dtype === 'float16') {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const out = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            out[i] = float16ToFloat32(view.getUint16(i * 2, littleEndian));
        }
        return out;
    }

    const out = new spec.ctor(count);

    if (spec.bytes === 1) {
        return new spec.ctor(bytes.buffer, bytes.byteOffset, count);
    }
    if (littleEndian) {
        // Copy into an aligned buffer so the typed array view is always valid.
        const aligned = new Uint8Array(count * spec.bytes);
        aligned.set(bytes.subarray(0, count * spec.bytes));
        return new spec.ctor(aligned.buffer);
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const getter = {
        int16: 'getInt16', int32: 'getInt32', int64: 'getBigInt64',
        uint16: 'getUint16', uint32: 'getUint32', uint64: 'getBigUint64',
        float32: 'getFloat32', float64: 'getFloat64',
    }[dtype];
    for (let i = 0; i < count; i++) {
        out[i] = view[getter](i * spec.bytes, false);
    }
    return out;
}

/**
 * A Zarr v3 store backed by browser `File` objects.
 *
 * Sites are the groups directly below the store root, scenarios are the groups
 * below a site.
 */
class ZarrStore {
    constructor(rootName, files) {
        this.rootName = rootName;
        /** @type {Map<string, File>} store-relative path -> File */
        this.files = files;
        /** @type {Map<string, object>} store-relative path -> parsed zarr.json */
        this.nodes = new Map();
    }

    static async fromEntries(entries) {
        const metaEntry = entries.find(e => e.path.split('/').pop() === 'zarr.json');
        if (!metaEntry) {
            throw new Error('No zarr.json found. Select the root folder of a Zarr store.');
        }

        // The store root is the shortest path that still contains a zarr.json.
        const rootDepth = Math.min(
            ...entries
                .filter(e => e.path.split('/').pop() === 'zarr.json')
                .map(e => e.path.split('/').length - 1)
        );
        const rootSegments = metaEntry.path.split('/').slice(0, rootDepth);
        const rootPrefix = rootSegments.length ? rootSegments.join('/') + '/' : '';

        const files = new Map();
        for (const entry of entries) {
            if (!entry.path.startsWith(rootPrefix)) continue;
            files.set(entry.path.slice(rootPrefix.length), entry.file);
        }

        const store = new ZarrStore(rootSegments[rootSegments.length - 1] || 'zarr store', files);
        await store.#indexNodes();
        return store;
    }

    async #indexNodes() {
        const metaPaths = [...this.files.keys()].filter(p => p.split('/').pop() === 'zarr.json');
        await Promise.all(metaPaths.map(async path => {
            const groupPath = path.slice(0, Math.max(0, path.length - 'zarr.json'.length - 1));
            try {
                this.nodes.set(groupPath, JSON.parse(await this.files.get(path).text()));
            } catch (e) {
                console.warn(`Skipping unreadable Zarr metadata at "${path}"`, e);
            }
        }));
    }

    #childGroups(parentPath) {
        const depth = parentPath === '' ? 1 : parentPath.split('/').length + 1;
        const names = [];
        for (const [path, meta] of this.nodes) {
            if (path === '' || meta.node_type !== 'group') continue;
            if (path.split('/').length !== depth) continue;
            if (parentPath !== '' && !path.startsWith(parentPath + '/')) continue;
            names.push(path.split('/').pop());
        }
        return names.sort();
    }

    get sites() {
        return this.#childGroups('');
    }

    scenariosOf(site) {
        return site ? this.#childGroups(site) : [];
    }

    hasArray(path) {
        return this.nodes.get(path)?.node_type === 'array';
    }

    /** Reads a whole Zarr array into a flat typed array plus its shape. */
    async readArray(path) {
        const meta = this.nodes.get(path);
        if (!meta) throw new Error(`Zarr array "${path}" not found in this store.`);
        if (meta.node_type !== 'array') throw new Error(`Zarr node "${path}" is not an array.`);

        const shape = meta.shape;
        const chunkShape = meta.chunk_grid?.configuration?.chunk_shape;
        if (!chunkShape) throw new Error(`Zarr array "${path}" has an unsupported chunk grid.`);

        const spec = ZARR_DTYPES[meta.data_type];
        if (!spec) throw new Error(`Unsupported Zarr data type: "${meta.data_type}".`);

        const encoding = meta.chunk_key_encoding || { name: 'default' };
        const separator = encoding.configuration?.separator || (encoding.name === 'v2' ? '.' : '/');
        const usePrefix = encoding.name !== 'v2';

        const total = shape.reduce((a, b) => a * b, 1);
        const out = new spec.ctor(total);
        if (meta.fill_value) out.fill(meta.fill_value);

        const chunkCounts = shape.map((s, i) => Math.ceil(s / chunkShape[i]));
        const totalChunks = chunkCounts.reduce((a, b) => a * b, 1);

        for (let linear = 0; linear < totalChunks; linear++) {
            // Expand the linear chunk index into per-dimension chunk coordinates.
            const chunkCoord = [];
            let rest = linear;
            for (let d = chunkCounts.length - 1; d >= 0; d--) {
                chunkCoord[d] = rest % chunkCounts[d];
                rest = Math.floor(rest / chunkCounts[d]);
            }

            const key = (usePrefix ? ['c', ...chunkCoord] : chunkCoord).join(separator);
            const file = this.files.get(`${path}/${key}`);
            if (!file) continue; // Missing chunks fall back to the fill value.

            const raw = new Uint8Array(await file.arrayBuffer());
            const { data, endian } = await decodeChunkBytes(raw, meta.codecs || []);
            const chunk = bytesToTypedArray(data, meta.data_type, endian);

            this.#placeChunk(out, chunk, shape, chunkShape, chunkCoord);
        }

        return { data: out, shape };
    }

    /** Copies a decoded chunk into the correct region of the full array. */
    #placeChunk(out, chunk, shape, chunkShape, chunkCoord) {
        if (shape.length !== 2) {
            if (shape.length === 1) {
                out.set(chunk.subarray(0, Math.min(chunk.length, shape[0])), chunkCoord[0] * chunkShape[0]);
                return;
            }
            throw new Error(`Only 1D and 2D Zarr arrays are supported (got ${shape.length}D).`);
        }

        const [rows, cols] = shape;
        const [chunkRows, chunkCols] = chunkShape;
        const rowOffset = chunkCoord[0] * chunkRows;
        const colOffset = chunkCoord[1] * chunkCols;

        for (let r = 0; r < chunkRows; r++) {
            const targetRow = rowOffset + r;
            if (targetRow >= rows) break;
            const copyCols = Math.min(chunkCols, cols - colOffset);
            if (copyCols <= 0) break;
            const source = chunk.subarray(r * chunkCols, r * chunkCols + copyCols);
            out.set(source, targetRow * cols + colOffset);
        }
    }

    /** Reads the DEM of a site together with its coordinate arrays. */
    async readSiteDem(site) {
        const demPath = `${site}/${ZARR_DEM_ARRAY}`;
        if (!this.hasArray(demPath)) {
            throw new Error(`Site "${site}" does not contain a "${ZARR_DEM_ARRAY}" array.`);
        }
        const { data, shape } = await this.readArray(demPath);
        const [height, width] = shape;

        const x = await this.#readCoordinate(`${site}/x`, width);
        const y = await this.#readCoordinate(`${site}/y`, height);

        return { data: Float32Array.from(data), width, height, x, y };
    }

    /** Reads the release areas of a scenario. */
    async readScenarioReleaseAreas(site, scenario) {
        const path = `${site}/${scenario}/${ZARR_RELEASE_ARRAY}`;
        if (!this.hasArray(path)) {
            throw new Error(`Scenario "${scenario}" does not contain a "${ZARR_RELEASE_ARRAY}" array.`);
        }
        const { data, shape } = await this.readArray(path);
        const [height, width] = shape;
        return { data: Float32Array.from(data), width, height };
    }

    async #readCoordinate(path, expectedLength) {
        if (!this.hasArray(path)) return null;
        const { data } = await this.readArray(path);
        return data.length === expectedLength ? Float32Array.from(data) : null;
    }
}
