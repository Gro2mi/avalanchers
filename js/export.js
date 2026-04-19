const DB_NAME = 'fileHandlesDB';
const STORE_NAME = 'handles';
const KEY = 'pngFile';

async function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            db.createObjectStore(STORE_NAME);
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function storeHandle(handle, key) {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).put(handle, key);
    return tx.complete;
}

async function getStoredHandle(key) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const request = store.get(key);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

directoryHandle = null;
(async () => {
    directoryHandle = await getStoredHandle("exportDirectory");
})();

async function setExportDirectory() {
    directoryHandle = await window.showDirectoryPicker();
    await storeHandle(directoryHandle, "exportDirectory");
}

async function exportResults() {
    if (!directoryHandle) {
        console.warn('Please set the export directory to export simulation results.');
        return;
    }
    const simDir = await directoryHandle.getDirectoryHandle(getTimestampedDirName(), { create: true });
    saveTextFile(simDir, 'results.txt', 'Simulation results go here...'); // Replace with actual content
    saveTextFile(simDir, 'simulation.log', 'Simulation log goes here...'); // Replace with actual log content
    savePNG(simDir, "velocity.png"); // Replace with actual PNG file name
}

async function saveTextFile(directory, fileName, content) {

    // Create or overwrite a file named "results.txt" inside the selected directory
    const fileHandle = await directory.getFileHandle(fileName, { create: true });

    const permission = await fileHandle.queryPermission({ mode: 'readwrite' }) ||
        await fileHandle.requestPermission({ mode: 'readwrite' });

    if (permission !== 'granted') {
        alert('Permission denied to write the file.');
        return;
    }

    // Create a writable stream
    const writable = await fileHandle.createWritable();

    // Write some dummy data (e.g., simulation results)
    const blob = new Blob([content], { type: 'text/plain' });
    await writable.write(blob);

    // Close the stream
    await writable.close();
}

function getTimestampedDirName() {
  const now = new Date();
  const date = now.toISOString().split('T')[0]; // "YYYY-MM-DD"
  const time = now.toTimeString().split(' ')[0].replace(/:/g, '-'); // "HH-MM-SS"

  return `${date}_${time}`; // e.g. "2025-07-16_14-30-15"
}

async function savePNG(directory, fileName) {
    const fileHandle = await directory.getFileHandle(fileName, { create: true });

    const permission = await fileHandle.queryPermission({ mode: 'readwrite' }) ||
        await fileHandle.requestPermission({ mode: 'readwrite' });

    if (permission !== 'granted') {
        alert('Permission denied to write the file.');
        return;
    }

    const writable = await fileHandle.createWritable();
    const blob = await createFakePNG(); // Replace with real PNG data if you want
    await writable.write(blob);
    await writable.close();
}

async function createFakePNG() {
    // This creates a 1x1 red pixel PNG blob
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'red';
    ctx.fillRect(0, 0, 1, 1);
    return new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
}

