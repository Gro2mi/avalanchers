function to2D(flatArray, width, height) {
    const matrix = [];
    for (let i = 0; i < height; i++) {
        // .subarray creates a new view without copying the underlying data
        matrix.push(flatArray.subarray(i * width, (i + 1) * width));
    }
    return matrix;
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