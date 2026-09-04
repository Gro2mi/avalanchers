// Live 3D viewer for the simulation.
//
// Owns the canvas the WASM renderer presents to and the requestAnimationFrame
// loop that steps the simulation and redraws. The engine rejects overlapping
// calls, so every sim interaction goes through `trySim` and the loop pauses
// while other WASM work (plot refreshes, result fetches) is in flight.

const DPR_CAP = 2;
const ZOOM_SCALE = 0.01;
// Pinch zoom strength per pixel of finger spread; spreading ~150px doubles zoom.
const PINCH_ZOOM_SCALE = 0.05;

export class LiveViewer {
    constructor(sim, {
        canvas = document.getElementById('viewerCanvas'),
        card = document.getElementById('viewerCard'),
        overlaySelect = document.getElementById('viewerOverlay'),
        particlesToggle = document.getElementById('viewerParticles'),
        stepsSelect = document.getElementById('viewerSteps'),
        resetButton = document.getElementById('viewerReset'),
        statusElement = document.getElementById('viewerStatus'),
        isBusy = () => false,
    } = {}) {
        this.sim = sim;
        this.canvas = canvas;
        this.card = card;
        this.overlaySelect = overlaySelect;
        this.particlesToggle = particlesToggle;
        this.stepsSelect = stepsSelect;
        this.resetButton = resetButton;
        this.statusElement = statusElement;
        this.isBusy = isBusy;

        // Presentation settings: one timestep per rendered frame by default, and
        // a slight vertical exaggeration baked into the terrain mesh.
        this.stepsPerFrame = parseInt(stepsSelect?.value, 10) || 1;
        this.exaggeration = 1.3;
        this.running = false;
        this.onFinished = null;
        this.pendingResize = null;
        this.wasAttached = false;
        // On-demand rendering: frames are only drawn while the simulation runs
        // or after something changed (camera, overlay, resize). An idle view
        // schedules no rAF callbacks at all.
        this.needsRender = false;
        this.rafId = null;
        // True while a tick is awaiting the engine: a re-entrant tick would
        // borrow the simulation twice and panic in wasm-bindgen.
        this.tickInFlight = false;

        this.bindControls();
        this.bindPointer();
        this.observeResize();
    }

    /** Schedules one frame; keeps an already scheduled frame pending. */
    requestRender() {
        this.needsRender = true;
        if (this.rafId === null && !this.tickInFlight) {
            // Arrow wrapper: `this` must be bound lexically — this can fire from
            // observeResize during construction, before any explicit binding.
            this.rafId = requestAnimationFrame(() => this.tick());
        }
    }

    setStatus(message, kind = 'info') {
        if (!this.statusElement) return;
        this.statusElement.textContent = message;
        this.statusElement.className = 'source-summary ' +
            (kind === 'error' ? 'text-danger' : kind === 'ok' ? 'text-success' : 'text-secondary');
    }

    /** Runs a sim call, ignoring borrow conflicts while other engine work runs. */
    trySim(action) {
        if (!this.running && this.isBusy()) return undefined;
        try {
            return action();
        } catch (error) {
            console.debug('viewer: engine busy, dropped interaction', error);
            return undefined;
        }
    }

    bindControls() {
        this.overlaySelect?.addEventListener('change', () => {
            this.trySim(() => this.sim.set_overlay(this.overlaySelect.value));
            this.requestRender();
        });

        this.particlesToggle?.addEventListener('change', () => {
            this.trySim(() => this.sim.set_particles_visible(this.particlesToggle.checked));
            this.requestRender();
        });

        this.stepsSelect?.addEventListener('change', () => {
            this.stepsPerFrame = parseInt(this.stepsSelect.value, 10) || 1;
        });

        this.resetButton?.addEventListener('click', () => {
            this.trySim(() => this.sim.reset_view());
            this.requestRender();
        });
    }

    bindPointer() {
        // One pointer entry per active finger/mouse, by pointerId. A single
        // pointer orbits (mouse buttons pan instead); two pointers pinch-zoom
        // and pan by their centroid. `touch-action: none` keeps the browser
        // from stealing gestures for scrolling.
        const pointers = new Map();
        let mousePan = false;

        this.canvas.addEventListener('pointerdown', event => {
            this.canvas.setPointerCapture(event.pointerId);
            pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
            if (event.pointerType === 'mouse') {
                mousePan = event.button === 1 || event.button === 2 || event.shiftKey;
            }
        });

        this.canvas.addEventListener('pointermove', event => {
            const last = pointers.get(event.pointerId);
            if (!last) return;

            const dx = event.clientX - last.x;
            const dy = event.clientY - last.y;

            if (pointers.size >= 2 && event.pointerType !== 'mouse') {
                const other = [...pointers.entries()]
                    .find(([id]) => id !== event.pointerId)[1];
                const spread = Math.hypot(other.x - event.clientX, other.y - event.clientY)
                    - Math.hypot(other.x - last.x, other.y - last.y);
                // Fingers spreading apart zoom in, pinching together zooms out.
                this.trySim(() => this.sim.zoom(spread * PINCH_ZOOM_SCALE));
                // Each finger pans half its delta, so the pair tracks its centroid.
                this.trySim(() => this.sim.pan(dx / 2, dy / 2));
            } else {
                const pan = event.pointerType === 'mouse' && mousePan;
                this.trySim(() => pan ? this.sim.pan(dx, dy) : this.sim.orbit(dx, dy));
            }
            this.requestRender();

            last.x = event.clientX;
            last.y = event.clientY;
        });

        const release = event => {
            pointers.delete(event.pointerId);
            if (event.pointerType === 'mouse') mousePan = false;
        };
        this.canvas.addEventListener('pointerup', release);
        this.canvas.addEventListener('pointercancel', release);

        this.canvas.addEventListener('contextmenu', event => event.preventDefault());
        // Scroll up (negative deltaY) zooms in, matching the native viewer.
        this.canvas.addEventListener('wheel', event => {
            event.preventDefault();
            this.trySim(() => this.sim.zoom(-event.deltaY * ZOOM_SCALE));
            this.requestRender();
        }, { passive: false });
        this.canvas.addEventListener('dblclick', () => {
            this.trySim(() => this.sim.reset_view());
            this.requestRender();
        });
    }

    observeResize() {
        const observer = new ResizeObserver(() => this.sizeCanvas());
        observer.observe(this.canvas);
        this.sizeCanvas();
    }

    /** Sizes the canvas backing store to its CSS box; the surface follows in tick. */
    sizeCanvas() {
        const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP);
        const width = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
        const height = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
            this.pendingResize = { width, height };
            this.requestRender();
        }
    }

    /** Reveals the viewer card, for example when a live run starts. */
    reveal() {
        if (this.card instanceof HTMLDetailsElement) this.card.open = true;
        this.sizeCanvas();
    }

    /** Attaches (or re-attaches) the renderer; requires a prepared simulation. */
    async attach() {
        this.sizeCanvas();
        try {
            await this.sim.attach_renderer(this.canvas, this.exaggeration);
            this.applyPreferences();
            this.requestRender();
            this.setStatus(
                'Drag to orbit, right-drag to pan, scroll or pinch to zoom ' +
                '(touch: one finger orbits, two fingers pan and pinch). ' +
                'Keys: 0–7 switch overlays, P particles, +/- steps per frame, ' +
                'V reset view, R run.', 'ok');
            return true;
        } catch (error) {
            const message = error?.message ?? String(error);
            this.setStatus(`3D view unavailable: ${message}`, 'error');
            return false;
        }
    }

    applyPreferences() {
        this.trySim(() => this.sim.set_overlay(this.overlaySelect?.value ?? 'peak_velocity'));
        this.trySim(() => this.sim.set_particles_visible(this.particlesToggle?.checked ?? true));
    }

    /** Switches the overlay and keeps the dropdown in sync (keyboard shortcuts). */
    setOverlay(name) {
        if (this.overlaySelect) this.overlaySelect.value = name;
        this.trySim(() => this.sim.set_overlay(name));
        this.requestRender();
    }

    /** Toggles particle visibility and keeps the checkbox in sync. */
    toggleParticles() {
        const visible = !(this.particlesToggle?.checked ?? true);
        if (this.particlesToggle) this.particlesToggle.checked = visible;
        this.trySim(() => this.sim.set_particles_visible(visible));
        this.requestRender();
    }

    resetView() {
        this.trySim(() => this.sim.reset_view());
        this.requestRender();
    }

    /** Doubles the steps per frame (`+` key), capped at the dropdown's top option. */
    doubleSteps() {
        this.scaleSteps(2);
    }

    /** Halves the steps per frame (`-` key), floored at 1. */
    halveSteps() {
        this.scaleSteps(0.5);
    }

    scaleSteps(factor) {
        const optionValues = [...(this.stepsSelect?.options ?? [])]
            .map(option => parseInt(option.value, 10))
            .filter(Number.isFinite);
        const max = optionValues.length ? Math.max(...optionValues) : 256;
        const current = this.stepsPerFrame > 0 ? this.stepsPerFrame : 1;
        this.stepsPerFrame = Math.min(Math.max(Math.round(current * factor), 1), max);
        if (this.stepsSelect) this.stepsSelect.value = String(this.stepsPerFrame);
    }

    /**
     * Forgets the saved camera so the next attach frames the terrain afresh.
     * Called when a different DEM is loaded; must not be gated on busy state,
     * because DEM loads run inside a busy workflow.
     */
    forgetView() {
        try {
            this.sim.forget_view();
        } catch (error) {
            console.debug('viewer: could not forget view', error);
        }
        this.needsRender = false;
    }

    /** Resolves once the simulation finishes; keeps rendering afterwards. */
    runToCompletion() {
        return new Promise(resolve => {
            this.running = true;
            this.onFinished = resolve;
            this.requestRender();
        });
    }

    async tick() {
        // A previous tick may still be awaiting the engine; it reschedules at
        // the end if more work is pending. Re-entering now would borrow the
        // simulation twice — wasm-bindgen panics on that ("recursive use").
        if (this.tickInFlight) return;
        this.tickInFlight = true;
        this.rafId = null;
        try {
            if (this.running || !this.isBusy()) {
                const attached = this.sim.renderer_attached;
                if (this.wasAttached && !attached) {
                    this.setStatus('Prepare or run the simulation to reattach the 3D view.');
                    this.needsRender = false; // nothing to draw until re-attached
                }
                this.wasAttached = attached;

                if (attached && (this.running || this.needsRender)) {
                    if (this.pendingResize) {
                        const { width, height } = this.pendingResize;
                        this.pendingResize = null;
                        await this.sim.resize(width, height);
                    }
                    const wasRunning = this.running;
                    this.needsRender = false;
                    await this.sim.render_frame(wasRunning ? this.stepsPerFrame : 0);
                    // Resolve the run only after the final frame finished
                    // rendering: the continuation starts the background result
                    // job, which must not race this tick for the engine borrow.
                    if (wasRunning && this.sim.is_finished) {
                        this.running = false;
                        const finished = this.onFinished;
                        this.onFinished = null;
                        if (finished) finished();
                    }
                }
            }
        } catch (error) {
            console.debug('viewer: skipped frame', error);
        } finally {
            this.tickInFlight = false;
        }
        // Keep animating only while the run drives frames or a frame is pending;
        // an idle view stays at zero wakeups until the next interaction.
        if (this.running || this.needsRender) {
            this.rafId = requestAnimationFrame(() => this.tick());
        }
    }
}
