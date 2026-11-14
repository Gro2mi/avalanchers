// Toast notifications for engine warnings and errors.
//
// Warnings flash up in yellow and fade out after WARNING_DURATION_MS;
// errors stay in red until the user closes them. A message that is already
// shown is not repeated: its counter goes up (and a warning's timer restarts).

const WARNING_DURATION_MS = 5000;
const FADE_MS = 400;
const MAX_TOASTS = 5;

let container = null;

function getContainer() {
    if (!container) {
        container = document.createElement('div');
        container.className = 'notice-stack';
        document.body.appendChild(container);
    }
    return container;
}

function removeToast(toast) {
    if (!toast.isConnected || toast.classList.contains('notice-leaving')) return;
    clearTimeout(toast.hideTimer);
    toast.classList.add('notice-leaving');
    // fallback for reduced motion, where no transition ends
    const remove = () => toast.remove();
    toast.addEventListener('transitionend', remove, { once: true });
    setTimeout(remove, FADE_MS + 100);
}

function scheduleFade(toast) {
    clearTimeout(toast.hideTimer);
    toast.hideTimer = setTimeout(() => removeToast(toast), WARNING_DURATION_MS);
}

/**
 * Shows `message` as a toast; `level` is "warn" or "error" (others are shown
 * like warnings).
 */
export function showToast(level, message) {
    const isError = level === 'error';
    const stack = getContainer();

    const existing = [...stack.children].find(toast =>
        !toast.classList.contains('notice-leaving')
        && toast.dataset.level === (isError ? 'error' : 'warn')
        && toast.dataset.message === message);
    if (existing) {
        const count = Number(existing.dataset.count) + 1;
        existing.dataset.count = String(count);
        existing.querySelector('.notice-count').textContent = `×${count}`;
        if (!isError) scheduleFade(existing);
        return;
    }

    const toast = document.createElement('div');
    toast.className = `notice ${isError ? 'notice-error' : 'notice-warning'}`;
    toast.dataset.level = isError ? 'error' : 'warn';
    toast.dataset.message = message;
    toast.dataset.count = '1';
    toast.setAttribute('role', isError ? 'alert' : 'status');

    const label = document.createElement('strong');
    label.className = 'notice-label';
    label.textContent = isError ? 'Error' : 'Warning';
    const text = document.createElement('span');
    text.className = 'notice-message';
    text.textContent = message;
    const count = document.createElement('span');
    count.className = 'notice-count';
    toast.append(label, text, count);

    if (isError) {
        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'notice-close';
        close.setAttribute('aria-label', 'Close');
        close.textContent = '×';
        close.addEventListener('click', () => removeToast(toast));
        toast.append(close);
    } else {
        // hovering keeps a warning readable
        toast.addEventListener('mouseenter', () => clearTimeout(toast.hideTimer));
        toast.addEventListener('mouseleave', () => scheduleFade(toast));
        scheduleFade(toast);
    }

    stack.append(toast);
    // too many: drop the oldest warning, else the oldest toast
    const shown = [...stack.children].filter(t => !t.classList.contains('notice-leaving'));
    if (shown.length > MAX_TOASTS) {
        removeToast(shown.find(t => t.dataset.level === 'warn') ?? shown[0]);
    }
}
