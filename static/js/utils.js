/* ===== STATIC/JS/UTILS.JS ===== */
// Utility functions for IA Project UI

/**
 * Show alert message
 * @param {string} message - Alert message
 * @param {string} type - Alert type (success, danger, warning, info)
 * @param {number} duration - Duration in ms (0 = persistent)
 */
function showAlert(message, type = 'info', duration = 5000) {
    const alertId = 'alert-' + Date.now();
    const alertHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert" id="${alertId}">
            <i class="fas fa-info-circle"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    const container = document.getElementById('alertContainer');
    container.insertAdjacentHTML('beforeend', alertHTML);

    if (duration > 0) {
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                alert.remove();
            }
        }, duration);
    }
}

/**
 * Format number to fixed decimal places
 * @param {number} num - Number to format
 * @param {number} digits - Decimal places
 */
function formatNumber(num, digits = 4) {
    if (num === null || num === undefined) return '-';
    return parseFloat(num).toFixed(digits);
}

/**
 * Format large numbers with commas
 * @param {number} num - Number to format
 */
function formatLargeNumber(num) {
    if (num === null || num === undefined) return '-';
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Format timestamp
 * @param {string} timestamp - ISO timestamp
 */
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Get color based on value (for status indicators)
 * @param {number} value - Value (0-1)
 * @returns {string} Color code
 */
function getStatusColor(value) {
    if (value >= 0.9) return 'success';
    if (value >= 0.7) return 'info';
    if (value >= 0.5) return 'warning';
    return 'danger';
}

/**
 * Get status badge HTML
 * @param {number} value - Value (0-1)
 * @param {string} label - Label
 */
function getStatusBadge(value, label) {
    const color = getStatusColor(value);
    return `<span class="badge bg-${color}">${label}</span>`;
}

/**
 * Create a chart context
 * @param {string} canvasId - Canvas element ID
 * @returns {Chart|null} Chart instance or null
 */
function createChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;

    const ctx = canvas.getContext('2d');
    return new Chart(ctx, config);
}

/**
 * Debounce function
 * @param {function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showAlert('Copied to clipboard!', 'success', 2000);
    }).catch(err => {
        showAlert('Failed to copy', 'danger');
    });
}

/**
 * Validate form
 * @param {string} formId - Form ID
 * @returns {boolean} Form is valid
 */
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;
    return form.checkValidity();
}

/**
 * Initialize tooltips (Bootstrap)
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize popovers (Bootstrap)
 */
function initializePopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function () {
    initializeTooltips();
    initializePopovers();
});

// Console info
console.log('%c IA Project ', 'background: #28a745; color: white; font-weight: bold; font-size: 14px; padding: 5px 10px;');
console.log('%c Neural Network Optimization with Genetic Algorithms', 'color: #28a745; font-size: 12px;');