// webgpu-logger.js
export async function logWebGPUSupport(containerId = 'webgpu-log') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container with id '${containerId}' not found`);
        return false;
    }

    function log(message, type = 'info') {
        const colors = {
            info: '#888',
            success: '#0f0',
            error: '#f44',
            warning: '#fa0'
        };
        
        container.innerHTML += `<div style="color: ${colors[type]}; font-family: monospace; margin: 2px 0;">
            [${new Date().toLocaleTimeString()}] ${message}
        </div>`;
    }

    // Set container styles
    container.style.cssText = `
        background: #1a1a1a; 
        padding: 10px; 
        border-radius: 4px; 
        max-height: 200px; 
        overflow-y: auto;
        font-size: 12px;
    `;

    log('Checking WebGPU support...', 'info');

    if (!navigator.gpu) {
        log('❌ WebGPU not supported', 'error');
        return false;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            log('❌ No GPU adapter available', 'error');
            return false;
        }

        const device = await adapter.requestDevice();
        log('✅ WebGPU fully supported!', 'success');
        return true;

    } catch (error) {
        log(`❌ WebGPU error: ${error.message}`, 'error');
        return false;
    }
}

// Non-module version for script tags
window.logWebGPUSupport = logWebGPUSupport;
