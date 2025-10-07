/**
 * Image processing utilities for the model inference demo
 * Handles DATX files with proper HDF5 parsing using h5wasm
 */

class ImageProcessor {
    constructor(intensityCanvas, surfaceCanvas, outputCanvas, cropCanvas) {
        this.intensityCanvas = intensityCanvas;
        this.surfaceCanvas = surfaceCanvas;
        this.outputCanvas = outputCanvas;
        this.cropCanvas = cropCanvas;
        
        // Store the raw data from DATX file
        this.rawIntensityData = null;
        this.rawSurfaceData = null;
        this.width = 0;
        this.height = 0;
        this.lastReconstructed = null;
        
        // Store normalization parameters for consistency
        this.intensityMin = 0;
        this.intensityMax = 1;
        this.surfaceMin = 0;
        this.surfaceMax = 1;
    }

    /**
     * Load DATX file using h5wasm and extract surface/intensity data
     * @param {ArrayBuffer} datxBuffer - DATX file buffer
     * @returns {Promise<boolean>} Success status
     */
    async loadDATXFile(datxBuffer) {
        try {
            // Import h5wasm dynamically
            const h5wasm = await import('https://cdn.jsdelivr.net/npm/h5wasm@0.4.9/dist/esm/hdf5_hl.js');
            const { FS } = await h5wasm.ready;
            
            // Write to virtual filesystem
            const fname = "temp.datx";
            FS.writeFile(fname, new Uint8Array(datxBuffer));
            
            // Open HDF5 file and read datasets
            const f = new h5wasm.File(fname, "r");
            const surface = f.get("Measurement/Surface");
            const intensity = f.get("Measurement/Intensity");
            
            // Get the data buffers and dimensions
            this.rawSurfaceData = new Float32Array(surface.value);
            this.rawIntensityData = new Float32Array(intensity.value);
            [this.height, this.width] = surface.shape;
            
            console.log(`Loaded DATX: ${this.width}x${this.height}`);
            
            // Calculate normalization parameters
            this.calculateNormalizationParams();
            
            // Cleanup
            f.close();
            FS.unlink(fname);
            
            // Display the data on canvases
            this.displayDataOnCanvas(this.rawSurfaceData, this.surfaceCanvas, 'surface');
            this.displayDataOnCanvas(this.rawIntensityData, this.intensityCanvas, 'intensity');
            
            return true;
        } catch (error) {
            console.error('Error loading DATX file:', error);
            return false;
        }
    }

    /**
     * Calculate and store normalization parameters for consistent processing
     */
    calculateNormalizationParams() {
        // Find min/max for intensity data
        this.intensityMin = Infinity;
        this.intensityMax = -Infinity;
        
        for (let i = 0; i < this.rawIntensityData.length; i++) {
            if (!isNaN(this.rawIntensityData[i]) && isFinite(this.rawIntensityData[i])) {
                if (this.rawIntensityData[i] < this.intensityMin) this.intensityMin = this.rawIntensityData[i];
                if (this.rawIntensityData[i] > this.intensityMax) this.intensityMax = this.rawIntensityData[i];
            }
        }
        
        // Find min/max for surface data
        this.surfaceMin = Infinity;
        this.surfaceMax = -Infinity;
        
        for (let i = 0; i < this.rawSurfaceData.length; i++) {
            if (!isNaN(this.rawSurfaceData[i]) && isFinite(this.rawSurfaceData[i])) {
                if (this.rawSurfaceData[i] < this.surfaceMin) this.surfaceMin = this.rawSurfaceData[i];
                if (this.rawSurfaceData[i] > this.surfaceMax) this.surfaceMax = this.rawSurfaceData[i];
            }
        }
        
        console.log(`Intensity range: ${this.intensityMin} to ${this.intensityMax}`);
        console.log(`Surface range: ${this.surfaceMin} to ${this.surfaceMax}`);
        
        // Avoid division by zero
        if (this.intensityMax === this.intensityMin) {
            this.intensityMax = this.intensityMin + 1;
        }
        if (this.surfaceMax === this.surfaceMin) {
            this.surfaceMax = this.surfaceMin + 1;
        }
    }

    /**
     * Display data on canvas with proper scaling
     */
    displayDataOnCanvas(data, canvas, type) {
        canvas.width = 1024;
        canvas.height = 1024;
        
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(1024, 1024);
        
        // Use pre-calculated min/max for consistent normalization
        const min = type === 'surface' ? this.surfaceMin : this.intensityMin;
        const max = type === 'surface' ? this.surfaceMax : this.intensityMax;
        const range = max - min;
        
        // Scale and display - no interpolation, direct pixel mapping
        for (let y = 0; y < 1024; y++) {
            for (let x = 0; x < 1024; x++) {
                let normalized = 0;
                let val = data[y * 1024 + x];
                if (!isNaN(val) && isFinite(val)) {
                    normalized = Math.max(0, Math.min(1, (val - min) / range));
                }
                
                const gray = Math.floor(normalized * 255);
                const idx = (y * 1024 + x) * 4;
                imageData.data[idx] = gray;
                imageData.data[idx + 1] = gray;
                imageData.data[idx + 2] = gray;
                imageData.data[idx + 3] = 255;
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    /**
     * Extract a crop from the intensity data at specified grid position
     * @param {number} gridX - Grid X position (0-3) 
     * @param {number} gridY - Grid Y position (0-3)
     * @returns {Float32Array} Processed crop data (normalized to 0-1)
     */
    processCrop(gridX, gridY) {
        if (!this.rawIntensityData) {
            throw new Error('No intensity data loaded');
        }

        const cropSize = 256;
        const result = new Float32Array(cropSize * cropSize);
        
        // Simple direct crop extraction - no overlapping
        const startX = gridX * cropSize;
        const startY = gridY * cropSize;
    	
        // Extract crop with direct pixel mapping and normalization
        for (let y = 0; y < cropSize; y++) {
            for (let x = 0; x < cropSize; x++) {
                const origX = startX + x;
                const origY = startY + y;
                const origIndex = origY * 1024 + origX;
                
                let val = this.rawIntensityData[origIndex];
                
                // Normalize to 0-1 range using global parameters
                let normalized = 0;
                if (!isNaN(val) && isFinite(val)) {
                    // normalized = Math.max(0, Math.min(1, (val - this.intensityMin) / (this.intensityMax - this.intensityMin)));
                    normalized = val; //(val - 50481.640625) / 16498.2578125;
		}
                
                result[y * cropSize + x] = (normalized - 50481.640625) / 16498.2578125;
            }
        }

        return result;
    }

    /**
     * Combine 16 model outputs into final result display
     * @param {Array<Array>} outputs - Array of 16 model outputs (each should be normalized 0-1)
     * @param {HTMLElement} infoDiv - Info display element
     * @returns {boolean} Success status
     */
    combineOutputs(outputs, infoDiv) {
        if (outputs.length !== 16) {
            infoDiv.textContent = `Error: Expected 16 outputs, got ${outputs.length}`;
            return false;
        }

        // Initialize reconstruction buffer
        this.lastReconstructed = new Float32Array(this.width * this.height);
        this.lastReconstructed.fill(0);
        const range = this.intensityMax - this.intensityMin;
        // Simple grid reconstruction - no overlapping or interpolation
        for (let gridY = 0; gridY < 4; gridY++) {
            for (let gridX = 0; gridX < 4; gridX++) {
                const cropIndex = gridY * 4 + gridX;
                const output = outputs[cropIndex];
                
                if (output.length !== 65536) {
                    infoDiv.textContent = `Error: Expected 65536 values per output, got ${output.length}`;
                    return false;
                }
                
                // Direct pixel placement - no averaging
                const startX = gridX * 256;
                const startY = gridY * 256;
                
                for (let cropY = 0; cropY < 256; cropY++) {
                    for (let cropX = 0; cropX < 256; cropX++) {
                        const fullX = startX + cropX;
                        const fullY = startY + cropY;
                        
                        // Make sure we don't go out of bounds
                        if (fullX < 1024 && fullY < 1024) {
                            const fullIndex = fullY * 1024 + fullX;
                            const cropOutputIndex = cropY * 256 + cropX;
                            
                            // Denormalize back to original intensity range
		            const cc = ((output[cropOutputIndex] * 16498.2578125) + 50481.640625);
			    const ccc = (cc - this.intensityMin) / range;
			    this.lastReconstructed[fullIndex] = cc; ///Math.floor(ccc * 255); 
                        }
                    }
                }
            }
        }

        // Display the reconstructed result
        this.displayDataOnCanvas(this.lastReconstructed, this.outputCanvas, 'intensity');
        
        // Calculate and display statistics
        let min = this.lastReconstructed[0], max = this.lastReconstructed[0];
        
        for (let i = 0; i < this.lastReconstructed.length; i++) {
            if (this.lastReconstructed[i] < min) min = this.lastReconstructed[i];
            if (this.lastReconstructed[i] > max) max = this.lastReconstructed[i];
        }
        
        infoDiv.innerHTML = `Processed 16 crops<br>Range: ${min.toFixed(4)} to ${max.toFixed(4)}`;
        return true;
    }

    /**
     * Get dimensions of loaded data
     * @returns {Object} Width and height
     */
    getDimensions() {
        return { width: this.width, height: this.height };
    }

    /**
     * Check if data is loaded
     * @returns {boolean} True if both surface and intensity data are loaded
     */
    isDataLoaded() {
        return this.rawSurfaceData !== null && this.rawIntensityData !== null;
    }
}

/**
 * Main application logic for the Model Inference Demo
 * Handles DATX file loading, 3D visualization, and 4x4 grid inference processing
 */

// Import real WASM module
import { default as wasm, Mnist } from "../pkg/browser_models.js";

class ModelInferenceApp {
    constructor() {
        this.initializeElements();
        
        if (/iPhone|iPad|iPod/i.test(navigator.userAgent)) {
            const startButton = document.createElement('button');
            startButton.textContent = 'Tap to Enable WebGPU';
            startButton.style.cssText = 'background: #007AFF; color: white; border: none; padding: 12px 20px; border-radius: 6px; font-size: 16px; margin: 10px; width: calc(100% - 20px);';
            startButton.onclick = async () => {
                try {
                    // This triggers WebGPU initialization with user gesture
                    const adapter = await navigator.gpu.requestAdapter();
                    const device = await adapter.requestDevice();
                    device.destroy(); // Just testing
                    startButton.remove();
                    console.log('WebGPU ready on mobile!');
                } catch (e) {
                    console.error('WebGPU failed:', e);
                    startButton.textContent = 'WebGPU Failed - Try Again';
                }
            };
            this.elements.uploadArea.parentNode.insertBefore(startButton, this.elements.uploadArea);
        }

	this.initializeImageProcessor();
        this.initializeThreeJS();
        this.bindEvents();
        this.loadModel();
        
        this.model = null;
        this.isRunning = false;
        this.imageLoaded = false;
        this.currentPercentile = 95;
        this.differenceData = null;
        this.diffMin = null;
        this.diffMax = null;
    }

    initializeElements() {
        this.elements = {
            surfaceCanvas: document.getElementById('surface-canvas'),
            intensityCanvas: document.getElementById('intensity-canvas'),
            outputCanvas: document.getElementById('output-canvas'),
            differenceCanvas: document.getElementById('difference-canvas'),
            cropCanvas: document.getElementById('crop-canvas'),
            statusDiv: document.getElementById('status'),
            infoDiv: document.getElementById('info'),
            diffInfo: document.getElementById('diff-info'),
            runBtn: document.getElementById('run-btn'),
            fileInput: document.getElementById('file-input'),
            uploadArea: document.getElementById('upload-area'),
            progressDiv: document.getElementById('progress'),
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            threejsContainer: document.getElementById('threejs-viewport'),
            diffContainer: document.getElementById('diff-container'),
            percentileControls: document.getElementById('percentile-controls'),
            percentileSlider: document.getElementById('percentile-slider'),
            percentileValue: document.getElementById('percentile-value')
        };
    }

    initializeImageProcessor() {
        this.imageProcessor = new ImageProcessor(
            this.elements.intensityCanvas,
            this.elements.surfaceCanvas,
            this.elements.outputCanvas,
            this.elements.cropCanvas
        );
    }

    initializeThreeJS() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, 1.0, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(1024, 1024);
        this.renderer.setClearColor(0x111111);
        this.elements.threejsContainer.appendChild(this.renderer.domElement);
        
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        
        this.camera.position.set(50, 50, 50);
        this.camera.lookAt(0, 0, 0);
        
        // Add some basic lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        this.scene.add(directionalLight);
        
        this.animate();
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    bindEvents() {
        // File upload events
        this.elements.uploadArea.addEventListener('click', () => {
            this.elements.fileInput.click();
        });

        this.elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.loadDatxFile(e.target.files[0]);
            }
        });

        // Drag and drop events
        this.elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.add('dragover');
        });

        this.elements.uploadArea.addEventListener('dragleave', () => {
            this.elements.uploadArea.classList.remove('dragover');
        });

        this.elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                this.loadDatxFile(e.dataTransfer.files[0]);
            }
        });

        // Run inference button
        this.elements.runBtn.addEventListener('click', () => this.runGridInference());

        // Percentile slider
        this.elements.percentileSlider.addEventListener('input', (e) => {
            this.currentPercentile = parseInt(e.target.value);
            this.elements.percentileValue.textContent = this.currentPercentile;
            this.updatePercentileHighlighting();
        });
    }

    /**
     * Update percentile highlighting in both point cloud and intensity image
     */
    updatePercentileHighlighting() {
        if (!this.differenceData || !this.points) return;

        // Calculate percentile threshold
        const validDifferences = [];
        for (let i = 0; i < this.differenceData.length; i++) {
            if (!isNaN(this.differenceData[i]) && isFinite(this.differenceData[i])) {
                validDifferences.push(this.differenceData[i]);
            }
        }
        
        if (validDifferences.length === 0) return;
        
        validDifferences.sort((a, b) => a - b);
        const thresholdIndex = Math.floor((this.currentPercentile / 100) * validDifferences.length);
        const threshold = validDifferences[Math.min(thresholdIndex, validDifferences.length - 1)];

        console.log(`Percentile ${this.currentPercentile}% threshold: ${threshold}`);

        // Update point cloud colors
        this.updatePointCloudHighlighting(threshold);
        
        // Update intensity image overlay
        this.updateIntensityHighlighting(threshold);
    }

    /**
     * Update point cloud with highlighting based on difference data and percentile
     */
    updatePointCloudHighlighting(threshold) {
        if (!this.points || !this.differenceData) return;

        // Calculate min/max once for the entire difference dataset
        if (!this.diffMin || !this.diffMax) {
            this.diffMin = Infinity;
            this.diffMax = -Infinity;
            for (let i = 0; i < this.differenceData.length; i++) {
                if (!isNaN(this.differenceData[i]) && isFinite(this.differenceData[i])) {
                    if (this.differenceData[i] < this.diffMin) this.diffMin = this.differenceData[i];
                    if (this.differenceData[i] > this.diffMax) this.diffMax = this.differenceData[i];
                }
            }
        }

        const { width, height } = this.imageProcessor.getDimensions();
        const colors = [];
        const stride = 1; // Match the stride used in createSurfaceVisualization
        const diffRange = this.diffMax - this.diffMin || 1;
        const surfaceBuffer = this.imageProcessor.rawSurfaceData;

        // Color the points based on difference values
        for (let y = 0; y < height; y += stride) {
            for (let x = 0; x < width; x += stride) {
                const diffIndex = y * width + x;
                const diff = this.differenceData[diffIndex];
                const z = surfaceBuffer[diffIndex];
                if (!isNaN(z) && isFinite(z)) {
                    if (diff >= threshold) {
                        // Red for high percentile (above threshold)
                        colors.push(1, 0, 0);
                    } else {
                        // Blue to white gradient for below threshold values
                        // const normalizedDiff = Math.max(0, Math.min(1, (diff - this.diffMin) / diffRange));
                        //const intensity = normalizedDiff * 0.8 + 0.2; // 0.2 to 1.0 range
                        // colors.push(intensity * 0.3, intensity * 0.3, intensity);
                        colors.push(0.5, 0.5, 0.5);
                    }
            	}
	     }
          }
    
	        // Actually update the point cloud colors - this was missing!
    if (this.points && this.points.geometry && this.points.geometry.attributes.color && colors.length > 0) {
        const colorAttribute = this.points.geometry.attributes.color;
        const colorArray = new Float32Array(colors);

        // Make sure the array sizes match
        if (colorArray.length === colorAttribute.array.length) {
            colorAttribute.array.set(colorArray);
            colorAttribute.needsUpdate = true;
        } else {
            console.warn(`Color array length mismatch: ${colorArray.length} vs ${colorAttribute.array.length}`);
        }
    }
    }

    /**
     * Update intensity image with red overlay for high percentile values
     */
    updateIntensityHighlighting(threshold) {
        if (!this.imageProcessor.rawIntensityData) return;

        // Redraw original intensity data first
        this.imageProcessor.displayDataOnCanvas(this.imageProcessor.rawIntensityData, this.elements.intensityCanvas, 'intensity');
        
        // Add red overlay for high percentile areas
        const canvas = this.elements.intensityCanvas;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, 1024, 1024);
        const data = imageData.data;
        
        // Apply red overlay where difference exceeds threshold - direct pixel mapping
        for (let y = 0; y < 1024; y++) {
            for (let x = 0; x < 1024; x++) {
                const origIndex = y * 1024 + x;
                
                if (origIndex < this.differenceData.length && 
                    !isNaN(this.differenceData[origIndex]) && 
                    isFinite(this.differenceData[origIndex]) &&
                    this.differenceData[origIndex] >= threshold) {
                    
                    const idx = (y * 1024 + x) * 4;
                    // Mix with red - increase red channel, reduce others
                    data[idx] = Math.min(255, data[idx] + 100);     // More red
                    data[idx + 1] = Math.max(0, data[idx + 1] - 50); // Less green
                    data[idx + 2] = Math.max(0, data[idx + 2] - 50); // Less blue
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    /**
     * Create 3D point cloud from surface data
     */
    createSurfaceVisualization() {
        // Remove existing visualization
        if (this.points) {
            this.scene.remove(this.points);
            this.points = null;
        }

        const { width, height } = this.imageProcessor.getDimensions();
        const surfaceBuffer = this.imageProcessor.rawSurfaceData;
        
        if (!surfaceBuffer) {
            console.error('No surface data available for visualization');
            return;
        }

        // Use the pre-calculated surface range for consistent normalization
        const surfaceMin = this.imageProcessor.surfaceMin;
        const surfaceMax = this.imageProcessor.surfaceMax;
        const surfaceRange = surfaceMax - surfaceMin;
        
        if (surfaceRange <= 0) {
            console.error('Invalid surface data range');
            return;
        }
        
        const positions = [];
        const colors = [];
        const stride = 1; // Every 2nd pixel for performance
        
        // Create point cloud with proper scaling
        for (let y = 0; y < height; y += stride) {
            for (let x = 0; x < width; x += stride) {
                const i = y * width + x;
                const z = surfaceBuffer[i];
               	 
                if (!isNaN(z) && isFinite(z)) {
                    // Position: scale X,Y to reasonable range, scale Z height
                    const scaledX = (x - width/2) * 0.1;
                    const scaledY = (y - height/2) * 0.1;
                    const scaledZ = ((z - surfaceMin) / surfaceRange) * 20; // Scale height to 0-20 units
                    
                    positions.push(scaledX, scaledZ, scaledY); // Note: Y and Z swapped for better view
                    
                    // Start with gray colors - will be updated when difference data is available
                    colors.push(0.5, 0.5, 0.5);
                }
            }
        }
        
        if (positions.length === 0) {
            console.error('No valid positions generated for point cloud');
            return;
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({ 
            size: 1, 
            vertexColors: true,
            sizeAttenuation: false 
        });
        
        this.points = new THREE.Points(geometry, material);
        this.scene.add(this.points);
        
        console.log(`3D visualization: ${positions.length/3} points`);
    }

    /**
     * Load and process DATX file
     * @param {File} file - DATX file
     */
    async loadDatxFile(file) {
        if (!file.name.toLowerCase().endsWith('.datx')) {
            this.showStatus('Please select a DATX file', 'error');
            return;
        }

        this.showStatus('Loading DATX file...', 'processing');
        
        try {
            const arrayBuffer = await file.arrayBuffer();
            const success = await this.imageProcessor.loadDATXFile(arrayBuffer);
            
            if (success) {
                this.imageLoaded = true;
                this.elements.runBtn.disabled = !this.model;
                this.showStatus('DATX file loaded successfully!', 'success');
                this.elements.infoDiv.innerHTML = 'Ready for inference';
                
                // Reset difference visualization
                this.differenceData = null;
                this.elements.diffContainer.style.display = 'none';
                this.elements.percentileControls.style.display = 'none';
                
                // Create 3D visualization
                this.createSurfaceVisualization();
                
                const dims = this.imageProcessor.getDimensions();
                console.log(`Loaded: ${dims.width}x${dims.height}`);
            } else {
                this.showStatus('Failed to process DATX file', 'error');
            }
        } catch (error) {
            console.error('Error loading DATX file:', error);
            this.showStatus(`Error loading file: ${error.message}`, 'error');
        }
    }

    /**
     * Display status message with appropriate styling
     * @param {string} message - Status message
     * @param {string} type - Status type: 'success', 'error', 'processing'
     */
    showStatus(message, type = 'processing') {
        this.elements.statusDiv.textContent = message;
        this.elements.statusDiv.className = `status ${type}`;
        this.elements.statusDiv.style.display = 'block';
        
        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                if (this.elements.statusDiv.classList.contains('success')) {
                    this.elements.statusDiv.style.display = 'none';
                }
            }, 5000);
        }
    }

    /**
     * Update progress bar and text
     * @param {number} current - Current crop number (0-based)
     * @param {number} total - Total crops
     */
    updateProgress(current, total) {
        const percentage = (current / total) * 100;
        this.elements.progressFill.style.width = `${percentage}%`;
        this.elements.progressText.textContent = `Processing crop ${current + 1}/${total}`;
    }

    /**
     * Run inference on 4x4 grid of crops
     */
    async runGridInference() {
        if (this.isRunning || !this.model || !this.imageLoaded) return;

        this.isRunning = true;
        this.elements.runBtn.disabled = true;
        this.elements.progressDiv.style.display = 'block';
        this.showStatus('Running inference on 16 crops...', 'processing');

        const outputs = [];
        const totalCrops = 16;
        const startTime = performance.now();

        try {
            // Process each crop in the 4x4 grid
            for (let gridY = 0; gridY < 4; gridY++) {
                for (let gridX = 0; gridX < 4; gridX++) {
                    const cropIndex = gridY * 4 + gridX;
                    this.updateProgress(cropIndex, totalCrops);

                    // Extract and process crop
                    const cropData = this.imageProcessor.processCrop(gridX, gridY);

                    // Run model inference
                    const cropStartTime = performance.now();
                    const output = await this.model.inference(cropData);
                    const cropLatency = performance.now() - cropStartTime;
			
                    console.log(`Crop ${cropIndex + 1}/16 (${gridX},${gridY}) processed in ${cropLatency.toFixed(1)}ms`);

                    // Convert output to array if needed
                    const outputArray = Array.isArray(output) ? output : Array.from(output);
                    outputs.push(outputArray);

                    // Small delay to allow UI updates
                    await new Promise(resolve => setTimeout(resolve, 10));
                }
            }
            
            this.updateProgress(totalCrops, totalCrops);
            
            // Combine results and create visualizations
            if (this.imageProcessor.combineOutputs(outputs, this.elements.infoDiv)) {
                this.createDifferenceImage();
                this.elements.diffContainer.style.display = 'block';
                this.elements.percentileControls.style.display = 'block';
                
                // Update point cloud with initial highlighting
                this.updatePercentileHighlighting();
                
                const totalLatency = (performance.now() - startTime).toFixed(1);
                this.showStatus(`Success! Total processing time: ${totalLatency}ms`, 'success');
            } else {
                this.showStatus('Error processing outputs', 'error');
            }
            
        } catch (error) {
            console.error('Inference error:', error);
            this.showStatus(`Error: ${error.message}`, 'error');
        } finally {
            this.isRunning = false;
            this.elements.runBtn.disabled = false;
            setTimeout(() => {
                this.elements.progressDiv.style.display = 'none';
            }, 1000);
        }
    }

    /**
     * Create absolute difference image between input and reconstructed output
     */
    createDifferenceImage() {
        const intensityData = this.imageProcessor.rawIntensityData;
        const reconstructedData = this.imageProcessor.lastReconstructed;
        
        if (!intensityData || !reconstructedData) {
            console.error('Missing data for difference calculation');
            this.elements.diffInfo.textContent = 'Error: Missing data for difference calculation';
            return;
        }

        if (intensityData.length !== reconstructedData.length) {
            console.error(`Data length mismatch: intensity=${intensityData.length}, reconstructed=${reconstructedData.length}`);
            this.elements.diffInfo.textContent = 'Error: Data size mismatch';
            return;
        }

        this.differenceData = new Float32Array(intensityData.length);
        // Reset cached min/max for new difference data
        this.diffMin = null;
        this.diffMax = null;
        let validDifferences = 0;
        
        // Calculate absolute difference
        for (let i = 0; i < intensityData.length; i++) {
            if (!isNaN(intensityData[i]) && isFinite(intensityData[i]) &&
                !isNaN(reconstructedData[i]) && isFinite(reconstructedData[i])) {
                this.differenceData[i] = Math.abs(intensityData[i] - reconstructedData[i]);
                validDifferences++;
            } else {
                this.differenceData[i] = 0;
            }
        }

        if (validDifferences === 0) {
            console.error('No valid differences calculated');
            this.elements.diffInfo.textContent = 'Error: No valid differences found';
            return;
        }

        // Display difference image
        this.displayDifferenceImage();
        
        console.log(`Calculated ${validDifferences} valid differences out of ${intensityData.length} pixels`);
    }

    /**
     * Display difference image with red coloring
     */
    displayDifferenceImage() {
        if (!this.differenceData) return;

        const canvas = this.elements.differenceCanvas;
        canvas.width = 1024;
        canvas.height = 1024;
        
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(1024, 1024);
        
        // Find min/max for normalization
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < this.differenceData.length; i++) {
            if (!isNaN(this.differenceData[i]) && isFinite(this.differenceData[i])) {
                if (this.differenceData[i] < min) min = this.differenceData[i];
                if (this.differenceData[i] > max) max = this.differenceData[i];
            }
        }
        
        if (min === Infinity || max === -Infinity) {
            console.error('No valid difference values found');
            this.elements.diffInfo.textContent = 'Error: No valid difference data';
            return;
        }
        
        const range = max - min || 1;
        
        // Display as red intensity with direct pixel mapping
        for (let y = 0; y < 1024; y++) {
            for (let x = 0; x < 1024; x++) {
                const origIndex = y * 1024 + x;
                
                let normalized = 0;
                if (origIndex < this.differenceData.length) {
                    const val = this.differenceData[origIndex];
                    if (!isNaN(val) && isFinite(val)) {
                        normalized = (val - min) / range;
                    }
                }
                
                const intensity = Math.floor(normalized * 255);
                const idx = (y * 1024 + x) * 4;
                imageData.data[idx] = intensity;     // Red channel
                imageData.data[idx + 1] = 0;         // Green channel  
                imageData.data[idx + 2] = 0;         // Blue channel
                imageData.data[idx + 3] = 255;       // Alpha
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        this.elements.diffInfo.innerHTML = `Range: ${min.toFixed(6)} to ${max.toFixed(6)}`;
    }

    /**
     * Load the WASM model
     */
    async loadModel() {
        try {
            this.showStatus('Loading model...', 'processing');
            await wasm();
            this.model = new Mnist();
            this.elements.runBtn.disabled = !this.imageLoaded;
            this.showStatus('Model loaded! Upload a DATX file to begin', 'success');
            console.log('Model loaded successfully');
        } catch (error) {
            this.showStatus(`Failed to load model: ${error.message}`, 'error');
            console.error('Model loading error:', error);
        }
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        new ModelInferenceApp();
        console.log('Model Inference App initialized successfully');
    } catch (error) {
        console.error('Failed to initialize app:', error);
    }
});
