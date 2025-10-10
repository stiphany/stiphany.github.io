/**
 * Minimal optical profilometry defect detection app
 */

import { default as wasm, Mnist } from "../pkg/browser_models.js";

class ImageProcessor {
    constructor() {
        this.rawIntensityData = null;
        this.rawSurfaceData = null;
        this.width = 0;
        this.height = 0;
        this.lastReconstructed = null;
        this.intensityMin = 0;
        this.intensityMax = 1;
        this.surfaceMin = 0;
        this.surfaceMax = 1;
    }

    async loadDATXFile(datxBuffer) {
        try {
            const h5wasm = await import('https://cdn.jsdelivr.net/npm/h5wasm@0.4.9/dist/esm/hdf5_hl.js');
            const { FS } = await h5wasm.ready;
            
            const fname = "temp.datx";
            FS.writeFile(fname, new Uint8Array(datxBuffer));
            
            const f = new h5wasm.File(fname, "r");
            
            // Try to find the datasets - they might be in different paths
            let surface = null;
            let intensity = null;
            
            try {
                surface = f.get("Measurement/Surface");
                intensity = f.get("Measurement/Intensity");
            } catch (e) {
                console.log('Standard paths not found, exploring file structure...');
                // Print available groups/datasets to help debug
                const keys = f.keys();
                console.log('Available top-level keys:', keys);
                
                // Try alternate paths or just fail gracefully
                f.close();
                FS.unlink(fname);
                throw new Error('Could not find Surface/Intensity datasets in file');
            }
            
            if (!surface || !intensity) {
                f.close();
                FS.unlink(fname);
                throw new Error('Surface or Intensity dataset is null');
            }
            
            this.rawSurfaceData = new Float32Array(surface.value);
            this.rawIntensityData = new Float32Array(intensity.value);
            [this.height, this.width] = surface.shape;
            
            this.calculateNormalization();
            
            f.close();
            FS.unlink(fname);
            return true;
        } catch (error) {
            console.error('Error loading DATX:', error);
            alert('Error loading file: ' + error.message + '\n\nPlease check the file format and ensure it contains Measurement/Surface and Measurement/Intensity datasets.');
            return false;
        }
    }

    calculateNormalization() {
        this.intensityMin = Infinity;
        this.intensityMax = -Infinity;
        this.surfaceMin = Infinity;
        this.surfaceMax = -Infinity;
        
        for (let i = 0; i < this.rawIntensityData.length; i++) {
            const iv = this.rawIntensityData[i];
            const sv = this.rawSurfaceData[i];
            if (isFinite(iv)) {
                if (iv < this.intensityMin) this.intensityMin = iv;
                if (iv > this.intensityMax) this.intensityMax = iv;
            }
            if (isFinite(sv)) {
                if (sv < this.surfaceMin) this.surfaceMin = sv;
                if (sv > this.surfaceMax) this.surfaceMax = sv;
            }
        }
        
        if (this.intensityMax === this.intensityMin) this.intensityMax = this.intensityMin + 1;
        if (this.surfaceMax === this.surfaceMin) this.surfaceMax = this.surfaceMin + 1;
    }

    displayIntensity(canvas) {
        canvas.width = 1024;
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(1024, 1024);
        const range = this.intensityMax - this.intensityMin;
        
        for (let i = 0; i < 1024 * 1024; i++) {
            const val = this.rawIntensityData[i];
            const normalized = isFinite(val) ? Math.max(0, Math.min(1, (val - this.intensityMin) / range)) : 0;
            const gray = Math.floor(normalized * 255);
            const idx = i * 4;
            imageData.data[idx] = imageData.data[idx + 1] = imageData.data[idx + 2] = gray;
            imageData.data[idx + 3] = 255;
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    processCrop(gridX, gridY) {
        const cropSize = 256;
        const result = new Float32Array(cropSize * cropSize);
        const startX = gridX * cropSize;
        const startY = gridY * cropSize;
        
        for (let y = 0; y < cropSize; y++) {
            for (let x = 0; x < cropSize; x++) {
                const val = this.rawIntensityData[(startY + y) * 1024 + startX + x];
                result[y * cropSize + x] = (val - 50481.640625) / 16498.2578125;
            }
        }
        return result;
    }

    combineOutputs(outputs) {
        if (outputs.length !== 16) return false;
        
        this.lastReconstructed = new Float32Array(this.width * this.height);
        
        for (let gridY = 0; gridY < 4; gridY++) {
            for (let gridX = 0; gridX < 4; gridX++) {
                const output = outputs[gridY * 4 + gridX];
                if (output.length !== 65536) return false;
                
                const startX = gridX * 256;
                const startY = gridY * 256;
                
                for (let y = 0; y < 256; y++) {
                    for (let x = 0; x < 256; x++) {
                        if (startX + x < 1024 && startY + y < 1024) {
                            const fullIdx = (startY + y) * 1024 + startX + x;
                            this.lastReconstructed[fullIdx] = (output[y * 256 + x] * 16498.2578125) + 50481.640625;
                        }
                    }
                }
            }
        }
        return true;
    }
}

class App {
    constructor() {
        this.elements = {
            intensityCanvas: document.getElementById('intensity-canvas'),
            runBtn: document.getElementById('run-btn'),
            fileInput: document.getElementById('file-input'),
            uploadArea: document.getElementById('upload-area'),
            demoSelect: document.getElementById('demo-select'),
            progressDiv: document.getElementById('progress'),
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            threejsViewport: document.getElementById('threejs-viewport'),
            intensityViewport: document.getElementById('intensity-viewport'),
            percentileControls: document.getElementById('percentile-controls'),
            percentileSlider: document.getElementById('percentile-slider'),
            percentileValue: document.getElementById('percentile-value'),
            infoBtn: document.getElementById('info-btn'),
            infoModal: document.getElementById('info-modal'),
            closeModal: document.getElementById('close-modal'),
            view3dBtn: document.getElementById('view-3d'),
            viewIntensityBtn: document.getElementById('view-intensity'),
            webgpuBadge: document.getElementById('webgpu-badge')
        };

        this.imageProcessor = new ImageProcessor();
        this.model = null;
        this.isRunning = false;
        this.imageLoaded = false;
        this.currentPercentile = 95;
        this.differenceData = null;
        this.diffMin = null;
        this.diffMax = null;
        this.currentView = '3d';
        
        this.checkWebGPU();
        this.initThreeJS();
        this.bindEvents();
        this.loadDemoFiles();
        this.loadModel();
    }

    async checkWebGPU() {
        if (navigator.gpu) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    this.elements.webgpuBadge.textContent = '✓ WebGPU';
                    this.elements.webgpuBadge.classList.add('supported');
                } else {
                    this.elements.webgpuBadge.textContent = '✗ WebGPU';
                    this.elements.webgpuBadge.classList.add('unsupported');
                }
            } catch (e) {
                this.elements.webgpuBadge.textContent = '✗ WebGPU';
                this.elements.webgpuBadge.classList.add('unsupported');
            }
        } else {
            this.elements.webgpuBadge.textContent = '✗ WebGPU';
            this.elements.webgpuBadge.classList.add('unsupported');
        }
    }

    initThreeJS() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000);
        this.elements.threejsViewport.appendChild(this.renderer.domElement);
        
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        
        this.camera.position.set(50, 50, 50);
        this.camera.lookAt(0, 0, 0);
        
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        this.scene.add(directionalLight);
        
        this.createAxesHelper();
        
        window.addEventListener('resize', () => this.onWindowResize());
        
        this.animate();
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    createAxesHelper() {
        const axesScene = new THREE.Scene();
        const axesCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
        axesCamera.position.set(0, 0, 10);
        
        const axisLength = 3;
        const axes = new THREE.Group();
        
        const xGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0), new THREE.Vector3(axisLength, 0, 0)
        ]);
        axes.add(new THREE.Line(xGeom, new THREE.LineBasicMaterial({ color: 0xff0000 })));
        
        const yGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, axisLength, 0)
        ]);
        axes.add(new THREE.Line(yGeom, new THREE.LineBasicMaterial({ color: 0x00ff00 })));
        
        const zGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, axisLength)
        ]);
        axes.add(new THREE.Line(zGeom, new THREE.LineBasicMaterial({ color: 0x0000ff })));
        
        axesScene.add(axes);
        
        this.axesScene = axesScene;
        this.axesCamera = axesCamera;
        this.axes = axes;
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        
        if (this.axes) {
            this.axes.quaternion.copy(this.camera.quaternion).invert();
        }
        
        this.renderer.autoClear = true;
        this.renderer.render(this.scene, this.camera);
        
        this.renderer.autoClear = false;
        this.renderer.setViewport(10, window.innerHeight - 110, 100, 100);
        this.renderer.render(this.axesScene, this.axesCamera);
        this.renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
    }

    bindEvents() {
        this.elements.uploadArea.addEventListener('click', () => this.elements.fileInput.click());
        this.elements.fileInput.addEventListener('change', e => {
            if (e.target.files[0]) this.loadDatxFile(e.target.files[0]);
        });

        this.elements.uploadArea.addEventListener('dragover', e => e.preventDefault());
        this.elements.uploadArea.addEventListener('drop', e => {
            e.preventDefault();
            if (e.dataTransfer.files[0]) this.loadDatxFile(e.dataTransfer.files[0]);
        });

        this.elements.runBtn.addEventListener('click', () => this.runInference());
        
        this.elements.percentileSlider.addEventListener('input', e => {
            this.currentPercentile = parseInt(e.target.value);
            this.elements.percentileValue.textContent = this.currentPercentile;
            this.updateHighlighting();
        });

        this.elements.infoBtn.addEventListener('click', () => {
            this.elements.infoModal.classList.add('active');
        });
        this.elements.closeModal.addEventListener('click', () => {
            this.elements.infoModal.classList.remove('active');
        });
        this.elements.infoModal.addEventListener('click', e => {
            if (e.target === this.elements.infoModal) {
                this.elements.infoModal.classList.remove('active');
            }
        });

        this.elements.demoSelect.addEventListener('change', e => {
            if (e.target.value) this.loadDemoFile(e.target.value);
        });

        this.elements.view3dBtn.addEventListener('click', () => this.switchView('3d'));
        this.elements.viewIntensityBtn.addEventListener('click', () => this.switchView('intensity'));
    }

    switchView(view) {
        this.currentView = view;
        
        if (view === '3d') {
            this.elements.threejsViewport.classList.add('active');
            this.elements.intensityViewport.classList.remove('active');
            this.elements.view3dBtn.classList.add('active');
            this.elements.viewIntensityBtn.classList.remove('active');
        } else {
            this.elements.threejsViewport.classList.remove('active');
            this.elements.intensityViewport.classList.add('active');
            this.elements.view3dBtn.classList.remove('active');
            this.elements.viewIntensityBtn.classList.add('active');
        }
    }

    async loadDemoFiles() {
        try {
            const files = ['volcano.datx',];
            files.forEach(file => {
                const option = document.createElement('option');
                option.value = `demo_data/${file}`;
                option.textContent = file;
                this.elements.demoSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Demo files not available');
        }
    }

    async loadDemoFile(path) {
        try {
            const response = await fetch(path);
            if (!response.ok) {
                throw new Error(`Could not load demo file: ${response.status} ${response.statusText}`);
            }
            const arrayBuffer = await response.arrayBuffer();
            
            if (arrayBuffer.byteLength === 0) {
                throw new Error('Demo file is empty');
            }
            
            const success = await this.imageProcessor.loadDATXFile(arrayBuffer);
            
            if (success) {
                this.onFileLoaded();
            } else {
                // Reset the select box if loading failed
                this.elements.demoSelect.value = '';
            }
        } catch (error) {
            console.error('Error loading demo:', error);
            alert('Error loading demo file: ' + error.message + '\n\nPlease ensure demo files are in the demo_data/ folder.');
            this.elements.demoSelect.value = '';
        }
    }

    async loadDatxFile(file) {
        if (!file.name.toLowerCase().endsWith('.datx')) return;
        
        try {
            const arrayBuffer = await file.arrayBuffer();
            const success = await this.imageProcessor.loadDATXFile(arrayBuffer);
            
            if (success) {
                this.onFileLoaded();
            }
        } catch (error) {
            console.error('Error loading file:', error);
        }
    }

    onFileLoaded() {
        this.imageLoaded = true;
        this.elements.runBtn.disabled = !this.model;
        this.differenceData = null;
        this.elements.percentileControls.style.display = 'none';
        
        this.imageProcessor.displayIntensity(this.elements.intensityCanvas);
        this.createSurfaceVisualization();
    }

    createSurfaceVisualization() {
        if (this.points) {
            this.scene.remove(this.points);
            this.points = null;
        }

        const surfaceBuffer = this.imageProcessor.rawSurfaceData;
        const surfaceMin = this.imageProcessor.surfaceMin;
        const surfaceMax = this.imageProcessor.surfaceMax;
        const surfaceRange = surfaceMax - surfaceMin;
        
        const positions = [];
        const colors = [];
        
        for (let y = 0; y < 1024; y++) {
            for (let x = 0; x < 1024; x++) {
                const z = surfaceBuffer[y * 1024 + x];
                if (isFinite(z)) {
                    const scaledX = (x - 512) * 0.1;
                    const scaledY = (y - 512) * 0.1;
                    const scaledZ = ((z - surfaceMin) / surfaceRange) * 20;
                    
                    positions.push(scaledX, scaledZ, scaledY);
                    colors.push(0.5, 0.5, 0.5);
                }
            }
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
    }

    updateProgress(current, total) {
        const pct = (current / total) * 100;
        this.elements.progressFill.style.width = `${pct}%`;
        this.elements.progressText.textContent = `Processing ${current}/${total}`;
    }

    async runInference() {
        if (this.isRunning || !this.model || !this.imageLoaded) return;

        this.isRunning = true;
        this.elements.runBtn.disabled = true;
        this.elements.progressDiv.style.display = 'block';

        const outputs = [];

        try {
            for (let gridY = 0; gridY < 4; gridY++) {
                for (let gridX = 0; gridX < 4; gridX++) {
                    const cropIndex = gridY * 4 + gridX + 1;
                    this.updateProgress(cropIndex, 16);

                    const cropData = this.imageProcessor.processCrop(gridX, gridY);
                    const output = await this.model.inference(cropData);
                    outputs.push(Array.isArray(output) ? output : Array.from(output));

                    await new Promise(resolve => setTimeout(resolve, 10));
                }
            }
            
            if (this.imageProcessor.combineOutputs(outputs)) {
                this.createDifferenceData();
                this.elements.percentileControls.style.display = 'block';
                this.updateHighlighting();
            }
            
        } catch (error) {
            console.error('Inference error:', error);
        } finally {
            this.isRunning = false;
            this.elements.runBtn.disabled = false;
            setTimeout(() => {
                this.elements.progressDiv.style.display = 'none';
            }, 1000);
        }
    }

    createDifferenceData() {
        const intensityData = this.imageProcessor.rawIntensityData;
        const reconstructedData = this.imageProcessor.lastReconstructed;
        
        this.differenceData = new Float32Array(intensityData.length);
        this.diffMin = null;
        this.diffMax = null;
        
        for (let i = 0; i < intensityData.length; i++) {
            if (isFinite(intensityData[i]) && isFinite(reconstructedData[i])) {
                this.differenceData[i] = Math.abs(intensityData[i] - reconstructedData[i]);
            } else {
                this.differenceData[i] = 0;
            }
        }
    }

    updateHighlighting() {
        if (!this.differenceData || !this.points) return;

        const validDiffs = this.differenceData.filter(d => isFinite(d) && d > 0);
        if (validDiffs.length === 0) return;
        
        validDiffs.sort((a, b) => a - b);
        const thresholdIdx = Math.floor((this.currentPercentile / 100) * validDiffs.length);
        const threshold = validDiffs[Math.min(thresholdIdx, validDiffs.length - 1)];

        this.updatePointCloud(threshold);
        this.updateIntensityOverlay(threshold);
    }

    updatePointCloud(threshold) {
        if (!this.diffMin || !this.diffMax) {
            this.diffMin = Infinity;
            this.diffMax = -Infinity;
            for (let i = 0; i < this.differenceData.length; i++) {
                if (isFinite(this.differenceData[i])) {
                    if (this.differenceData[i] < this.diffMin) this.diffMin = this.differenceData[i];
                    if (this.differenceData[i] > this.diffMax) this.diffMax = this.differenceData[i];
                }
            }
        }

        const colors = [];
        const surfaceBuffer = this.imageProcessor.rawSurfaceData;

        for (let y = 0; y < 1024; y++) {
            for (let x = 0; x < 1024; x++) {
                const idx = y * 1024 + x;
                const z = surfaceBuffer[idx];
                if (isFinite(z)) {
                    if (this.differenceData[idx] >= threshold) {
                        colors.push(1, 0, 0);
                    } else {
                        colors.push(0.5, 0.5, 0.5);
                    }
                }
            }
        }

        if (this.points && this.points.geometry && this.points.geometry.attributes.color) {
            const colorAttr = this.points.geometry.attributes.color;
            const colorArray = new Float32Array(colors);
            if (colorArray.length === colorAttr.array.length) {
                colorAttr.array.set(colorArray);
                colorAttr.needsUpdate = true;
            }
        }
    }

    updateIntensityOverlay(threshold) {
        this.imageProcessor.displayIntensity(this.elements.intensityCanvas);
        
        const canvas = this.elements.intensityCanvas;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, 1024, 1024);
        const data = imageData.data;
        
        for (let i = 0; i < this.differenceData.length; i++) {
            if (isFinite(this.differenceData[i]) && this.differenceData[i] >= threshold) {
                const idx = i * 4;
                data[idx] = Math.min(255, data[idx] + 100);
                data[idx + 1] = Math.max(0, data[idx + 1] - 50);
                data[idx + 2] = Math.max(0, data[idx + 2] - 50);
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }

    async loadModel() {
        try {
            await wasm();
            this.model = new Mnist();
            this.elements.runBtn.disabled = !this.imageLoaded;
        } catch (error) {
            console.error('Model loading error:', error);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    try {
        new App();
    } catch (error) {
        console.error('Failed to initialize:', error);
    }
});
