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
        this.allDatasets = {}; // Store all loaded datasets
        this.datasetInfo = {}; // Store min/max for each dataset
    }

    async loadDATXFile(datxBuffer) {
        try {
            const h5wasm = await import('https://cdn.jsdelivr.net/npm/h5wasm@0.4.9/dist/esm/hdf5_hl.js');
            const { FS } = await h5wasm.ready;
            
            const fname = "temp.datx";
            FS.writeFile(fname, new Uint8Array(datxBuffer));
            
            const f = new h5wasm.File(fname, "r");
            
            // Explore the entire file structure
            console.log('Exploring DATX file structure...');
            this.allDatasets = {};
            this.datasetInfo = {};
            
            // Function to recursively explore HDF5 structure
            const exploreGroup = (path = '') => {
                const keys = path ? f.get(path).keys() : f.keys();
                console.log(`Keys at ${path || 'root'}:`, keys);
                
                for (const key of keys) {
                    const fullPath = path ? `${path}/${key}` : key;
                    try {
                        const item = f.get(fullPath);
                        
                        // Check if it's a group or dataset
                        if (item.type === 'Group') {
                            console.log(`Found group: ${fullPath}`);
                            exploreGroup(fullPath);
                        } else if (item.type === 'Dataset') {
                            console.log(`Found dataset: ${fullPath}`, item.shape, item.dtype);
                            
                            // Only load 2D datasets that look like images
                            if (item.shape && item.shape.length === 2) {
                                const data = new Float32Array(item.value);
                                const [height, width] = item.shape;
                                
                                // Calculate min/max
                                let min = Infinity;
                                let max = -Infinity;
                                for (let i = 0; i < data.length; i++) {
                                    if (isFinite(data[i])) {
                                        if (data[i] < min) min = data[i];
                                        if (data[i] > max) max = data[i];
                                    }
                                }
                                
                                this.allDatasets[fullPath] = {
                                    data: data,
                                    width: width,
                                    height: height,
                                    name: key,
                                    fullPath: fullPath
                                };
                                
                                this.datasetInfo[fullPath] = { min, max };
                                console.log(`Loaded ${fullPath}: ${width}x${height}, range [${min.toFixed(2)}, ${max.toFixed(2)}]`);
                            }
                        }
                    } catch (e) {
                        console.log(`Could not access ${fullPath}:`, e.message);
                    }
                }
            };
            
            exploreGroup();
            
            // Try to find Surface and Intensity for backwards compatibility
            let surface = null;
            let intensity = null;
            
            // Look for Surface and Intensity in any of the loaded datasets
            for (const [path, dataset] of Object.entries(this.allDatasets)) {
                if (path.toLowerCase().includes('surface')) {
                    surface = dataset;
                }
                if (path.toLowerCase().includes('intensity')) {
                    intensity = dataset;
                }
            }
            
            if (!surface || !intensity) {
                console.warn('Could not find Surface/Intensity datasets by name');
                // Use first two datasets as fallback
                const datasets = Object.values(this.allDatasets);
                if (datasets.length >= 2) {
                    surface = datasets[0];
                    intensity = datasets[1];
                    console.log('Using first two datasets as Surface and Intensity');
                } else {
                f.close();
                FS.unlink(fname);
                    throw new Error('Could not find enough 2D datasets in file');
                }
            }
            
            this.rawSurfaceData = surface.data;
            this.rawIntensityData = intensity.data;
            this.width = surface.width;
            this.height = surface.height;
            
            this.calculateNormalization();
            
            f.close();
            FS.unlink(fname);
            
            console.log(`Loaded ${Object.keys(this.allDatasets).length} datasets from DATX file`);
            return true;
        } catch (error) {
            console.error('Error loading DATX:', error);
            alert('Error loading file: ' + error.message + '\n\nPlease check the file format.');
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

    displayDataset(canvas, datasetPath) {
        const dataset = this.allDatasets[datasetPath];
        if (!dataset) return;
        
        const { data, width, height } = dataset;
        const { min, max } = this.datasetInfo[datasetPath];
        const range = max - min;
        
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(width, height);
        
        for (let i = 0; i < width * height; i++) {
            const val = data[i];
            const normalized = isFinite(val) ? Math.max(0, Math.min(1, (val - min) / range)) : 0;
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
            datasetGridViewport: document.getElementById('dataset-grid-viewport'),
            percentileControls: document.getElementById('percentile-controls'),
            percentileSlider: document.getElementById('percentile-slider'),
            percentileValue: document.getElementById('percentile-value'),
            infoBtn: document.getElementById('info-btn'),
            infoModal: document.getElementById('info-modal'),
            closeModal: document.getElementById('close-modal'),
            view3dBtn: document.getElementById('view-3d'),
            viewIntensityBtn: document.getElementById('view-intensity'),
            viewDatasetsBtn: document.getElementById('view-datasets'),
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
        
        // Height scaling controls
        this.heightScaleMode = 'auto'; // 'auto' or 'manual'
        this.manualHeightMin = 0;
        this.manualHeightMax = 100;
        this.heightMultiplier = 20;
        
        // Multi-file management
        this.loadedFiles = []; // Array of {name, processor, thumbnail}
        this.currentFileIndex = -1;
        
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
        this.elements.fileInput.addEventListener('change', async e => {
            if (e.target.files.length > 0) {
                await this.loadMultipleDatxFiles(Array.from(e.target.files));
            }
        });

        this.elements.uploadArea.addEventListener('dragover', e => e.preventDefault());
        this.elements.uploadArea.addEventListener('drop', async e => {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) {
                await this.loadMultipleDatxFiles(Array.from(e.dataTransfer.files));
            }
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
        if (this.elements.viewDatasetsBtn) {
            this.elements.viewDatasetsBtn.addEventListener('click', () => this.switchView('datasets'));
        }

        // Keyboard navigation for multiple files
        document.addEventListener('keydown', e => {
            if (this.loadedFiles.length <= 1) return;
            
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
                e.preventDefault();
                if (this.currentFileIndex > 0) {
                    this.switchToFile(this.currentFileIndex - 1);
                }
            } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
                e.preventDefault();
                if (this.currentFileIndex < this.loadedFiles.length - 1) {
                    this.switchToFile(this.currentFileIndex + 1);
                }
            }
        });

        // Copy data info button
        const copyDataBtn = document.getElementById('copy-data-btn');
        if (copyDataBtn) {
            copyDataBtn.addEventListener('click', () => this.copyDataInfo());
        }

        // Height scaling controls
        const heightModeAuto = document.getElementById('height-mode-auto');
        const heightModeManual = document.getElementById('height-mode-manual');
        const heightMinSlider = document.getElementById('height-min-slider');
        const heightMaxSlider = document.getElementById('height-max-slider');
        const heightMultiplierSlider = document.getElementById('height-multiplier-slider');
        const heightMinValue = document.getElementById('height-min-value');
        const heightMaxValue = document.getElementById('height-max-value');
        const heightMultiplierValue = document.getElementById('height-multiplier-value');

        if (heightModeAuto) {
            heightModeAuto.addEventListener('change', () => {
                this.heightScaleMode = 'auto';
                this.updateHeightControlsVisibility();
                this.updateHeightScaling();
            });
        }

        if (heightModeManual) {
            heightModeManual.addEventListener('change', () => {
                this.heightScaleMode = 'manual';
                this.updateHeightControlsVisibility();
                this.updateHeightScaling();
            });
        }

        if (heightMinSlider) {
            heightMinSlider.addEventListener('input', e => {
                this.manualHeightMin = parseFloat(e.target.value);
                if (heightMinValue) heightMinValue.textContent = this.manualHeightMin.toFixed(1);
                this.updateHeightScaling();
            });
        }

        if (heightMaxSlider) {
            heightMaxSlider.addEventListener('input', e => {
                this.manualHeightMax = parseFloat(e.target.value);
                if (heightMaxValue) heightMaxValue.textContent = this.manualHeightMax.toFixed(1);
                this.updateHeightScaling();
            });
        }

        if (heightMultiplierSlider) {
            heightMultiplierSlider.addEventListener('input', e => {
                this.heightMultiplier = parseFloat(e.target.value);
                if (heightMultiplierValue) heightMultiplierValue.textContent = this.heightMultiplier.toFixed(1);
                this.updateHeightScaling();
            });
        }
    }

    switchView(view) {
        this.currentView = view;
        
        // Hide all viewports
        this.elements.threejsViewport.classList.remove('active');
        this.elements.intensityViewport.classList.remove('active');
        if (this.elements.datasetGridViewport) {
            this.elements.datasetGridViewport.classList.remove('active');
        }
        
        // Deactivate all buttons
        this.elements.view3dBtn.classList.remove('active');
        this.elements.viewIntensityBtn.classList.remove('active');
        if (this.elements.viewDatasetsBtn) {
            this.elements.viewDatasetsBtn.classList.remove('active');
        }
        
        // Show selected viewport and activate button
        if (view === '3d') {
            this.elements.threejsViewport.classList.add('active');
            this.elements.view3dBtn.classList.add('active');
        } else if (view === 'intensity') {
            this.elements.intensityViewport.classList.add('active');
            this.elements.viewIntensityBtn.classList.add('active');
        } else if (view === 'datasets' && this.elements.datasetGridViewport) {
            this.elements.datasetGridViewport.classList.add('active');
            if (this.elements.viewDatasetsBtn) {
                this.elements.viewDatasetsBtn.classList.add('active');
            }
            this.renderDatasetGrid();
        }
    }

    updateHeightControlsVisibility() {
        const manualControls = document.getElementById('manual-height-controls');
        if (manualControls) {
            manualControls.style.display = this.heightScaleMode === 'manual' ? 'block' : 'none';
        }
    }

    generateThumbnail(processor) {
        const canvas = document.createElement('canvas');
        const size = 128;
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        
        const intensityData = processor.rawIntensityData;
        const width = processor.width;
        const height = processor.height;
        const range = processor.intensityMax - processor.intensityMin;
        
        // Downsample to thumbnail size
        const imageData = ctx.createImageData(size, size);
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const srcX = Math.floor((x / size) * width);
                const srcY = Math.floor((y / size) * height);
                const srcIdx = srcY * width + srcX;
                const val = intensityData[srcIdx];
                const normalized = isFinite(val) ? Math.max(0, Math.min(1, (val - processor.intensityMin) / range)) : 0;
                const gray = Math.floor(normalized * 255);
                const idx = (y * size + x) * 4;
                imageData.data[idx] = imageData.data[idx + 1] = imageData.data[idx + 2] = gray;
                imageData.data[idx + 3] = 255;
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
        return canvas.toDataURL();
    }

    switchToFile(index) {
        if (index < 0 || index >= this.loadedFiles.length) return;
        
        this.currentFileIndex = index;
        const fileData = this.loadedFiles[index];
        
        // Replace current processor
        this.imageProcessor = fileData.processor;
        
        // Reset inference results
        this.differenceData = null;
        this.differenceTexture = null;
        this.elements.percentileControls.style.display = 'none';
        
        // Update UI
        this.onFileLoaded();
        this.renderThumbnailGallery(); // Update selection highlight
        
        // Update file counter
        const fileCounter = document.getElementById('file-counter');
        if (fileCounter && this.loadedFiles.length > 1) {
            fileCounter.textContent = `File ${index + 1} of ${this.loadedFiles.length}: ${fileData.name}`;
            fileCounter.style.display = 'block';
        } else if (fileCounter && this.loadedFiles.length === 1) {
            fileCounter.textContent = fileData.name;
            fileCounter.style.display = 'block';
        } else if (fileCounter) {
            fileCounter.style.display = 'none';
        }
        
        console.log(`Switched to: ${fileData.name} (${index + 1}/${this.loadedFiles.length})`);
    }

    renderThumbnailGallery() {
        const gallery = document.getElementById('thumbnail-gallery');
        if (!gallery || this.loadedFiles.length <= 1) {
            if (gallery) gallery.style.display = 'none';
            return;
        }
        
        gallery.style.display = 'flex';
        gallery.innerHTML = '';
        
        for (let i = 0; i < this.loadedFiles.length; i++) {
            const fileData = this.loadedFiles[i];
            const item = document.createElement('div');
            item.className = 'thumbnail-item';
            if (i === this.currentFileIndex) {
                item.classList.add('active');
            }
            
            const img = document.createElement('img');
            img.src = fileData.thumbnail;
            img.alt = fileData.name;
            
            const label = document.createElement('div');
            label.className = 'thumbnail-label';
            label.textContent = fileData.name;
            label.title = fileData.name;
            
            item.appendChild(img);
            item.appendChild(label);
            
            item.addEventListener('click', () => this.switchToFile(i));
            
            gallery.appendChild(item);
        }
        
        // Add navigation buttons
        const prevBtn = document.createElement('button');
        prevBtn.className = 'nav-btn prev-btn';
        prevBtn.innerHTML = '◀';
        prevBtn.title = 'Previous';
        prevBtn.onclick = () => {
            if (this.currentFileIndex > 0) {
                this.switchToFile(this.currentFileIndex - 1);
            }
        };
        prevBtn.disabled = this.currentFileIndex === 0;
        
        const nextBtn = document.createElement('button');
        nextBtn.className = 'nav-btn next-btn';
        nextBtn.innerHTML = '▶';
        nextBtn.title = 'Next';
        nextBtn.onclick = () => {
            if (this.currentFileIndex < this.loadedFiles.length - 1) {
                this.switchToFile(this.currentFileIndex + 1);
            }
        };
        nextBtn.disabled = this.currentFileIndex === this.loadedFiles.length - 1;
        
        gallery.insertBefore(prevBtn, gallery.firstChild);
        gallery.appendChild(nextBtn);
    }

    renderDatasetGrid() {
        const gridContainer = this.elements.datasetGridViewport;
        gridContainer.innerHTML = ''; // Clear existing content
        
        const datasets = Object.entries(this.imageProcessor.allDatasets);
        if (datasets.length === 0) {
            const message = document.createElement('div');
            message.textContent = 'No datasets loaded. Please upload or select a DATX file.';
            message.style.textAlign = 'center';
            message.style.padding = '50px';
            message.style.color = '#888';
            message.style.fontSize = '16px';
            gridContainer.appendChild(message);
            return;
        }
        
        // Create grid layout
        const cols = Math.ceil(Math.sqrt(datasets.length));
        gridContainer.style.display = 'grid';
        gridContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        gridContainer.style.gap = '15px';
        gridContainer.style.padding = '20px';
        gridContainer.style.overflowY = 'auto';
        
        for (const [path, dataset] of datasets) {
            const panel = document.createElement('div');
            panel.className = 'dataset-panel';
            panel.style.background = 'rgba(0,0,0,0.5)';
            panel.style.borderRadius = '8px';
            panel.style.padding = '10px';
            panel.style.border = '1px solid #333';
            
            const title = document.createElement('div');
            title.textContent = dataset.name;
            title.style.fontSize = '12px';
            title.style.marginBottom = '8px';
            title.style.color = '#ccc';
            title.style.textAlign = 'center';
            title.style.fontWeight = 'bold';
            
            const info = document.createElement('div');
            const { min, max } = this.imageProcessor.datasetInfo[path];
            info.textContent = `${dataset.width}×${dataset.height} | [${min.toFixed(1)}, ${max.toFixed(1)}]`;
            info.style.fontSize = '10px';
            info.style.marginBottom = '8px';
            info.style.color = '#888';
            info.style.textAlign = 'center';
            
            const canvasContainer = document.createElement('div');
            canvasContainer.style.display = 'flex';
            canvasContainer.style.justifyContent = 'center';
            canvasContainer.style.alignItems = 'center';
            canvasContainer.style.minHeight = '200px';
            
            const canvas = document.createElement('canvas');
            canvas.style.maxWidth = '100%';
            canvas.style.maxHeight = '300px';
            canvas.style.width = 'auto';
            canvas.style.height = 'auto';
            canvas.style.border = '1px solid #555';
            
            this.imageProcessor.displayDataset(canvas, path);
            
            canvasContainer.appendChild(canvas);
            panel.appendChild(title);
            panel.appendChild(info);
            panel.appendChild(canvasContainer);
            gridContainer.appendChild(panel);
        }
    }

    updateHeightScaling() {
        if (!this.pointsMaterial || !this.imageLoaded) return;
        
        // Determine effective min/max based on mode
        let effectiveMin, effectiveMax;
        if (this.heightScaleMode === 'manual') {
            effectiveMin = this.manualHeightMin;
            effectiveMax = this.manualHeightMax;
        } else {
            effectiveMin = this.imageProcessor.surfaceMin;
            effectiveMax = this.imageProcessor.surfaceMax;
        }
        
        // Update shader uniforms - GPU recalculates everything automatically!
        this.pointsMaterial.uniforms.heightMin.value = effectiveMin;
        this.pointsMaterial.uniforms.heightMax.value = effectiveMax;
        this.pointsMaterial.uniforms.heightMultiplier.value = this.heightMultiplier;
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
            
            const processor = new ImageProcessor();
            const success = await processor.loadDATXFile(arrayBuffer);
            
            if (success) {
                this.imageProcessor = processor;
                
                // Add to loaded files for gallery
                const filename = path.split('/').pop();
                const thumbnail = this.generateThumbnail(processor);
                this.loadedFiles = [{
                    name: filename,
                    processor: processor,
                    thumbnail: thumbnail
                }];
                this.currentFileIndex = 0;
                
                this.onFileLoaded();
                this.renderThumbnailGallery();
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

    async loadMultipleDatxFiles(files) {
        const datxFiles = files.filter(f => f.name.toLowerCase().endsWith('.datx'));
        if (datxFiles.length === 0) {
            alert('No DATX files found. Please select .datx files.');
            return;
        }
        
        console.log(`Loading ${datxFiles.length} DATX file(s)...`);
        this.loadedFiles = [];
        
        // Show loading indicator
        const loadingMsg = document.getElementById('loading-message');
        if (loadingMsg) {
            loadingMsg.style.display = 'block';
            loadingMsg.textContent = `Loading 0/${datxFiles.length} files...`;
        }
        
        for (let i = 0; i < datxFiles.length; i++) {
            const file = datxFiles[i];
            if (loadingMsg) {
                loadingMsg.textContent = `Loading ${i + 1}/${datxFiles.length}: ${file.name}`;
            }
            
            try {
                const arrayBuffer = await file.arrayBuffer();
                const processor = new ImageProcessor();
                const success = await processor.loadDATXFile(arrayBuffer);
                
                if (success) {
                    // Generate thumbnail
                    const thumbnail = this.generateThumbnail(processor);
                    
                    this.loadedFiles.push({
                        name: file.name,
                        processor: processor,
                        thumbnail: thumbnail
                    });
                    
                    console.log(`Loaded: ${file.name}`);
                }
            } catch (error) {
                console.error(`Error loading ${file.name}:`, error);
            }
        }
        
        if (loadingMsg) {
            loadingMsg.style.display = 'none';
        }
        
        if (this.loadedFiles.length > 0) {
            this.switchToFile(0);
            this.renderThumbnailGallery();
        } else {
            alert('Failed to load any files.');
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
        this.differenceTexture = null;
        this.elements.percentileControls.style.display = 'none';
        
        // Update manual height controls with actual data range
        const heightMinSlider = document.getElementById('height-min-slider');
        const heightMaxSlider = document.getElementById('height-max-slider');
        if (heightMinSlider && heightMaxSlider) {
            const dataMin = this.imageProcessor.surfaceMin;
            const dataMax = this.imageProcessor.surfaceMax;
            const range = dataMax - dataMin;
            
            // Set slider ranges to accommodate the data
            heightMinSlider.min = Math.floor(dataMin - range * 0.2);
            heightMinSlider.max = Math.ceil(dataMax + range * 0.2);
            heightMinSlider.value = dataMin;
            this.manualHeightMin = dataMin;
            
            heightMaxSlider.min = Math.floor(dataMin - range * 0.2);
            heightMaxSlider.max = Math.ceil(dataMax + range * 0.2);
            heightMaxSlider.value = dataMax;
            this.manualHeightMax = dataMax;
            
            document.getElementById('height-min-value').textContent = dataMin.toFixed(1);
            document.getElementById('height-max-value').textContent = dataMax.toFixed(1);
            
            console.log(`Surface data range: ${dataMin.toFixed(2)} to ${dataMax.toFixed(2)}`);
        }
        
        // Update data info display
        this.updateDataInfoDisplay();
        
        this.imageProcessor.displayIntensity(this.elements.intensityCanvas);
        this.createSurfaceVisualization();
    }

    updateDataInfoDisplay() {
        const dataInfo = document.getElementById('data-info');
        if (!dataInfo) return;
        
        const width = this.imageProcessor.width;
        const height = this.imageProcessor.height;
        const surfaceMin = this.imageProcessor.surfaceMin;
        const surfaceMax = this.imageProcessor.surfaceMax;
        const surfaceRange = surfaceMax - surfaceMin;
        const intensityMin = this.imageProcessor.intensityMin;
        const intensityMax = this.imageProcessor.intensityMax;
        
        document.getElementById('dimensions-display').textContent = `${width} × ${height}`;
        document.getElementById('surface-min-display').textContent = surfaceMin.toFixed(3);
        document.getElementById('surface-max-display').textContent = surfaceMax.toFixed(3);
        document.getElementById('surface-range-display').textContent = surfaceRange.toFixed(3);
        document.getElementById('intensity-min-display').textContent = intensityMin.toFixed(2);
        document.getElementById('intensity-max-display').textContent = intensityMax.toFixed(2);
        
        dataInfo.style.display = 'block';
    }

    copyDataInfo() {
        const filename = this.loadedFiles.length > 0 && this.currentFileIndex >= 0 
            ? this.loadedFiles[this.currentFileIndex].name 
            : 'Unknown';
        
        const text = `Data Information: ${filename}
Dimensions: ${this.imageProcessor.width} × ${this.imageProcessor.height}
Depth Min: ${this.imageProcessor.surfaceMin.toFixed(3)}
Depth Max: ${this.imageProcessor.surfaceMax.toFixed(3)}
Depth Range: ${(this.imageProcessor.surfaceMax - this.imageProcessor.surfaceMin).toFixed(3)}
Intensity Min: ${this.imageProcessor.intensityMin.toFixed(2)}
Intensity Max: ${this.imageProcessor.intensityMax.toFixed(2)}`;
        
        navigator.clipboard.writeText(text).then(() => {
            const btn = document.getElementById('copy-data-btn');
            if (btn) {
                const originalText = btn.innerHTML;
                btn.innerHTML = '✓';
                btn.style.color = '#4caf50';
                btn.style.borderColor = '#4caf50';
                setTimeout(() => {
                    btn.innerHTML = originalText;
                    btn.style.color = '#888';
                    btn.style.borderColor = '#555';
                }, 1500);
            }
        }).catch(err => {
            console.error('Failed to copy:', err);
            alert('Failed to copy to clipboard');
        });
    }

    createSurfaceVisualization() {
        console.time('Create Surface Visualization (GPU-accelerated)');
        
        if (this.points) {
            this.scene.remove(this.points);
            this.points = null;
        }

        const surfaceBuffer = this.imageProcessor.rawSurfaceData;
        
        // Determine scaling parameters based on mode
        let effectiveMin, effectiveMax;
        if (this.heightScaleMode === 'manual') {
            effectiveMin = this.manualHeightMin;
            effectiveMax = this.manualHeightMax;
        } else {
            effectiveMin = this.imageProcessor.surfaceMin;
            effectiveMax = this.imageProcessor.surfaceMax;
        }
        
        const surfaceRange = effectiveMax - effectiveMin;
        if (surfaceRange === 0) {
            console.warn('Surface range is zero, cannot visualize');
            return;
        }
        
        // Create a DataTexture from the surface data for GPU access
        const width = 1024;
        const height = 1024;
        this.surfaceTexture = new THREE.DataTexture(
            surfaceBuffer,
            width,
            height,
            THREE.RedFormat,
            THREE.FloatType
        );
        this.surfaceTexture.needsUpdate = true;
        
        // Create a grid geometry (points will be displaced by shader)
        const geometry = new THREE.PlaneGeometry(102.4, 102.4, width - 1, height - 1);
        
        // Custom shader material for GPU-accelerated height displacement
        const material = new THREE.ShaderMaterial({
            uniforms: {
                surfaceTexture: { value: this.surfaceTexture },
                heightMin: { value: effectiveMin },
                heightMax: { value: effectiveMax },
                heightMultiplier: { value: this.heightMultiplier },
                pointColor: { value: new THREE.Color(0.5, 0.5, 0.5) },
                highlightColor: { value: new THREE.Color(1, 0, 0) },
                differenceTexture: { value: null },
                threshold: { value: 0 },
                hasHighlight: { value: 0 }
            },
            vertexShader: `
                uniform sampler2D surfaceTexture;
                uniform float heightMin;
                uniform float heightMax;
                uniform float heightMultiplier;
                uniform sampler2D differenceTexture;
                uniform float threshold;
                uniform float hasHighlight;
                
                varying vec3 vColor;
                
                void main() {
                    // Get UV coordinates
                    vec2 uv = vec2(position.x / 102.4 + 0.5, position.y / 102.4 + 0.5);
                    
                    // Sample height from texture
                    float height = texture2D(surfaceTexture, uv).r;
                    
                    // Check if valid (not NaN/Inf)
                    bool isValid = (height == height) && (height < 1e30) && (height > -1e30);
                    
                    vec3 pos;
                    if (isValid) {
                        // Normalize and scale height
                        float normalizedHeight = (height - heightMin) / (heightMax - heightMin);
                        float scaledHeight = normalizedHeight * heightMultiplier;
                        
                        // Map to correct axes: X stays X, height goes to Y (up), Z comes from position.y
                        pos = vec3(position.x, scaledHeight, position.y);
                        
                        // Check for highlighting
                        if (hasHighlight > 0.5) {
                            float diff = texture2D(differenceTexture, uv).r;
                            if (diff >= threshold) {
                                vColor = vec3(1.0, 0.0, 0.0);
                            } else {
                                vColor = vec3(0.5, 0.5, 0.5);
                            }
                        } else {
                            vColor = vec3(0.5, 0.5, 0.5);
                        }
                    } else {
                        // Hide invalid points by collapsing them
                        pos = vec3(0.0, -1000.0, 0.0);
                        vColor = vec3(0.0, 0.0, 0.0);
                    }
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                    gl_PointSize = 1.0;
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                
                void main() {
                    gl_FragColor = vec4(vColor, 1.0);
                }
            `,
            side: THREE.DoubleSide
        });
        
        // Use Points for point cloud rendering
        this.points = new THREE.Points(geometry, material);
        this.scene.add(this.points);
        
        // Store references for updates
        this.pointsMaterial = material;
        
        console.timeEnd('Create Surface Visualization (GPU-accelerated)');
        console.log(`GPU rendering: ${width * height} points processed in parallel on GPU`);
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
        this.differenceTexture = null; // Reset texture so it gets recreated with new data
        
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
        if (!this.pointsMaterial) return;
        
        // Create difference texture if not exists
        if (!this.differenceTexture) {
            this.differenceTexture = new THREE.DataTexture(
                this.differenceData,
                1024,
                1024,
                THREE.RedFormat,
                THREE.FloatType
            );
            this.differenceTexture.needsUpdate = true;
            this.pointsMaterial.uniforms.differenceTexture.value = this.differenceTexture;
        }
        
        // Update shader uniforms - GPU does the rest!
        this.pointsMaterial.uniforms.threshold.value = threshold;
        this.pointsMaterial.uniforms.hasHighlight.value = 1;
        this.differenceTexture.needsUpdate = true;
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
