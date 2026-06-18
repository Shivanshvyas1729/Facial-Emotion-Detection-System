document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Navigation
    const tabUploadBtn = document.getElementById('tab-upload-btn');
    const tabCameraBtn = document.getElementById('tab-camera-btn');
    const panelUpload = document.getElementById('panel-upload');
    const panelCamera = document.getElementById('panel-camera');
    const inputCardTitle = document.getElementById('input-card-title');

    // DOM Elements - Upload
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');

    // DOM Elements - Camera
    const webcamVideo = document.getElementById('webcam-video');
    const toggleCameraBtn = document.getElementById('toggle-camera-btn');
    const captureBtn = document.getElementById('capture-btn');
    const cameraSelect = document.getElementById('camera-select');

    // DOM Elements - Results & State
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const spinnerContainer = document.getElementById('spinner-container');
    const resultsContent = document.getElementById('results-content');
    const outputImage = document.getElementById('output-image');
    const detectionList = document.getElementById('detection-list');

    // Global variables
    let cameraStream = null;
    let hasEnumeratedDevices = false;

    // --- Tab Switching Logic ---
    tabUploadBtn.addEventListener('click', () => {
        switchTab('upload');
    });

    tabCameraBtn.addEventListener('click', () => {
        switchTab('camera');
    });

    function switchTab(mode) {
        if (mode === 'upload') {
            tabUploadBtn.classList.add('active');
            tabCameraBtn.classList.remove('active');
            panelUpload.classList.add('active');
            panelCamera.classList.remove('active');
            inputCardTitle.innerHTML = '<i class="fa-solid fa-file-image"></i> Select Image Source';
            stopCamera();
        } else {
            tabCameraBtn.classList.add('active');
            tabUploadBtn.classList.remove('active');
            panelCamera.classList.add('active');
            panelUpload.classList.remove('active');
            inputCardTitle.innerHTML = '<i class="fa-solid fa-video"></i> Live Camera Feed';
        }
    }

    // --- Drag and Drop / File Selection Logic ---
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUploadedFile(e.target.files[0]);
        }
    });

    // Drag-over styling cues
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });

    function handleUploadedFile(file) {
        // Validation check
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (PNG, JPG, or JPEG).');
            return;
        }
        
        showLoadingState();

        const formData = new FormData();
        formData.append('file', file);

        fetch('/detect-upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) throw new Error('API server returned error during upload detection.');
            return response.json();
        })
        .then(data => {
            renderResults(data);
        })
        .catch(err => {
            console.error(err);
            showErrorState(err.message || 'Failed to analyze the image.');
        });
    }

    // --- Live Camera Logic ---
    toggleCameraBtn.addEventListener('click', () => {
        if (cameraStream) {
            stopCamera();
        } else {
            startCamera();
        }
    });

    // Handle manual camera selection change
    cameraSelect.addEventListener('change', () => {
        if (cameraStream) {
            stopCamera();
            startCamera();
        }
    });

    function startCamera() {
        toggleCameraBtn.disabled = true;
        toggleCameraBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Initializing...';
        
        const constraints = {
            audio: false,
            video: {
                width: { ideal: 1280 },
                height: { ideal: 960 }
            }
        };

        if (cameraSelect.value) {
            constraints.video.deviceId = { exact: cameraSelect.value };
        } else {
            constraints.video.facingMode = 'user';
        }

        navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            cameraStream = stream;
            webcamVideo.srcObject = stream;
            
            toggleCameraBtn.disabled = false;
            toggleCameraBtn.innerHTML = '<i class="fa-solid fa-video-slash"></i> Stop Camera';
            toggleCameraBtn.classList.remove('btn-secondary');
            toggleCameraBtn.style.backgroundColor = 'var(--accent-red)';
            toggleCameraBtn.style.color = '#ffffff';
            
            captureBtn.disabled = false;

            // Enumerate devices to populate the dropdown (only if labels are now available post-permission)
            if (!hasEnumeratedDevices) {
                navigator.mediaDevices.enumerateDevices()
                .then(devices => {
                    const videoDevices = devices.filter(d => d.kind === 'videoinput');
                    
                    if (videoDevices.length > 1) {
                        cameraSelect.innerHTML = '';
                        
                        videoDevices.forEach((device, index) => {
                            const option = document.createElement('option');
                            option.value = device.deviceId;
                            option.text = device.label || `Camera ${index + 1}`;
                            cameraSelect.appendChild(option);
                        });
                        
                        cameraSelect.style.display = 'block';
                        hasEnumeratedDevices = true;

                        // Auto-select laptop/built-in webcam if it exists in the list and is not currently selected
                        const laptopKeywords = ['integrated', 'built-in', 'internal', 'webcam', 'facetime', 'front', 'hd camera', 'chicony', 'realtek', 'laptop'];
                        const phoneKeywords = ['phone', 'droidcam', 'epoccam', 'continuity', 'virtual', 'iriun', 'camo', 'obs'];

                        let preferredDevice = videoDevices.find(device => {
                            const label = device.label.toLowerCase();
                            return laptopKeywords.some(keyword => label.includes(keyword)) && 
                                   !phoneKeywords.some(keyword => label.includes(keyword));
                        });

                        if (!preferredDevice) {
                            // Secondary fallback: find first device that doesn't explicitly look like a phone/virtual camera
                            preferredDevice = videoDevices.find(device => {
                                const label = device.label.toLowerCase();
                                return !phoneKeywords.some(keyword => label.includes(keyword));
                            });
                        }

                        if (preferredDevice && cameraSelect.value !== preferredDevice.deviceId) {
                            cameraSelect.value = preferredDevice.deviceId;
                            
                            // Check if current track deviceId differs from preferredDevice
                            const activeTrack = stream.getVideoTracks()[0];
                            const activeSettings = activeTrack ? activeTrack.getSettings() : null;
                            if (activeSettings && activeSettings.deviceId !== preferredDevice.deviceId) {
                                // Restart stream with preferred laptop camera
                                stopCamera();
                                startCamera();
                            }
                        }
                    } else {
                        cameraSelect.style.display = 'none';
                    }
                })
                .catch(err => console.error('Error enumerating video devices:', err));
            }
        })
        .catch(err => {
            console.error('Webcam Access Error:', err);
            toggleCameraBtn.disabled = false;
            toggleCameraBtn.innerHTML = '<i class="fa-solid fa-power-off"></i> Start Camera';
            alert('Could not access the camera. Please ensure camera permissions are allowed.');
        });
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        webcamVideo.srcObject = null;
        
        toggleCameraBtn.innerHTML = '<i class="fa-solid fa-power-off"></i> Start Camera';
        toggleCameraBtn.classList.add('btn-secondary');
        toggleCameraBtn.removeAttribute('style');
        
        captureBtn.disabled = true;
        
        // Hide camera selector on stop to keep UI clean
        cameraSelect.style.display = 'none';
        hasEnumeratedDevices = false; // Reset to re-populate on next start if configuration changes
    }

    // Capture off-screen canvas snapshot and POST to /detect-webcam
    captureBtn.addEventListener('click', () => {
        if (!cameraStream) return;

        showLoadingState();

        // Create virtual canvas
        const canvas = document.createElement('canvas');
        canvas.width = webcamVideo.videoWidth || 640;
        canvas.height = webcamVideo.videoHeight || 480;
        
        const ctx = canvas.getContext('2d');
        // Mirror the canvas context horizontally to match mirrored video display if user is looking at screen
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
        
        // Convert to dataUrl base64 JPEG format
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

        fetch('/detect-webcam', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: dataUrl })
        })
        .then(response => {
            if (!response.ok) throw new Error('API server returned error during camera capture analysis.');
            return response.json();
        })
        .then(data => {
            renderResults(data);
        })
        .catch(err => {
            console.error(err);
            showErrorState(err.message || 'Failed to analyze webcam image.');
        });
    });

    // --- Loading, Error, and Output Rendering ---
    function showLoadingState() {
        resultsPlaceholder.style.display = 'none';
        resultsContent.style.display = 'none';
        spinnerContainer.style.display = 'flex';
    }

    function showErrorState(message) {
        spinnerContainer.style.display = 'none';
        resultsContent.style.display = 'none';
        resultsPlaceholder.style.display = 'flex';
        resultsPlaceholder.innerHTML = `
            <span style="color: var(--accent-red);"><i class="fa-solid fa-triangle-exclamation"></i></span>
            <p style="color: var(--accent-red); font-weight: 600;">Error Occurred</p>
            <p style="margin-top: -0.5rem; font-size: 0.9rem;">${message}</p>
            <button class="btn btn-secondary btn-sm" onclick="window.location.reload()" style="margin-top: 1rem; padding: 0.5rem 1rem; font-size: 0.85rem;">
                <i class="fa-solid fa-rotate"></i> Reset Application
            </button>
        `;
    }

    function renderResults(data) {
        spinnerContainer.style.display = 'none';
        resultsPlaceholder.style.display = 'none';
        resultsContent.style.display = 'flex';

        // 1. Show the annotated detection image returned by the server
        outputImage.src = data.image_data;

        // 2. Clear out the previous detection widgets
        detectionList.innerHTML = '';

        const detections = data.detections;

        if (!detections || detections.length === 0) {
            detectionList.innerHTML = `
                <div style="text-align: center; padding: 1.5rem; color: var(--text-muted); font-size: 0.95rem; border: 1px dashed var(--border-color); border-radius: var(--radius-md);">
                    <i class="fa-solid fa-circle-question" style="margin-bottom: 0.5rem; display: block; font-size: 1.5rem; opacity: 0.6;"></i>
                    No faces or emotions detected. Try aligning the camera or choosing a clearer image.
                </div>
            `;
            return;
        }

        // 3. Populate detections list with visual progress meters
        detections.forEach((detection, idx) => {
            const label = detection.label.toLowerCase();
            const confidencePercent = (detection.confidence * 100).toFixed(1);
            
            // Map specific styles (color, icons) to emotions for a gamified, beautiful layout
            let emotionConfig = {
                color: 'var(--primary-color)',
                icon: '<i class="fa-solid fa-user"></i>',
                bgFill: 'var(--primary-grad)'
            };

            if (label.includes('happy')) {
                emotionConfig = {
                    color: 'var(--accent-green)',
                    icon: '<i class="fa-solid fa-face-smile-beam"></i>',
                    bgFill: 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
                };
            } else if (label.includes('sad')) {
                emotionConfig = {
                    color: '#3b82f6',
                    icon: '<i class="fa-solid fa-face-sad-tear"></i>',
                    bgFill: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)'
                };
            } else if (label.includes('angry')) {
                emotionConfig = {
                    color: 'var(--accent-red)',
                    icon: '<i class="fa-solid fa-face-angry"></i>',
                    bgFill: 'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)'
                };
            } else if (label.includes('surprise') || label.includes('shock')) {
                emotionConfig = {
                    color: 'var(--accent-amber)',
                    icon: '<i class="fa-solid fa-face-surprise"></i>',
                    bgFill: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
                };
            } else if (label.includes('fear') || label.includes('scared')) {
                emotionConfig = {
                    color: '#8b5cf6',
                    icon: '<i class="fa-solid fa-face-grimace"></i>',
                    bgFill: 'linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%)'
                };
            } else if (label.includes('disgust')) {
                emotionConfig = {
                    color: '#ec4899',
                    icon: '<i class="fa-solid fa-face-rolling-eyes"></i>',
                    bgFill: 'linear-gradient(135deg, #ec4899 0%, #be185d 100%)'
                };
            } else if (label.includes('neutral') || label.includes('calm')) {
                emotionConfig = {
                    color: '#9ca3af',
                    icon: '<i class="fa-solid fa-face-meh"></i>',
                    bgFill: 'linear-gradient(135deg, #9ca3af 0%, #4b5563 100%)'
                };
            }

            const item = document.createElement('div');
            item.className = 'detection-item';
            item.innerHTML = `
                <div class="detection-header">
                    <span class="emotion-label" style="color: ${emotionConfig.color}">
                        ${emotionConfig.icon} Face #${idx + 1}: ${detection.label}
                    </span>
                    <span class="confidence-value" style="color: ${emotionConfig.color}">
                        ${confidencePercent}% Match
                    </span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" id="bar-${idx}" style="background: ${emotionConfig.bgFill}"></div>
                </div>
            `;
            
            detectionList.appendChild(item);

            // Animate progress bar filling on the next event loop tick
            setTimeout(() => {
                const bar = document.getElementById(`bar-${idx}`);
                if (bar) bar.style.width = `${confidencePercent}%`;
            }, 50);
        });
    }
});
