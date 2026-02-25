/**
 * MemeShield - Main JavaScript Application
 * Handles file upload, preview, API calls, and UI interactions
 */

// ============================================
// DOM Elements
// ============================================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingState = document.getElementById('loadingState');
const resultsContent = document.getElementById('resultsContent');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

// State
let currentFile = null;
let uploadedFileData = null;

// ============================================
// File Upload Handling
// ============================================

/**
 * Initialize event listeners for file upload
 */
function initUploadHandlers() {
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Remove button
    removeBtn.addEventListener('click', removeImage);
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeMeme);
    
    // New analysis button
    if (newAnalysisBtn) {
        newAnalysisBtn.addEventListener('click', resetAnalysis);
    }
}

/**
 * Handle file selection from input
 */
function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

/**
 * Handle drag over event
 */
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.add('dragover');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove('dragover');
}

/**
 * Handle file drop event
 */
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

/**
 * Process and validate uploaded file
 */
function processFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload PNG, JPG, GIF, or WEBP images.');
        return;
    }
    
    // Validate file size (max 16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }
    
    currentFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

/**
 * Remove uploaded image
 */
function removeImage(event) {
    event.stopPropagation();
    currentFile = null;
    uploadedFileData = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
}

/**
 * Reset analysis for new upload
 */
function resetAnalysis() {
    removeImage({ stopPropagation: () => {} });
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================
// API Calls
// ============================================

/**
 * Upload file to server and analyze
 */
async function analyzeMeme() {
    if (!currentFile) {
        showError('Please select a file first.');
        return;
    }
    
    // Show results section with loading state
    resultsSection.style.display = 'block';
    loadingState.style.display = 'block';
    resultsContent.style.display = 'none';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    try {
        // Step 1: Upload file
        updateLoadingStep(1);
        const uploadResponse = await uploadFile(currentFile);
        
        if (!uploadResponse.success) {
            throw new Error(uploadResponse.error || 'Upload failed');
        }
        
        uploadedFileData = uploadResponse;
        
        // Step 2: Processing image
        updateLoadingStep(2);
        await sleep(500); // Brief pause for UX
        
        // Step 3: Run prediction
        updateLoadingStep(3);
        const predictionResponse = await getPrediction(uploadResponse.filename, uploadResponse.original_filename);
        
        if (!predictionResponse.success) {
            throw new Error(predictionResponse.error || 'Prediction failed');
        }
        
        // Show results
        displayResults(predictionResponse);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'An error occurred during analysis.');
        loadingState.style.display = 'none';
    }
}

/**
 * Upload file to server
 */
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

/**
 * Get prediction for uploaded file
 */
async function getPrediction(filename, originalFilename) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: filename,
            original_filename: originalFilename
        })
    });
    
    return await response.json();
}

/**
 * Update loading step indicator
 */
function updateLoadingStep(step) {
    const steps = document.querySelectorAll('.loading-steps span');
    steps.forEach((el, index) => {
        const icon = el.querySelector('i');
        if (index < step - 1) {
            el.classList.add('complete');
            el.classList.remove('active');
            icon.className = 'fas fa-check-circle';
        } else if (index === step - 1) {
            el.classList.add('active');
            el.classList.remove('complete');
            icon.className = 'fas fa-spinner fa-spin';
        } else {
            el.classList.remove('active', 'complete');
            icon.className = 'fas fa-circle';
        }
    });
}

// ============================================
// Display Results
// ============================================

/**
 * Display prediction results
 */
function displayResults(data) {
    loadingState.style.display = 'none';
    resultsContent.style.display = 'block';
    
    // Update prediction badge
    const predictionBadge = document.getElementById('predictionBadge');
    const predictionLabel = document.getElementById('predictionLabel');
    const confidenceScore = document.getElementById('confidenceScore');
    const badgeIcon = predictionBadge.querySelector('.badge-icon i');
    
    if (data.prediction === 'hate') {
        predictionBadge.className = 'prediction-badge hate';
        predictionLabel.textContent = 'Hate Speech Detected';
        confidenceScore.textContent = `Confidence: ${data.confidence}%`;
        badgeIcon.className = 'fas fa-exclamation-triangle';
    } else {
        predictionBadge.className = 'prediction-badge non-hate';
        predictionLabel.textContent = 'Non-Hate Speech';
        confidenceScore.textContent = `Confidence: ${data.confidence}%`;
        badgeIcon.className = 'fas fa-check-circle';
    }
    
    // Update extracted text
    const extractedTextEl = document.getElementById('extractedText');
    extractedTextEl.innerHTML = `<p>${data.extracted_text || 'No text detected in image'}</p>`;
    
    // Update processed text
    const processedTextEl = document.getElementById('processedText');
    processedTextEl.innerHTML = `<p>${data.processed_text || 'N/A'}</p>`;
    
    // Update analysis status
    document.getElementById('visualStatus').textContent = 'Complete';
    document.getElementById('textStatus').textContent = data.extracted_text ? 'Complete' : 'No Text Found';
    document.getElementById('fusionStatus').textContent = 'Complete';
    
    // Refresh stats
    loadStats();
}

// ============================================
// Statistics
// ============================================

/**
 * Load and display statistics
 */
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.success) {
            animateNumber('totalAnalyzed', data.total_analyzed);
            animateNumber('hateDetected', data.hate_speech_count);
            animateNumber('safeContent', data.non_hate_count);
            document.getElementById('avgConfidence').textContent = `${data.average_confidence}%`;
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

/**
 * Animate number counting
 */
function animateNumber(elementId, targetValue) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const duration = 1000;
    const startValue = 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Ease out cubic
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const currentValue = Math.floor(startValue + (targetValue - startValue) * easeOut);
        
        element.textContent = currentValue;
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// ============================================
// Utility Functions
// ============================================

/**
 * Show error message
 */
function showError(message) {
    // Create error toast
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
    `;
    
    // Add toast styles if not exists
    if (!document.getElementById('toastStyles')) {
        const styles = document.createElement('style');
        styles.id = 'toastStyles';
        styles.textContent = `
            .error-toast {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #ef4444;
                color: white;
                padding: 16px 24px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                gap: 12px;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
                z-index: 9999;
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            
            .error-toast.fadeOut {
                animation: slideOut 0.3s ease forwards;
            }
            
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(styles);
    }
    
    document.body.appendChild(toast);
    
    // Remove after 5 seconds
    setTimeout(() => {
        toast.classList.add('fadeOut');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

/**
 * Sleep utility
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// Keyboard Shortcuts
// ============================================

document.addEventListener('keydown', (event) => {
    // Ctrl/Cmd + U to upload
    if ((event.ctrlKey || event.metaKey) && event.key === 'u') {
        event.preventDefault();
        fileInput.click();
    }
    
    // Enter to analyze (when file is selected)
    if (event.key === 'Enter' && currentFile && !analyzeBtn.disabled) {
        event.preventDefault();
        analyzeMeme();
    }
    
    // Escape to remove image
    if (event.key === 'Escape' && currentFile) {
        removeImage({ stopPropagation: () => {} });
    }
});

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initUploadHandlers();
    loadStats();
    
    console.log('🛡️ MemeShield initialized');
});
