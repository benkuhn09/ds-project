// ============================================
// Traffic Accidents Classification - Frontend JS
// ============================================

// Global variables
let modelsData = null;
let featureNames = [];

// API Base URL
const API_BASE = '';

// Model icons mapping
const MODEL_ICONS = {
    'gradient_boosting': '🚀',
    'mlp': '🧠',
    'decision_tree': '🌳',
    'random_forest': '🌲',
    'logistic_regression': '📈',
    'knn': '👥',
    'naive_bayes': '📊'
};

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', async function() {
    await loadModels();
    await loadFeatures();
    setupFormHandlers();
    setupFileUpload();
    initializeSliders();
});

// ============================================
// Load Data from API
// ============================================
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        if (!response.ok) throw new Error('Failed to load models');

        modelsData = await response.json();
        populateModelSelect();
        populateModelCheckboxes();
        displayModelsGrid();
    } catch (error) {
        console.error('Error loading models:', error);
        document.getElementById('models-grid').innerHTML =
            '<div class="error-message"><i class="fas fa-exclamation-triangle"></i> Failed to load models. Please refresh the page.</div>';
    }
}

async function loadFeatures() {
    try {
        const response = await fetch(`${API_BASE}/api/features`);
        if (!response.ok) throw new Error('Failed to load features');

        const data = await response.json();
        featureNames = data.features || [];
    } catch (error) {
        console.error('Error loading features:', error);
    }
}

// ============================================
// Populate UI Elements
// ============================================
function populateModelSelect() {
    const select = document.getElementById('model-select');
    select.innerHTML = '';

    // Add default option
    const defaultModel = modelsData.models.find(m => m.is_default);
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = `🏆 Default (${defaultModel?.name || 'Best Model'})`;
    select.appendChild(defaultOption);

    // Add all models sorted by F1
    modelsData.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.key;
        const icon = MODEL_ICONS[model.key] || '🤖';
        option.textContent = `${icon} ${model.name} (F1: ${(model.metrics.f1 * 100).toFixed(1)}%)`;
        select.appendChild(option);
    });

    // Update badge on change
    select.addEventListener('change', updateModelBadge);
}

function updateModelBadge() {
    const select = document.getElementById('model-select');
    const badge = document.getElementById('selected-model-badge');
    const selectedValue = select.value;

    if (selectedValue) {
        const model = modelsData.models.find(m => m.key === selectedValue);
        badge.innerHTML = `<i class="fas fa-star"></i> <span>${model.name}</span>`;
    } else {
        const defaultModel = modelsData.models.find(m => m.is_default);
        badge.innerHTML = `<i class="fas fa-star"></i> <span>${defaultModel?.name || 'Gradient Boosting'}</span>`;
    }
}

function populateModelCheckboxes() {
    const container = document.getElementById('model-checkboxes');
    container.innerHTML = '';

    modelsData.models.forEach(model => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.name = 'eval-models';
        checkbox.value = model.key;
        checkbox.checked = true;

        const icon = MODEL_ICONS[model.key] || '🤖';
        const badges = model.is_default ? ' ⭐' : '';

        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(` ${icon} ${model.name}${badges}`));
        container.appendChild(label);
    });
}

function displayModelsGrid() {
    const container = document.getElementById('models-grid');
    container.innerHTML = '';

    modelsData.models.forEach((model, index) => {
        const icon = MODEL_ICONS[model.key] || '🤖';
        const rankClass = index === 0 ? 'rank-1' : index === 1 ? 'rank-2' : index === 2 ? 'rank-3' : 'rank-other';
        const cardClasses = ['model-info-card'];
        if (model.is_default) cardClasses.push('is-default');
        if (index === 0) cardClasses.push('is-best');

        const paramsHtml = Object.entries(model.hyperparameters)
            .map(([k, v]) => `${k}: ${v}`)
            .join(' | ');

        const card = document.createElement('div');
        card.className = cardClasses.join(' ');
        card.innerHTML = `
            <div class="model-rank ${rankClass}">#${index + 1}</div>
            <div class="model-icon">${icon}</div>
            <h3>${model.name}</h3>
            <div class="model-type">${model.key.replace(/_/g, ' ').toUpperCase()}</div>
            <div class="model-metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">${(model.metrics.accuracy * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">${(model.metrics.precision * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">${(model.metrics.recall * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">${(model.metrics.f1 * 100).toFixed(1)}%</div>
                </div>
            </div>
            <div class="model-params">
                <strong>⚙️ Params:</strong> ${paramsHtml}
            </div>
        `;
        container.appendChild(card);
    });
}

// ============================================
// Slider Display Functions
// ============================================
function initializeSliders() {
    // Initialize all slider displays
    updateSliderDisplay('crash_hour');
    updateSliderDisplay('crash_day_of_week');
    updateSliderDisplay('crash_month');
    updateSliderDisplay('num_units');
    updateSliderDisplay('damage');
    updateSliderDisplay('injuries_fatal');
    updateSliderDisplay('injuries_incapacitating');
    updateSliderDisplay('injuries_no_indication');
}

function updateSliderDisplay(sliderId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(`${sliderId}_value`);
    if (!slider || !display) return;

    const value = parseFloat(slider.value);

    switch(sliderId) {
        case 'crash_hour':
            const hour = Math.round(value * 23);
            display.textContent = `${hour.toString().padStart(2, '0')}:00`;
            break;
        case 'crash_day_of_week':
            const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
            const dayIndex = Math.round(value * 6);
            display.textContent = days[dayIndex];
            break;
        case 'crash_month':
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            const monthIndex = Math.round(value * 11);
            display.textContent = months[monthIndex];
            break;
        case 'num_units':
            const units = Math.round(1 + value * 4);
            display.textContent = units >= 5 ? '5+' : units.toString();
            break;
        case 'damage':
            if (value < 0.33) display.textContent = '$500';
            else if (value < 0.66) display.textContent = '$1,500';
            else display.textContent = '$1,500+';
            break;
        case 'injuries_fatal':
        case 'injuries_incapacitating':
            const count = Math.round(value * 4);
            display.textContent = count.toString();
            break;
        case 'injuries_no_indication':
            if (value < 0.33) display.textContent = 'Low';
            else if (value < 0.66) display.textContent = 'Med';
            else display.textContent = 'High';
            break;
    }
}

// ============================================
// Mode Switching
// ============================================
function switchMode(mode) {
    // Update buttons
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-${mode}`).classList.add('active');

    // Update panels
    document.querySelectorAll('.mode-panel').forEach(panel => panel.classList.remove('active'));
    document.getElementById(`${mode}-mode`).classList.add('active');

    // Hide results
    document.getElementById('prediction-result').classList.add('hidden');
    document.getElementById('evaluation-result').classList.add('hidden');
}

// ============================================
// Form Handlers
// ============================================
function setupFormHandlers() {
    // Single prediction form
    document.getElementById('prediction-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        await makePrediction();
    });

    // Evaluation form
    document.getElementById('evaluation-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        await evaluateModels();
    });
}

function setupFileUpload() {
    const fileInput = document.getElementById('validation-file');
    const fileInfo = document.getElementById('file-info');
    const uploadCard = document.querySelector('.upload-card');

    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            fileInfo.innerHTML = `<i class="fas fa-check-circle"></i> <span>${file.name} (${formatFileSize(file.size)})</span>`;
            fileInfo.classList.add('has-file');
        } else {
            fileInfo.innerHTML = '<i class="fas fa-file-csv"></i> <span>No file selected</span>';
            fileInfo.classList.remove('has-file');
        }
    });

    // Drag and drop
    uploadCard.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadCard.classList.add('drag-over');
    });

    uploadCard.addEventListener('dragleave', () => {
        uploadCard.classList.remove('drag-over');
    });

    uploadCard.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadCard.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============================================
// Make Prediction
// ============================================
async function makePrediction() {
    const resultPanel = document.getElementById('prediction-result');
    const resultContent = document.getElementById('result-content');

    // Show loading
    resultPanel.classList.remove('hidden');
    resultContent.innerHTML = `
        <div class="loading">
            <div class="loading-spinner"></div>
            <div class="loading-text">🔮 Analyzing crash data...</div>
        </div>
    `;

    // Gather features
    const features = {};

    // Numeric features from sliders
    const sliderFields = ['crash_hour', 'crash_day_of_week', 'crash_month', 'num_units', 'damage',
                          'injuries_fatal', 'injuries_incapacitating', 'injuries_no_indication'];

    sliderFields.forEach(field => {
        const element = document.getElementById(field);
        if (element) {
            features[field] = parseFloat(element.value) || 0;
        }
    });

    // Crash date (using a middle value)
    features['crash_date'] = 0.5;

    // Intersection toggle
    const intersectionCheckbox = document.getElementById('intersection_related_i');
    features['intersection_related_i'] = intersectionCheckbox.checked ? 1 : 0;

    // Categorical features (convert to one-hot encoding)
    const categoricalFields = [
        'traffic_control_device', 'weather_condition', 'lighting_condition',
        'first_crash_type', 'trafficway_type', 'alignment', 'road_defect',
        'prim_contributory_cause'
    ];

    categoricalFields.forEach(field => {
        const element = document.getElementById(field);
        if (element) {
            const value = element.value;
            // Set all related features to false, then set selected one to true
            featureNames.forEach(feat => {
                if (feat.startsWith(field + '_')) {
                    features[feat] = false;
                }
            });
            const oneHotFeature = `${field}_${value}`;
            if (featureNames.includes(oneHotFeature)) {
                features[oneHotFeature] = true;
            }
        }
    });

    // Get selected model
    const modelSelect = document.getElementById('model-select');
    const modelName = modelSelect.value || null;

    try {
        const response = await fetch(`${API_BASE}/api/predict-single`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                features: features,
                model_name: modelName
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const result = await response.json();
        displayPredictionResult(result);

    } catch (error) {
        resultContent.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <span>Error: ${error.message}</span>
            </div>
        `;
    }
}

function displayPredictionResult(result) {
    const resultContent = document.getElementById('result-content');
    const isNoInjury = result.predicted_class === 0;

    const icon = isNoInjury ? '✅' : '🚨';
    const confidencePercent = result.confidence ? (result.confidence * 100).toFixed(1) : null;

    let confidenceHtml = '';
    if (confidencePercent) {
        confidenceHtml = `
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
            </div>
            <div class="confidence-text">🎯 Confidence: ${confidencePercent}%</div>
        `;
    }

    resultContent.innerHTML = `
        <div class="prediction-box ${isNoInjury ? 'no-injury' : 'injury'}">
            <div class="prediction-icon">${icon}</div>
            <div class="prediction-class">${result.predicted_label}</div>
            <div class="prediction-details">
                🤖 Model: ${result.model_used}<br>
                📊 Class: ${result.predicted_class}
            </div>
            ${confidenceHtml}
        </div>
    `;
}

// ============================================
// Evaluate Models
// ============================================
async function evaluateModels() {
    const resultPanel = document.getElementById('evaluation-result');
    const resultContent = document.getElementById('evaluation-content');

    // Get file
    const fileInput = document.getElementById('validation-file');
    if (!fileInput.files.length) {
        alert('📁 Please select a CSV file');
        return;
    }

    // Get selected models
    const checkboxes = document.querySelectorAll('input[name="eval-models"]:checked');
    const selectedModels = Array.from(checkboxes).map(cb => cb.value);

    if (selectedModels.length === 0) {
        alert('🤖 Please select at least one model');
        return;
    }

    // Show loading
    resultPanel.classList.remove('hidden');
    resultContent.innerHTML = `
        <div class="loading">
            <div class="loading-spinner"></div>
            <div class="loading-text">📊 Evaluating models on your data...</div>
        </div>
    `;

    // Prepare form data
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model_names', selectedModels.join(','));

    try {
        const response = await fetch(`${API_BASE}/api/evaluate-models`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Evaluation failed');
        }

        const result = await response.json();
        displayEvaluationResults(result);

    } catch (error) {
        resultContent.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <span>Error: ${error.message}</span>
            </div>
        `;
    }
}

function displayEvaluationResults(result) {
    const resultContent = document.getElementById('evaluation-content');

    let tableHtml = `
        <div class="results-summary">
            📊 Evaluated on <strong>${result.num_samples}</strong> samples
        </div>
        <table class="results-table">
            <thead>
                <tr>
                    <th>🤖 Model</th>
                    <th>📈 Accuracy</th>
                    <th>🎯 Precision</th>
                    <th>🔄 Recall</th>
                    <th>⭐ F1 Score</th>
                </tr>
            </thead>
            <tbody>
    `;

    result.results.forEach((modelResult, index) => {
        const rowClass = index === 0 ? 'best-row' : '';
        const icon = MODEL_ICONS[modelResult.model_key] || '🤖';
        const badge = index === 0 ? '<span class="badge badge-best">🏆 Best</span>' : '';

        tableHtml += `
            <tr class="${rowClass}">
                <td>${icon} ${modelResult.model_name}${badge}</td>
                <td class="metric-cell ${getMetricClass(modelResult.accuracy)}">${(modelResult.accuracy * 100).toFixed(2)}%</td>
                <td class="metric-cell ${getMetricClass(modelResult.precision)}">${(modelResult.precision * 100).toFixed(2)}%</td>
                <td class="metric-cell ${getMetricClass(modelResult.recall)}">${(modelResult.recall * 100).toFixed(2)}%</td>
                <td class="metric-cell ${getMetricClass(modelResult.f1)}">${(modelResult.f1 * 100).toFixed(2)}%</td>
            </tr>
        `;
    });

    tableHtml += '</tbody></table>';

    // Add confusion matrix for best model
    if (result.results.length > 0) {
        const best = result.results[0];
        const cm = best.confusion_matrix;
        const icon = MODEL_ICONS[best.model_key] || '🤖';

        tableHtml += `
            <div class="confusion-section">
                <h4>📊 Confusion Matrix - ${icon} ${best.model_name}</h4>
                <div class="confusion-matrix">
                    <table>
                        <tr>
                            <th></th>
                            <th>✅ Pred: No Injury</th>
                            <th>🚨 Pred: Injury</th>
                        </tr>
                        <tr>
                            <th>✅ Actual: No Injury</th>
                            <td class="cm-tn">${cm[0][0]}</td>
                            <td class="cm-fp">${cm[0][1]}</td>
                        </tr>
                        <tr>
                            <th>🚨 Actual: Injury</th>
                            <td class="cm-fn">${cm[1][0]}</td>
                            <td class="cm-tp">${cm[1][1]}</td>
                        </tr>
                    </table>
                </div>
            </div>
        `;
    }

    resultContent.innerHTML = tableHtml;
}

function getMetricClass(value) {
    if (value >= 0.8) return 'metric-high';
    if (value >= 0.6) return 'metric-medium';
    return 'metric-low';
}

// ============================================
// Selection Helpers
// ============================================
function selectAllModels() {
    document.querySelectorAll('input[name="eval-models"]').forEach(cb => {
        cb.checked = true;
    });
}

function deselectAllModels() {
    document.querySelectorAll('input[name="eval-models"]').forEach(cb => {
        cb.checked = false;
    });
}

// ============================================
// Reset Form
// ============================================
function resetForm() {
    // Reset sliders to defaults that represent a typical "no injury" scenario
    // Based on training data analysis: Class 0 avg damage=0.32, injuries_no_indication=0.72
    document.getElementById('crash_hour').value = 0.5;
    document.getElementById('crash_day_of_week').value = 0.5;
    document.getElementById('crash_month').value = 0.5;
    document.getElementById('num_units').value = 0.3;
    document.getElementById('damage').value = 0.0;  // Minimal damage for no-injury scenario
    document.getElementById('injuries_fatal').value = 0;
    document.getElementById('injuries_incapacitating').value = 0;
    document.getElementById('injuries_no_indication').value = 1.0;  // Maximum = most people had no injury indication

    // Reset toggle
    document.getElementById('intersection_related_i').checked = true;

    // Reset dropdowns
    document.getElementById('traffic_control_device').selectedIndex = 0;
    document.getElementById('weather_condition').selectedIndex = 0;
    document.getElementById('lighting_condition').selectedIndex = 0;
    document.getElementById('first_crash_type').selectedIndex = 0;
    document.getElementById('trafficway_type').selectedIndex = 0;
    document.getElementById('alignment').selectedIndex = 0;
    document.getElementById('road_defect').selectedIndex = 0;
    document.getElementById('prim_contributory_cause').selectedIndex = 0;

    // Reset model select
    document.getElementById('model-select').selectedIndex = 0;
    updateModelBadge();

    // Update slider displays
    initializeSliders();

    // Hide result
    document.getElementById('prediction-result').classList.add('hidden');
}
