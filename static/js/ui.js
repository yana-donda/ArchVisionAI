// ui.js

// ==============================
// Модалки
// ==============================
function openModal(modalId) {
    var modal = document.getElementById(modalId);
    if (modal) modal.style.display = 'block';
}

function closeModal(modalId) {
    var modal = document.getElementById(modalId);
    if (modal) modal.style.display = 'none';
}

function switchModal(currentModal, targetModal) {
    closeModal(currentModal);
    openModal(targetModal);
}

// Закриття модалок по кліку поза ними
window.onclick = function (event) {
    var modals = document.querySelectorAll('.modal');
    modals.forEach(function (modal) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
};

// ==============================
// Background images
// ==============================
async function loadBackgroundImages() {
    try {
        const response = await fetch('/api/dataset/images');
        const data = await response.json();
        const parallaxBg = document.querySelector('.parallax-bg');

        if (parallaxBg && data.images && data.images.length > 0) {
            parallaxBg.innerHTML = '';
            data.images.forEach(function (imagePath) {
                const img = document.createElement('img');
                img.src = imagePath;
                img.className = 'bg-image';
                img.loading = 'lazy';
                img.alt = '';
                img.onerror = function () { this.style.display = 'none'; };
                parallaxBg.appendChild(img);
            });
        }
    } catch (error) {
        console.log('Background images not loaded:', error);
    }
}

// ==============================
// Карта
// ==============================
function initMap() {
    if (architecturalMap) return;

    architecturalMap = L.map('architecturalMap').setView([40, 0], 2);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: 'OpenStreetMap contributors'
    }).addTo(architecturalMap);
}

async function addStyleToMap(styleName) {
    if (!architecturalMap) return;

    // Remove previous style markers
    currentStyleLayers.forEach(function (layer) {
        architecturalMap.removeLayer(layer);
    });
    currentStyleLayers = [];

    try {
        const response = await fetch('/models/architectural_styles_geography.json');
        const geoData = await response.json();

        let styleData = null;
        if (geoData.architectural_styles && geoData.architectural_styles[styleName]) {
            styleData = geoData.architectural_styles[styleName];
        } else if (geoData[styleName]) {
            styleData = geoData[styleName];
        }

        if (styleData) {
            // Center map on style region
            if (styleData.regions && styleData.regions.length > 0) {
                const region = styleData.regions[0];
                if (region.center && Array.isArray(region.center)) {
                    architecturalMap.setView([region.center[0], region.center[1]], 5);
                }
            }

            // Add style distribution areas
            if (styleData.regions) {
                styleData.regions.forEach(function (region) {
                    let lat, lng;
                    if (Array.isArray(region.center)) {
                        lat = region.center[0];
                        lng = region.center[1];
                    } else if (region.center && region.center.lat && region.center.lng) {
                        lat = region.center.lat;
                        lng = region.center.lng;
                    } else {
                        return;
                    }

                    const circle = L.circle([lat, lng], {
                        color: '#379683',
                        fillColor: '#7395AE',
                        fillOpacity: 0.3,
                        radius: (region.radius_km || 100) * 1000
                    }).addTo(architecturalMap);

                    circle.bindPopup(`
                        <div style="text-align: center; padding: 10px;">
                            <h4 style="color: var(--teal); margin: 0 0 5px 0;">${region.name}</h4>
                            <p style="margin: 5px 0;"><strong>Style:</strong> ${styleName}</p>
                            <p style="margin: 5px 0;">${region.description}</p>
                        </div>
                    `);

                    currentStyleLayers.push(circle);
                });
            }

            // Add markers for famous buildings
            const buildings = styleData.buildings || styleData.famous_buildings;
            if (buildings) {
                buildings.forEach(function (building) {
                    let lat, lng;
                    if (Array.isArray(building.coordinates)) {
                        lat = building.coordinates[0];
                        lng = building.coordinates[1];
                    } else if (building.lat && building.lon) {
                        lat = building.lat;
                        lng = building.lon;
                    } else if (building.lat && building.lng) {
                        lat = building.lat;
                        lng = building.lng;
                    } else if (Array.isArray(building.location)) {
                        lat = building.location[0];
                        lng = building.location[1];
                    } else if (building.location && building.location.lat && building.location.lng) {
                        lat = building.location.lat;
                        lng = building.location.lng;
                    } else {
                        return;
                    }

                    const marker = L.marker([lat, lng], {
                        icon: L.icon({
                            iconUrl: 'https://cdn-icons-png.flaticon.com/32/684/684908.png',
                            iconSize: [25, 25]
                        })
                    }).addTo(architecturalMap);

                    marker.bindPopup(`
                        <div style="text-align: center; padding: 10px;">
                            <h4 style="color: var(--teal); margin: 0 0 5px 0;">${building.name}</h4>
                            <p style="margin: 5px 0;"><strong>Style:</strong> ${styleName}</p>
                            <p style="margin: 5px 0;">${building.description}</p>
                        </div>
                    `);

                    currentStyleLayers.push(marker);
                });
            }

            // Якщо юзер залогінений — оновити історію
            const userBlocks = document.getElementById('userBlocks');
            if (userBlocks && userBlocks.style.display === 'flex' && typeof loadUserHistory === 'function') {
                loadUserHistory();
            }
        }
    } catch (error) {
        console.log('Failed to load geographical data:', error);
    }
}

function getMarkerColor(style) {
    const colors = {
        'Gothic': '#8B4513',
        'Baroque': '#DAA520',
        'Modern': '#4682B4',
        'Byzantine': '#800080',
        'Renaissance': '#FF4500',
        'Neo-Classical': '#2E8B57',
        'Medieval': '#556B2F',
        'Armenian': '#DC143C'
    };
    return colors[style] || '#666666';
}

// ==============================
// Model Zoo
// ==============================
const modelInfo = {
    'efficientnet_b0': {
        name: 'EfficientNet-B0',
        params: '5.3M',
        input_size: 224,
        batch_size: 32,
        accuracy: '73.64%',
        description: 'Швидка та точна модель, оптимальний баланс'
    },
    'resnet50': {
        name: 'ResNet-50',
        params: '25.6M',
        input_size: 224,
        batch_size: 24,
        accuracy: '71.96%',
        description: 'Класична архітектура, стабільна точність'
    },
    'ensemble': {
        name: 'Ensemble',
        params: '30.9M',
        input_size: 224,
        batch_size: 16,
        accuracy: '~75%',
        description: 'Комбінація обох моделей для вищої точності'
    }
};

function initModelZoo() {
    populateModelSelector();
}

function populateModelSelector() {
    updateModelZooStatus(true, 3);
    updateModelInfo(modelInfo['efficientnet_b0']);
}

function updateModelInfo(model) {
    const modelParams = document.getElementById('modelParams');
    const modelInputSize = document.getElementById('modelInputSize');
    const modelBatchSize = document.getElementById('modelBatchSize');
    const modelDescription = document.getElementById('modelDescription');
    const currentModelName = document.getElementById('currentModelName');

    if (modelParams) modelParams.textContent = model.params || '?';
    if (modelInputSize) modelInputSize.textContent = `${model.input_size || 224}x${model.input_size || 224}`;
    if (modelBatchSize) modelBatchSize.textContent = model.batch_size || 32;
    if (modelDescription) modelDescription.textContent = model.description || 'Немає опису';
    if (currentModelName) currentModelName.textContent = model.name || 'Model';
}

function updateModelZooStatus(available, countOrMessage) {
    const statusBadge = document.getElementById('modelZooStatus');
    if (!statusBadge) return;

    if (available) {
        const count = typeof countOrMessage === 'number' ? countOrMessage : availableModels.length;
        const modelWord = count === 1 ? 'модель' : (count < 5 ? 'моделі' : 'моделей');
        statusBadge.textContent = `${count} ${modelWord}`;
        statusBadge.className = 'model-badge fast';
        statusBadge.style.background = '';
        statusBadge.style.color = '';
    } else {
        statusBadge.textContent = countOrMessage || 'Недоступно';
        statusBadge.className = 'model-badge';
        statusBadge.style.background = '#f8d7da';
        statusBadge.style.color = '#721c24';
    }
}

async function onModelChange() {
    const selector = document.getElementById('modelSelector');
    if (!selector) return;

    const selectedType = selector.value;
    const model = modelInfo[selectedType];

    if (model) {
        updateModelInfo(model);
    }

    try {
        const response = await fetch('/api/models/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_type: selectedType })
        });

        const data = await response.json();

        if (data.success) {
            console.log('Model switched to:', selectedType);
            showModelNotification(`Режим: ${model?.name || selectedType}`);
        } else {
            console.warn('Failed to switch model:', data.message);
            showModelNotification(data.message || 'Помилка перемикання', true);
        }
    } catch (error) {
        console.error('Error switching model:', error);
        showModelNotification('Помилка перемикання моделі', true);
    }
}

function setAnalysisMode(mode) {
    currentAnalysisMode = mode;

    document.querySelectorAll('.analysis-option-btn').forEach(function (btn) {
        btn.classList.remove('active');
    });

    const btnId = mode === 'standard' ? 'btnStandard' : 'btnTTA';
    const activeBtn = document.getElementById(btnId);
    if (activeBtn) activeBtn.classList.add('active');

    console.log('Analysis mode set to:', mode);
}

function showModelNotification(message, isError) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 15px 25px;
        background: ${isError ? '#dc3545' : '#28a745'};
        color: white;
        border-radius: 8px;
        font-weight: bold;
        z-index: 9999;
        animation: slideIn 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(function () {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(function () { toast.remove(); }, 300);
    }, 3000);
}

// CSS анімації для toast
(function injectToastAnimations() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
})();

// На всякий випадок явно експортуємо в window (для inline onclick)
window.openModal = openModal;
window.closeModal = closeModal;
window.switchModal = switchModal;
window.onModelChange = onModelChange;
window.setAnalysisMode = setAnalysisMode;