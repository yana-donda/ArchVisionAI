let availableModels = [];
let modelInfo = {};

// Модалки
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
window.addEventListener('click', function (event) {
    var modals = document.querySelectorAll('.modal');

    modals.forEach(function (modal) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
});

// Background images
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

// Карта
function initMap() {
    if (architecturalMap) return;

    const mapElement = document.getElementById('architecturalMap');

    if (!mapElement || typeof L === 'undefined') {
        console.warn('Map element or Leaflet is not available');
        return;
    }

    architecturalMap = L.map('architecturalMap').setView([40, 0], 2);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: 'OpenStreetMap contributors'
    }).addTo(architecturalMap);
}

async function addStyleToMap(styleName, styleLabelUk) {
    if (!architecturalMap) return;

    // Remove previous style markers
    currentStyleLayers.forEach(function (layer) {
        architecturalMap.removeLayer(layer);
    });
    currentStyleLayers = [];

    try {
        const response = await fetch('/models/architectural_styles_geography.json');
        const geoData = await response.json();

        const trResp = await fetch('/models/architectural_geography_ukrainian.json?v=1');
        const tr = await trResp.json();

        const styleData = geoData[styleName] || null;

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
                        <div class="map-popup">
                            <h4 class="map-popup-title">${(tr.regions && tr.regions[region.name]) || region.name}</h4>
                            <p class="map-popup-text"><strong>Стиль:</strong> ${styleLabelUk || styleName}</p>
                            <p class="map-popup-text">${(tr.descriptions && tr.descriptions[region.description]) || region.description}</p>
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
                            iconUrl: '/static/icons/location_pin.png',
                            iconSize: [25, 25]
                        })
                    }).addTo(architecturalMap);

                    marker.bindPopup(`
                        <div class="map-popup">
                            <h4 class="map-popup-title">${(tr.buildings && tr.buildings[building.name]) || building.name}</h4>
                            <p class="map-popup-text"><strong>Стиль:</strong> ${styleLabelUk || styleName}</p>
                            <p class="map-popup-text">${(tr.building_descriptions && tr.building_descriptions[building.description]) || building.description}</p>
                        </div>
                    `);

                    currentStyleLayers.push(marker);
                });
            }
        }
    } catch (error) {
        console.log('Failed to load geographical data:', error);
    }
}

// Model Zoo
async function initModelZoo() {
    try {
        const response = await fetch('/api/models/available');
        const data = await response.json();

        if (!response.ok || !data.available) {
            throw new Error(data.message || 'Моделі недоступні');
        }

        availableModels = (data.models || []).filter(Boolean);
        modelInfo = {};

        availableModels.forEach(function (model) {
            modelInfo[model.type] = model;
        });

        renderModelOptions(data.current_model || 'efficientnet_b0');
        updateModelZooStatus(true, data.count || availableModels.length);
        updateModelInfo(modelInfo[data.current_model] || modelInfo.efficientnet_b0 || availableModels[0]);
    } catch (error) {
        console.error('Failed to load model zoo:', error);
        updateModelZooStatus(false, 'Недоступно');
    }
}

function renderModelOptions(currentModel) {
    const selector = document.getElementById('modelSelector');
    if (!selector) return;

    selector.innerHTML = '';

    const group = document.createElement('optgroup');
    group.label = 'Режим аналізу';

    availableModels.forEach(function (model) {
        const option = document.createElement('option');
        option.value = model.type;
        option.textContent = model.name;
        option.selected = model.type === currentModel;
        group.appendChild(option);
    });

    selector.appendChild(group);
}

function formatModelParams(model) {
    if (!model) return '—';

    if (model.params) {
        return model.params;
    }

    const value = Number(model.params_millions);
    if (!Number.isFinite(value)) {
        return '—';
    }

    return `${Math.round(value * 10) / 10}M`;
}

function formatInputSize(model) {
    const inputSize = Number(model?.input_size || 224);
    return `${inputSize}×${inputSize}`;
}

function updateModelInfo(model) {
    if (!model) return;

    const modelParams = document.getElementById('modelParams');
    const modelInputSize = document.getElementById('modelInputSize');
    const modelBatchSize = document.getElementById('modelBatchSize');
    const modelDescription = document.getElementById('modelDescription');
    const currentModelName = document.getElementById('currentModelName');

    const displayName = model.full_name || model.name || model.type || 'Модель';

    if (modelParams) {
        modelParams.textContent = formatModelParams(model);
    }

    if (modelInputSize) {
        modelInputSize.textContent = formatInputSize(model);
    }

    if (modelBatchSize) {
        modelBatchSize.textContent = model.batch_size || '—';
    }

    if (modelDescription) {
        modelDescription.textContent = model.description || 'Опис недоступний';
    }

    if (currentModelName) {
        currentModelName.textContent = displayName;
    }
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
            showAppMessage(`Режим: ${model?.name || selectedType}`, 'success');
        } else {
            console.warn('Failed to switch model:', data.message);
            showAppMessage(data.message || 'Помилка перемикання', 'error');
        }
    } catch (error) {
        console.error('Error switching model:', error);
        showAppMessage('Помилка перемикання моделі', 'error');
    }
}

function setAnalysisMode(mode) {
    if (!['standard', 'tta'].includes(mode)) {
        console.warn('Unknown analysis mode:', mode);
        return;
    }

    currentAnalysisMode = mode;

    document.querySelectorAll('.analysis-option-btn').forEach(function (btn) {
        btn.classList.remove('active');
    });

    const btnId = mode === 'standard' ? 'btnStandard' : 'btnTTA';
    const activeBtn = document.getElementById(btnId);
    if (activeBtn) activeBtn.classList.add('active');

    console.log('Analysis mode set to:', mode);
}

function showAppMessage(message, type = 'success') {
    const oldToast = document.getElementById('appMessageToast');
    if (oldToast) {
        oldToast.remove();
    }

    const toast = document.createElement('div');
    toast.id = 'appMessageToast';

    const isError = type === 'error';

    toast.style.cssText = `
        position: fixed;
        right: 20px;
        bottom: 20px;
        max-width: 360px;
        padding: 15px 25px;
        border-radius: 12px;
        background: ${isError ? '#dc3545' : '#28a745'};
        color: #fff;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        z-index: 10000;
        font-size: 15px;
        font-weight: 700;
        line-height: 1.4;
        text-align: center;
        opacity: 1;
        transform: translateX(0);
        transition: opacity 0.25s ease, transform 0.25s ease;
    `;

    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(function () {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';

        setTimeout(function () {
            toast.remove();
        }, 250);
    }, 3000);
}

window.showAppMessage = showAppMessage;
window.openModal = openModal;
window.closeModal = closeModal;
window.switchModal = switchModal;
window.onModelChange = onModelChange;
window.setAnalysisMode = setAnalysisMode;