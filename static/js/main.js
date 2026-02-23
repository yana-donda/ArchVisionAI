// main.js
// Глобальні змінні (shared між файлами)
var selectedImage = null;
var architecturalMap = null;
var currentStyleLayers = [];

var availableModels = [];
var currentAnalysisMode = 'standard';
var loadedModels = [];

// DOM elements (будуть закешовані після завантаження DOM)
var uploadArea = null;
var imageInput = null;
var imagePreview = null;
var analyzeBtn = null;
var results = null;
var uploadBtn = null;

// Кешуємо DOM-елементи
function cacheDomElements() {
    uploadArea = document.getElementById('uploadArea');
    imageInput = document.getElementById('imageInput');
    imagePreview = document.getElementById('imagePreview');
    analyzeBtn = document.getElementById('analyzeBtn');
    results = document.getElementById('results');
    uploadBtn = document.getElementById('uploadBtn');
}

// Єдина ініціалізація сторінки
document.addEventListener('DOMContentLoaded', function () {
    cacheDomElements();

    // Підв’язуємо upload events
    if (typeof bindUploadEvents === 'function') {
        bindUploadEvents();
    }

    // Ініціалізація карти
    if (typeof initMap === 'function') {
        initMap();
    }

    // Перевірка авторизації
    if (typeof checkAuthStatus === 'function') {
        checkAuthStatus();
    }

    // Model Zoo
    if (typeof initModelZoo === 'function') {
        initModelZoo();
    }

    // Background images
    if (typeof loadBackgroundImages === 'function') {
        loadBackgroundImages();
    }
});