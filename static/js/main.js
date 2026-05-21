var selectedImage = null;
var architecturalMap = null;
var currentStyleLayers = [];

var currentAnalysisMode = 'standard';

// DOM elements
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

document.addEventListener('DOMContentLoaded', function () {
    cacheDomElements();

    if (typeof bindUploadEvents === 'function') {
        bindUploadEvents();
    }

    if (typeof initMap === 'function') {
        initMap();
    }

    if (typeof checkAuthStatus === 'function') {
        checkAuthStatus();
    }

    if (typeof initModelZoo === 'function') {
        initModelZoo();
    }

    if (typeof loadBackgroundImages === 'function') {
        loadBackgroundImages();
    }
});