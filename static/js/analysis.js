function bindUploadEvents() {
    if (!uploadArea || !imageInput) return;

    uploadArea.addEventListener('click', function (e) {
        if (e.target !== imageInput) {
            selectFile();
        }
    });

    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function () {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageUpload(files[0]);
        }
    });

    if (uploadBtn) {
        uploadBtn.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            selectFile();
        });
    }

    imageInput.addEventListener('change', function (e) {
        if (e.target.files.length > 0) {
            handleImageUpload(e.target.files[0]);
        }
    });
}

function selectFile() {
    const input = document.getElementById('imageInput');
    if (input) input.click();
}

function handleImageUpload(file) {
    const maxSizeMb = 15;
    const maxSizeBytes = maxSizeMb * 1024 * 1024;

    function notify(message, type) {
        if (typeof showAppMessage === 'function') {
            showAppMessage(message, type || 'error');
        } else if (results) {
            results.innerHTML = `<div class="results">${message}</div>`;
        } else {
            console.warn(message);
        }
    }

    if (!file || !file.type.startsWith('image/')) {
        notify('Оберіть файл зображення.', 'error');
        return;
    }

    if (file.size > maxSizeBytes) {
        selectedImage = null;

        if (imageInput) {
            imageInput.value = '';
        }

        if (imagePreview) {
            imagePreview.innerHTML = '';
        }

        if (analyzeBtn) {
            analyzeBtn.disabled = true;
        }

        notify(`Файл занадто великий. Максимальний розмір — ${maxSizeMb} МБ.`, 'error');
        return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {
        selectedImage = e.target.result;

        if (imagePreview) {
            imagePreview.innerHTML = `<img src="${selectedImage}" class="preview" alt="Preview">`;
        }

        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }

        if (results) {
            results.innerHTML = '';
        }
    };

    reader.onerror = function () {
        selectedImage = null;

        if (analyzeBtn) {
            analyzeBtn.disabled = true;
        }

        notify('Не вдалося прочитати файл. Спробуйте вибрати інше зображення.', 'error');
    };

    reader.readAsDataURL(file);
}

function isAuthenticatedUser() {
    const userBlocks = document.getElementById('userBlocks');
    return Boolean(userBlocks && userBlocks.style.display !== 'none');
}

function normalizeGeminiError(errorText) {
    if (!errorText) return '';

    const text = String(errorText);

    if (
        text.includes('503') ||
        text.includes('UNAVAILABLE') ||
        text.includes('high demand') ||
        text.includes('Service Unavailable')
    ) {
        return 'Gemini тимчасово перевантажений. Локальний аналіз архітектурного стилю вже виконано. Спробуйте розширений AI-опис ще раз через кілька хвилин.';
    }

    if (
        text.includes('429') ||
        text.includes('RESOURCE_EXHAUSTED') ||
        text.includes('quota')
    ) {
        return 'Ліміт Gemini API тимчасово вичерпано. Локальний аналіз стилю виконано, але розширений AI-опис зараз недоступний.';
    }

    return text;
}

function extractGeminiText(data) {
    const result = data.data || data;
    const gemini = result.gemini_analysis || {};

    return (
        gemini.analysis ||
        gemini.description ||
        gemini.summary ||
        gemini.historical_context ||
        result.ai_analysis ||
        normalizeGeminiError(gemini.error) ||
        ''
    );
}

async function runGeminiAnalysis(baseBody, historyId) {
    if (typeof updateGeminiBlock === 'function') {
        updateGeminiBlock('Gemini AI аналізує зображення. Це може зайняти кілька секунд...');
    }

    try {
        const body = {
            ...baseBody,
            history_id: historyId || null,
        };

        const response = await fetch('/api/analyze/gemini', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Помилка Gemini');
        }

        const geminiText = extractGeminiText(data);

        if (geminiText && typeof updateGeminiBlock === 'function') {
            updateGeminiBlock(geminiText);
        }

        if (typeof loadUserHistory === 'function') {
            loadUserHistory();
        }

    } catch (error) {
        const message = normalizeGeminiError(error.message);

        if (typeof updateGeminiBlock === 'function') {
            updateGeminiBlock(`**Статус:** ${message || 'Gemini тимчасово недоступний. Спробуйте пізніше.'}`);
        }

        console.error('Gemini analysis error:', error);
    }
}

async function analyzeImage() {
    if (!selectedImage) return;

    if (analyzeBtn) analyzeBtn.disabled = true;

    if (results) {
        results.innerHTML = '<div class="loading">Визначаю архітектурний стиль...</div>';
    }

    const body = {
        image: selectedImage.split(',')[1],
    };

    if (currentAnalysisMode === 'tta') {
        body.use_tta = true;
    }

    const selector = document.getElementById('modelSelector');
    if (selector) {
        body.model_type = selector.value;
    }

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Помилка аналізу');
        }

        displayResults(data);

        if (typeof loadUserHistory === 'function' && isAuthenticatedUser()) {
            loadUserHistory();
        }

        if (typeof loadUserStats === 'function' && isAuthenticatedUser()) {
            loadUserStats();
        }

        if (analyzeBtn) analyzeBtn.disabled = false;

        if (isAuthenticatedUser()) {
            runGeminiAnalysis(body, data.history_id);
        }

    } catch (error) {
        if (results) {
            const friendlyMessage = typeof normalizeAnalyzeError === 'function'
                ? normalizeAnalyzeError(error)
                : (error.message || 'Сталася помилка аналізу.');

            results.innerHTML = `<div class="results">${friendlyMessage}</div>`;
        }

        if (analyzeBtn) analyzeBtn.disabled = false;

        console.error('Analyze error:', error);
    }
}

function displayResults(data) {
    let html = '<div class="results">';
    let styleName = null;
    let originalStyleName = null;
    let confidence = 0;

    const result = data.data || data;

    if (result.architectural_style) {
        const style = result.architectural_style;

        if (style.top_prediction) {
            originalStyleName = style.top_prediction.style;
            styleName = style.top_prediction.style_uk || style.top_prediction.style;
            confidence = style.top_prediction.confidence || 0;
        } else if (style.all_predictions && style.all_predictions.length > 0) {
            originalStyleName = style.all_predictions[0].style;
            styleName = style.all_predictions[0].style_uk || style.all_predictions[0].style;
            confidence = style.all_predictions[0].confidence || 0;
        } else {
            styleName = 'Unknown style';
            originalStyleName = null;
            confidence = 0;
        }

        if (result.style_mapping && result.style_mapping[originalStyleName]) {
            styleName = result.style_mapping[originalStyleName];
        }

        console.log('Final style names - display:', styleName, 'original:', originalStyleName, 'confidence:', confidence);

        html += `
            <div class="style-result">
                Архітектурний стиль: ${styleName || 'Невідомий'}
                <span class="confidence">${Math.round((confidence || 0) * 100)}%</span>
            </div>
        `;
    }

    // Карта
    if (originalStyleName && result.architectural_style) {
        if (typeof addStyleToMap === 'function') {
            addStyleToMap(originalStyleName, styleName);
        }
    }

    html += '</div>';

    if (results) {
        results.innerHTML = html;
    }
}

window.analyzeImage = analyzeImage;
window.selectFile = selectFile;