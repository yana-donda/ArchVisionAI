function bindUploadEvents() {
    if (!uploadArea || !imageInput) return;

    uploadArea.addEventListener('click', function (e) {
        if (e.target !== imageInput) {
            selectFile();
        }
    });

    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#379683';
    });

    uploadArea.addEventListener('dragleave', function () {
        uploadArea.style.borderColor = '#7395AE';
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#7395AE';

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
    if (!file.type.startsWith('image/')) {
        alert('Оберіть файл зображення');
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
    };

    reader.readAsDataURL(file);
}

async function analyzeImage() {
    if (!selectedImage) return;

    if (analyzeBtn) analyzeBtn.disabled = true;
    if (results) results.innerHTML = '<div class="loading">Аналізую...</div>';

    try {
        const body = {
            image: selectedImage.split(',')[1]
        };

        // TTA режим
        if (currentAnalysisMode === 'tta') {
            body.use_tta = true;
        }

        // Поточна модель
        const selector = document.getElementById('modelSelector');
        if (selector) {
            body.model_type = selector.value;
        }

        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body)
        });

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        if (results) {
            results.innerHTML = '<div class="results">Помилка аналізу: ' + error.message + '</div>';
        }
    } finally {
        if (analyzeBtn) analyzeBtn.disabled = false;
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

    // Gemini block (для зареєстрованих)
    if (result.gemini_analysis && result.gemini_analysis.analysis) {
        if (typeof updateGeminiBlock === 'function') {
            updateGeminiBlock(result.gemini_analysis.analysis);
        }
    } else if (result.gemini_analysis && result.gemini_analysis.description) {
        if (typeof updateGeminiBlock === 'function') {
            updateGeminiBlock(result.gemini_analysis.description);
        }
    } else if (result.ai_analysis) {
        if (typeof updateGeminiBlock === 'function') {
            updateGeminiBlock(result.ai_analysis);
        }
    }

    // Карта
    if (originalStyleName && result.architectural_style) {
        if (typeof addStyleToMap === 'function') {
            addStyleToMap(originalStyleName);
        }
    }

    if (result.historical_period) {
        html += `
            <div class="analysis-section">
                <div class="section-title">Історичний період:</div>
                <div>${result.historical_period}</div>
            </div>
        `;
    }

    if (result.cultural_significance) {
        html += `
            <div class="analysis-section">
                <div class="section-title">Культурне значення:</div>
                <div>${result.cultural_significance}</div>
            </div>
        `;
    }

    html += '</div>';

    if (results) {
        results.innerHTML = html;
    }
}

window.analyzeImage = analyzeImage;
window.selectFile = selectFile;