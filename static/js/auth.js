async function register(event) {
    event.preventDefault();

    const username = document.getElementById('registerUsername')?.value;
    const password = document.getElementById('registerPassword')?.value;

    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        const alertEl = document.getElementById('registerAlert');
        if (!alertEl) return;

        if (response.ok) {
            alertEl.innerHTML = '<div class="alert alert-success">Реєстрація успішна! Будь ласка, увійдіть.</div>';
            setTimeout(function () {
                switchModal('registerModal', 'loginModal');
            }, 1500);
        } else {
            alertEl.innerHTML = `<div class="alert alert-error">${data.error}</div>`;
        }
    } catch (error) {
        const alertEl = document.getElementById('registerAlert');
        if (alertEl) {
            alertEl.innerHTML = '<div class="alert alert-error">Помилка реєстрації. Спробуйте ще раз.</div>';
        }
    }
}

async function login(event) {
    event.preventDefault();

    const username = document.getElementById('loginUsername')?.value;
    const password = document.getElementById('loginPassword')?.value;

    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();
        const alertEl = document.getElementById('loginAlert');
        if (!alertEl) return;

        if (response.ok) {
            alertEl.innerHTML = '<div class="alert alert-success">Успішний вхід!</div>';
            setTimeout(function () {
                closeModal('loginModal');
                updateUserInterface({
                    logged_in: true,
                    username: data.user.username
                });
            }, 1000);
        } else {
            alertEl.innerHTML = `<div class="alert alert-error">${data.error}</div>`;
        }
    } catch (error) {
        const alertEl = document.getElementById('loginAlert');
        if (alertEl) {
            alertEl.innerHTML = '<div class="alert alert-error">Помилка входу. Спробуйте ще раз.</div>';
        }
    }
}

async function logout() {
    try {
        const response = await fetch('/api/auth/logout', { method: 'POST' });

        if (!response.ok) {
            throw new Error('Не вдалося виконати вихід');
        }

        updateUserInterface({ logged_in: false });
    } catch (error) {
        console.error('Logout error:', error);

        if (typeof showAppMessage === 'function') {
            showAppMessage('Не вдалося вийти з акаунту. Спробуйте ще раз.', 'error');
        }
    }
}

async function checkAuthStatus() {
    try {
        const response = await fetch('/api/auth/status');
        const data = await response.json();

        if (data.authenticated) {
            updateUserInterface({
                logged_in: true,
                username: data.user.username
            });
        } else {
            updateUserInterface({ logged_in: false });
        }
    } catch (error) {
        console.log('Auth status check error:', error);
    }
}

function resetUserBlocks() {
    const geminiContent = document.getElementById('geminiContent');
    const historyContent = document.getElementById('historyContent');
    const statsContent = document.getElementById('statsContent');

    if (geminiContent) {
        geminiContent.textContent = 'Виконайте аналіз зображення для отримання розширеного аналізу ШІ';
    }

    if (historyContent) {
        historyContent.textContent = 'Ваша історія аналізу буде відображена тут після входу';
    }

    if (statsContent) {
        statsContent.textContent = 'Ваша статистика аналізу буде відображена тут';
    }

    window.userHistoryData = [];
    window.userStatsData = null;
}

// User UI blocks
function updateUserInterface(userData) {
    const guestButtons = document.getElementById('guestButtons');
    const userButtons = document.getElementById('userButtons');
    const welcomeText = document.getElementById('welcomeText');
    const userBlocks = document.getElementById('userBlocks');

    if (userData && userData.logged_in) {
        if (guestButtons) guestButtons.style.display = 'none';
        if (userButtons) userButtons.style.display = 'flex';
        if (welcomeText) welcomeText.textContent = `Вітаємо, ${userData.username}!`;

        // блоки Gemini / History / Stats / Styles
        if (userBlocks) userBlocks.style.display = 'flex';

        if (typeof loadUserHistory === 'function') {
            loadUserHistory();
        }

        if (typeof loadUserStats === 'function') {
            loadUserStats();
        }
    } else {
        if (guestButtons) guestButtons.style.display = 'flex';
        if (userButtons) userButtons.style.display = 'none';
        if (welcomeText) welcomeText.textContent = '';

        if (userBlocks) userBlocks.style.display = 'none';

        resetUserBlocks();
    }
}

// Gemini / history / stats blocks
function updateGeminiBlock(analysisText) {
    const geminiContent = document.getElementById('geminiContent');
    if (!geminiContent || !analysisText) return;

    let formattedText = analysisText;

    formattedText = formattedText.split('**').map(function (part, index) {
        return index % 2 === 1 ? `<strong>${part}</strong>` : part;
    }).join('');

    let lines = formattedText.split('\n');
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].match(/^\d+\./)) {
            lines[i] = '<div class="gemini-line">' + lines[i] + '</div>';
        }
    }
    formattedText = lines.join('<br>');

    geminiContent.innerHTML = `
        <div class="gemini-analysis-box">
            ${formattedText}
        </div>
        <div class="gemini-source">
            Аналіз від Google Gemini Vision
        </div>
    `;
}

function updateHistoryBlock(historyData) {
    const historyContent = document.getElementById('historyContent');
    if (!historyContent || !historyData || !historyData.history) return;

    const fallbackImage = '/static/icons/no-image.svg';

    function buildThumbnailSrc(item) {
        const thumbnail = (item.image_thumbnail || '').trim();

        if (thumbnail.startsWith('data:image/')) {
            return thumbnail;
        }

        return fallbackImage;
    }

    if (historyData.history.length === 0) {
        historyContent.innerHTML = '<p class="empty-state">Немає історії аналізів</p>';
        return;
    }

    const historyHTML = historyData.history.slice(0, 10).map(function (item) {
        const thumbnailSrc = buildThumbnailSrc(item);

        const confidence = Math.round((item.confidence || 0) * 100);
        const date = new Date(item.created_at).toLocaleDateString('uk-UA');

        return `
            <div class="history-item">
                <img src="${thumbnailSrc}"
                    onerror="this.onerror=null;this.src='${fallbackImage}'"
                    class="history-thumb"
                    alt="">
                <div class="history-info">
                    <div class="history-style">
                        ${item.architectural_style || 'Невідомий стиль'}
                    </div>
                    <div class="history-confidence">
                        Впевненість: ${confidence}%
                    </div>
                    <div class="history-date">
                        ${date}
                    </div>
                </div>
            </div>
        `;
    }).join('');

    historyContent.innerHTML = `
        <div class="history-list">
            ${historyHTML}
        </div>
        <div class="history-footer">
            <small class="history-footer-text">
                Останні ${Math.min(historyData.history.length, 10)} аналізів
            </small>
        </div>
    `;
}

async function loadUserHistory() {
    try {
        const response = await fetch('/api/user/history');
        if (response.ok) {
            const historyData = await response.json();
            window.userHistoryData = historyData.history;
            updateHistoryBlock(historyData);
        }
    } catch (error) {
        console.log('Failed to load user history:', error);
    }
}

async function loadUserStats() {
    try {
        const response = await fetch('/api/user/stats');
        if (response.ok) {
            const statsData = await response.json();
            window.userStatsData = statsData;
            console.log('User stats:', statsData);

            const statsContent = document.getElementById('statsContent');
            if (statsContent) {
                let favoriteStyle = 'Немає';
                if (statsData.popular_styles && statsData.popular_styles.length > 0) {
                    const sorted = [...statsData.popular_styles].sort(function (a, b) {
                        return b.count - a.count;
                    });
                    favoriteStyle = sorted[0].style || 'Немає';
                }

                statsContent.innerHTML = `
                    <div class="stats-grid">
                        <div class="stats-card">
                            <div class="stats-number">
                                ${statsData.total_analyses || 0}
                            </div>
                            <div class="stats-label">Всього аналізів</div>
                        </div>
                        <div class="stats-card">
                            <div class="stats-favorite">
                                ${favoriteStyle}
                            </div>
                            <div class="stats-label">Улюблений стиль</div>
                        </div>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.log('Failed to load user stats:', error);
    }
}

window.register = register;
window.login = login;
window.logout = logout;