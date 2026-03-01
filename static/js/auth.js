// Auth
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
            alertEl.innerHTML = '<div class="alert alert-error">Registration error</div>';
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
            alertEl.innerHTML = '<div class="alert alert-error">Login error</div>';
        }
    }
}

async function logout() {
    try {
        await fetch('/api/auth/logout', { method: 'POST' });
        updateUserInterface({ logged_in: false });
    } catch (error) {
        console.log('Logout error:', error);
    }
}

async function quickLogin(username, password) {
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (response.ok) {
            updateUserInterface({ logged_in: true, username: data.user.username });
        } else {
            alert('Помилка входу: ' + data.error);
        }
    } catch (error) {
        alert('Помилка: ' + error.message);
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

// User UI blocks
function updateUserInterface(userData) {
    const guestButtons = document.getElementById('guestButtons');
    const userButtons = document.getElementById('userButtons');
    const welcomeText = document.getElementById('welcomeText');
    const userBlocks = document.getElementById('userBlocks');
    const quickLoginBlock = document.getElementById('quickLogin');

    if (userData && userData.logged_in) {
        if (guestButtons) guestButtons.style.display = 'none';
        if (userButtons) userButtons.style.display = 'flex';
        if (welcomeText) welcomeText.textContent = `Вітаємо, ${userData.username}!`;
        if (quickLoginBlock) quickLoginBlock.style.display = 'none';
        if (userBlocks) userBlocks.style.display = 'flex';

        loadUserHistory();
        loadUserStats();
    } else {
        if (guestButtons) guestButtons.style.display = 'flex';
        if (userButtons) userButtons.style.display = 'none';
        if (userBlocks) userBlocks.style.display = 'none';
        if (quickLoginBlock) quickLoginBlock.style.display = 'flex';
    }
}

function showUserHistory() {
    loadUserHistory().then(function () {
        if (window.userHistoryData && window.userHistoryData.length > 0 && architecturalMap) {
            const content = window.userHistoryData.slice(0, 5).map(function (item) {
                return `
                    <div style="margin: 5px 0; padding: 5px; border-bottom: 1px solid #ddd;">
                        <strong>${item.predicted_style}</strong><br>
                        <small>${new Date(item.timestamp).toLocaleDateString('uk-UA')} - ${item.confidence}%</small>
                    </div>
                `;
            }).join('');

            L.popup()
                .setLatLng([49.8397, 24.0297])
                .setContent(`<div><h4>Analysis History</h4>${content}</div>`)
                .openOn(architecturalMap);
        }
    });
}

function showUserStats() {
    loadUserStats().then(function () {
        if (!architecturalMap) return;

        const content = `
            <div style="text-align: center;">
                <div style="margin: 10px 0;">
                    <div style="font-size: 24px; color: var(--teal); font-weight: bold;">${window.userStatsData?.total_analyses || 0}</div>
                    <div style="font-size: 12px;">Total Analyses</div>
                </div>
                <div style="margin: 10px 0;">
                    <div style="font-size: 18px; color: var(--teal); font-weight: bold;">${window.userStatsData?.favorite_style || '-'}</div>
                    <div style="font-size: 12px;">Favorite Style</div>
                </div>
            </div>
        `;

        L.popup()
            .setLatLng([46.4825, 30.7233])
            .setContent(`<div><h4>Statistics</h4>${content}</div>`)
            .openOn(architecturalMap);
    });
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
            lines[i] = '<div style="margin: 8px 0; padding-left: 15px;">' + lines[i] + '</div>';
        }
    }
    formattedText = lines.join('<br>');

    geminiContent.innerHTML = `
        <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; border-left: 4px solid var(--teal); font-size: 14px; line-height: 1.6; color: #fff;">
            ${formattedText}
        </div>
        <div style="margin-top: 10px; text-align: center; font-size: 12px; color: rgba(255,255,255,0.6);">
            Аналіз від Google Gemini Vision
        </div>
    `;
}

function updateHistoryBlock(historyData) {
    const historyContent = document.getElementById('historyContent');
    if (!historyContent || !historyData || !historyData.history) return;

    if (historyData.history.length === 0) {
        historyContent.innerHTML = '<p style="color: rgba(255,255,255,0.6); text-align: center;">No analysis history</p>';
        return;
    }

    const historyHTML = historyData.history.slice(0, 10).map(function (item) {
        const thumbnailSrc = item.image_thumbnail
            ? `data:image/jpeg;base64,${item.image_thumbnail}`
            : 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgdmlld0JveD0iMCAwIDE1MCAxNTAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTAiIGZpbGw9IiNmMGYwZjAiLz48dGV4dCB4PSI3NSIgeT0iNzUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIGltYWdlPC90ZXh0Pjwvc3ZnPg==';

        const confidence = Math.round((item.confidence || 0) * 100);
        const date = new Date(item.created_at).toLocaleDateString('uk-UA');

        return `
            <div style="display: flex; gap: 10px; padding: 10px; border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; margin-bottom: 10px; background: rgba(0,0,0,0.2);">
                <img src="${thumbnailSrc}"
                     style="width: 60px; height: 60px; object-fit: cover; border-radius: 6px; flex-shrink: 0;"
                     alt="">
                <div style="flex: 1; min-width: 0; color: #fff;">
                    <div style="font-weight: bold; color: var(--teal); margin-bottom: 4px; font-size: 14px;">
                        ${item.architectural_style || 'Unknown style'}
                    </div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 12px; margin-bottom: 4px;">
                        Confidence: ${confidence}%
                    </div>
                    <div style="color: #999; font-size: 11px;">
                        ${date}
                    </div>
                </div>
            </div>
        `;
    }).join('');

    historyContent.innerHTML = `
        <div style="max-height: 400px; overflow-y: auto;">
            ${historyHTML}
        </div>
        <div style="text-align: center; margin-top: 10px;">
            <small style="color: rgba(255,255,255,0.6);">Latest ${Math.min(historyData.history.length, 10)} analyses</small>
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
                    const sorted = statsData.popular_styles.sort(function (a, b) { return b.count - a.count; });
                    favoriteStyle = sorted[0].style || 'Немає';
                }

                statsContent.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div style="text-align: center; padding: 15px; background: rgba(255, 255, 255, 0.15); border-radius: 8px; backdrop-filter: blur(5px);">
                            <div style="font-size: 28px; color: var(--teal); font-weight: bold; margin-bottom: 5px;">
                                ${statsData.total_analyses || 0}
                            </div>
                            <div style="font-size: 12px; color: rgba(255, 255, 255, 0.8);">Всього аналізів</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background: rgba(255, 255, 255, 0.15); border-radius: 8px; backdrop-filter: blur(5px);">
                            <div style="font-size: 16px; color: var(--teal); font-weight: bold; margin-bottom: 5px;">
                                ${favoriteStyle}
                            </div>
                            <div style="font-size: 12px; color: rgba(255, 255, 255, 0.8);">Улюблений стиль</div>
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
window.quickLogin = quickLogin;