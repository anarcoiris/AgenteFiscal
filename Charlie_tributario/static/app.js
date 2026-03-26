document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatWindow = document.getElementById('chat-window');
    const settingsToggle = document.getElementById('settings-toggle');
    const settingsPanel = document.getElementById('settings-panel');
    const dropOverlay = document.getElementById('drop-overlay');
    const confBadge = document.getElementById('confidence-badge');
    const modelSelect = document.getElementById('model-select');
    const clearBtn = document.getElementById('clear-btn');
    const statusDot = document.getElementById('server-status');
    const contrastToast = document.getElementById('contrast-toast');
    const contrastToastDetail = document.getElementById('contrast-toast-detail');
    const contrastToastClose = document.getElementById('contrast-toast-close');
    const clearBadBtn = document.getElementById('clear-bad-btn');
    const cacheStatsLabel = document.getElementById('cache-stats');

    // ── Config Init ─────────────────────────────────────────────────────────
    fetch('/api/config')
        .then(res => res.json())
        .then(data => {
            if (data.current_model) {
                // Intentar seleccionar el modelo actual
                const opt = modelSelect.querySelector(`option[value="${data.current_model}"]`);
                if (opt) modelSelect.value = data.current_model;
            }
            if (!data.openai_available) {
                // Si OpenAI no disponible, marcar las opciones
                modelSelect.querySelectorAll('option[value^="gpt"]').forEach(opt => {
                    opt.textContent += ' (sin API key)';
                });
            }
        });

    // Cargar stats de caché
    refreshCacheStats();

    // ── Health Check (cada 15s) ─────────────────────────────────────────────
    async function checkHealth() {
        try {
            const res = await fetch('/api/health', { signal: AbortSignal.timeout(3000) });
            if (res.ok) {
                statusDot.className = 'status-dot online';
                statusDot.title = 'Servidor conectado';
            } else { throw new Error(); }
        } catch {
            statusDot.className = 'status-dot offline';
            statusDot.title = 'Servidor desconectado';
        }
    }
    checkHealth();
    setInterval(checkHealth, 15000);

    // ── Cache Stats ─────────────────────────────────────────────────────────
    async function refreshCacheStats() {
        try {
            const res = await fetch('/api/cache/stats');
            const data = await res.json();
            if (data.success) {
                cacheStatsLabel.textContent = `📦 ${data.total} en caché (👍${data.good} 👎${data.bad})`;
            }
        } catch { /* silent */ }
    }

    clearBadBtn.addEventListener('click', async () => {
        const res = await fetch('/api/cache/clear-bad', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            clearBadBtn.textContent = `✅ ${data.removed} eliminadas`;
            setTimeout(() => clearBadBtn.textContent = '🧹 Limpiar malas', 2000);
            refreshCacheStats();
        }
    });

    // ── UI Toggles ──────────────────────────────────────────────────────────
    settingsToggle.addEventListener('click', () => {
        settingsPanel.classList.toggle('hidden');
        if (!settingsPanel.classList.contains('hidden')) refreshCacheStats();
    });

    clearBtn.addEventListener('click', async () => {
        await fetch('/api/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ clear_cache: false })
        });
        chatWindow.innerHTML = `<div class="message charlie"><div class="bubble">Historial limpiado. ¿En qué más te puedo ayudar?</div></div>`;
    });

    // ── Toast de contraste ──────────────────────────────────────────────────
    contrastToastClose.addEventListener('click', () => {
        contrastToast.classList.add('hidden');
    });

    function showContrastToast(reason) {
        contrastToastDetail.textContent = reason;
        contrastToast.classList.remove('hidden');
        // Auto-ocultar después de 15s
        setTimeout(() => contrastToast.classList.add('hidden'), 15000);
    }

    // ── Polling de contraste ────────────────────────────────────────────────
    function pollContrast(contrastId) {
        if (!contrastId) return;

        const poll = setInterval(async () => {
            try {
                const res = await fetch(`/api/contrast/${contrastId}`);
                const data = await res.json();

                if (data.status === 'done') {
                    clearInterval(poll);
                    if (data.discrepancy && data.discrepancy.differs) {
                        showContrastToast(data.discrepancy.reason);
                    }
                } else if (data.status === 'error') {
                    clearInterval(poll);
                }
            } catch {
                clearInterval(poll);
            }
        }, 3000); // Poll cada 3s
    }

    // ── Textarea auto-resize + Enter ────────────────────────────────────────
    chatInput.addEventListener('input', function() {
        this.style.height = '50px';
        this.style.height = (this.scrollHeight) + 'px';
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', (e) => {
        e.preventDefault();
        sendMessage();
    });

    // ── Core: Send Message ──────────────────────────────────────────────────
    let lastQuestion = ''; // Para feedback

    async function sendMessage(textOverride = null) {
        const textStr = (typeof textOverride === 'string') ? textOverride : null;
        const text = textStr || chatInput.value.trim();
        if (!text) return;

        lastQuestion = text;
        addMessage('user', `<p>${escapeHtml(text)}</p>`);
        chatInput.value = '';
        chatInput.style.height = '50px';

        const loaderId = 'loader-' + Date.now();
        addMessage('charlie', `<div class="typing-indicator" id="${loaderId}"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, model: modelSelect.value })
            });

            const data = await response.json();
            removeLoader(loaderId);

            if (data.success) {
                const answerHTML = marked.parse(data.answer || '');
                const metaHTML = buildMetaHTML(data, text);
                const sourcesHTML = buildSourcesHTML(data.sources);
                const suggestionsHTML = buildSuggestionsHTML(data.sources);

                addMessage('charlie', answerHTML + metaHTML + sourcesHTML + suggestionsHTML);

                // Update confidence badge
                confBadge.textContent = (data.confidence || 'N/A').toUpperCase();
                confBadge.className = 'badge ' + (data.confidence || '');

                // Iniciar polling de contraste si aplica
                if (data.contrast_id) {
                    pollContrast(data.contrast_id);
                }

                // Refrescar stats
                refreshCacheStats();
            } else {
                addMessage('charlie', `<p style="color:#ff7b72;">Error: ${data.error}</p>`);
            }
        } catch (err) {
            removeLoader(loaderId);
            addMessage('charlie', `<p style="color:#ff7b72;">Error de conexión al servidor.</p>`);
        }
    }

    // ── Build: Response Metadata (model tag + timer + copy + feedback) ──────
    function buildMetaHTML(data, question) {
        const isFallback = (data.model_used || '').includes('Fallback');
        const isOpenAI = (data.model_used || '').includes('OpenAI');
        const modelClass = isFallback ? 'model-tag fallback' : (isOpenAI ? 'model-tag openai' : 'model-tag');
        const elapsed = data.elapsed_seconds ? `⏱️ ${data.elapsed_seconds}s` : '';
        const cachedTag = data.cached ? '<span class="cached-tag">⚡ CACHÉ</span>' : '';
        const copyId = 'copy-' + Date.now();
        const fbId = 'fb-' + Date.now();
        const escapedQ = escapeHtml(question).replace(/"/g, '&quot;');

        return `
        <div class="response-meta">
            <span class="${modelClass}">${data.model_used || 'Local'}</span>
            <span>${elapsed}</span>
            ${cachedTag}
            <button class="copy-btn" id="${copyId}" title="Copiar respuesta">📋 Copiar</button>
            <span class="feedback-buttons" id="${fbId}" data-question="${escapedQ}">
                <button class="fb-btn fb-good" data-quality="good" title="Buena respuesta">👍</button>
                <button class="fb-btn fb-bad" data-quality="bad" title="Mala respuesta">👎</button>
            </span>
        </div>`;
    }

    // ── Build: Source References ─────────────────────────────────────────────
    function buildSourcesHTML(sources) {
        if (!sources || sources.length === 0) return '';

        // De-duplicate sources
        const seen = new Set();
        const unique = [];
        for (const s of sources) {
            const key = `${s.fuente}|${s.paginas}`;
            if (!seen.has(key)) { seen.add(key); unique.push(s); }
        }

        let items = '';
        for (const s of unique) {
            const relevance = Math.max(0, Math.min(100, Math.round((1 - s.distancia) * 100)));
            const barColor = relevance > 70 ? '#3fb950' : relevance > 40 ? '#d29922' : '#da3633';
            items += `
            <li class="source-item">
                <span class="source-pill">📄 ${s.fuente}</span>
                <span>${s.paginas || ''}</span>
                <span class="dist-bar"><span class="dist-bar-fill" style="width:${relevance}%;background:${barColor}"></span></span>
                <span>${relevance}%</span>
            </li>`;
        }

        return `
        <details class="sources-block">
            <summary>📚 ${unique.length} fuente(s) consultada(s)</summary>
            <ul class="source-list">${items}</ul>
        </details>`;
    }

    // ── Build: Suggested Questions ──────────────────────────────────────────
    function buildSuggestionsHTML(sources) {
        if (!sources || sources.length === 0) return '';

        const suggestions = [
            '¿Qué deducciones puedo aplicar en mi declaración?',
            '¿Cuándo tengo obligación de declarar?',
            '¿Cómo se compensan las pérdidas patrimoniales?'
        ];

        let btns = '';
        for (const q of suggestions) {
            btns += `<button class="suggestion-btn" data-question="${escapeHtml(q)}">${q}</button>`;
        }

        return `<div class="suggestions">${btns}</div>`;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    function addMessage(sender, htmlContent) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = htmlContent;
        msgDiv.appendChild(bubble);
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function removeLoader(loaderId) {
        const el = document.getElementById(loaderId);
        if (el) el.closest('.message').remove();
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ── Delegated Events (Copy + Suggestions + Feedback) ────────────────────
    chatWindow.addEventListener('click', async (e) => {
        // Copy button
        if (e.target.classList.contains('copy-btn')) {
            const bubble = e.target.closest('.bubble');
            if (bubble) {
                const clone = bubble.cloneNode(true);
                clone.querySelectorAll('.response-meta, .sources-block, .suggestions').forEach(el => el.remove());
                navigator.clipboard.writeText(clone.textContent.trim());
                e.target.textContent = '✅ Copiado';
                setTimeout(() => e.target.textContent = '📋 Copiar', 2000);
            }
        }

        // Feedback buttons (👍/👎)
        if (e.target.classList.contains('fb-btn')) {
            const quality = e.target.dataset.quality;
            const container = e.target.closest('.feedback-buttons');
            const question = container?.dataset.question;

            if (question) {
                try {
                    await fetch('/api/feedback', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question, quality })
                    });

                    // Feedback visual
                    container.querySelectorAll('.fb-btn').forEach(btn => {
                        btn.disabled = true;
                        btn.classList.remove('fb-active');
                    });
                    e.target.classList.add('fb-active');
                    e.target.classList.add(quality === 'good' ? 'fb-active-good' : 'fb-active-bad');

                    refreshCacheStats();
                } catch { /* silent */ }
            }
        }

        // Suggestion buttons
        if (e.target.classList.contains('suggestion-btn')) {
            const question = e.target.dataset.question;
            if (question) {
                chatWindow.querySelectorAll('.suggestions').forEach(el => el.remove());
                sendMessage(question);
            }
        }
    });

    // ── Drag & Drop / Paste for OCR ─────────────────────────────────────────
    let dragCounter = 0;

    document.addEventListener('dragenter', (e) => { e.preventDefault(); dragCounter++; dropOverlay.classList.remove('hidden'); });
    document.addEventListener('dragleave', (e) => { e.preventDefault(); dragCounter--; if (dragCounter === 0) dropOverlay.classList.add('hidden'); });
    document.addEventListener('dragover', (e) => { e.preventDefault(); });

    document.addEventListener('drop', (e) => {
        e.preventDefault();
        dragCounter = 0;
        dropOverlay.classList.add('hidden');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });

    document.addEventListener('paste', (e) => {
        if (e.clipboardData.files && e.clipboardData.files.length > 0) {
            const file = e.clipboardData.files[0];
            if (file.type.startsWith('image/')) {
                e.preventDefault();
                handleImageUpload(file);
            }
        }
    });

    async function handleImageUpload(file) {
        if (!file.type.startsWith('image/')) { alert('Solo imágenes.'); return; }

        const reader = new FileReader();
        reader.onload = async (ev) => {
            const imgHTML = `<img src="${ev.target.result}" class="upload-image-preview" />`;
            addMessage('user', imgHTML);

            const loaderId = 'loader-ocr-' + Date.now();
            addMessage('charlie', `<div class="typing-indicator" id="${loaderId}"><div class="dot"></div><div class="dot"></div><div class="dot"></div><span style="margin-left:8px;font-size:0.8rem">Leyendo imagen (OCR)...</span></div>`);

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/ocr', { method: 'POST', body: formData });
                const data = await res.json();
                removeLoader(loaderId);

                if (data.success && data.text) {
                    addMessage('charlie', `<p><i>✅ Texto extraído:</i></p><blockquote>${escapeHtml(data.text.substring(0, 300))}${data.text.length > 300 ? '...' : ''}</blockquote><p>Analizando...</p>`);
                    await sendMessage(`[Texto extraído por OCR]:\n"${data.text}"\n\nAnaliza este texto fiscal.`);
                } else {
                    addMessage('charlie', `<p style="color:#d29922;">No pude leer texto claro en esa imagen.</p>`);
                }
            } catch {
                removeLoader(loaderId);
                addMessage('charlie', `<p style="color:#ff7b72;">Error al procesar la imagen.</p>`);
            }
        };
        reader.readAsDataURL(file);
    }
});
