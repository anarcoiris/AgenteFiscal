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

    // Inicializar Configs
    fetch('/api/config')
        .then(res => res.json())
        .then(data => {
            if (data.current_model) {
                modelSelect.value = data.current_model;
            }
        });

    settingsToggle.addEventListener('click', () => {
        settingsPanel.classList.toggle('hidden');
    });

    clearBtn.addEventListener('click', async () => {
        await fetch('/api/clear', { method: 'POST' });
        chatWindow.innerHTML = `<div class="message charlie">
                <div class="bubble">Historial limpiado. ¿En qué más te puedo ayudar?</div>
            </div>`;
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', function() {
        this.style.height = '50px';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Handling Enter to send
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

    async function sendMessage(textOverride = null, imageHTML = null) {
        const textStr = (typeof textOverride === 'string') ? textOverride : null;
        const text = textStr || chatInput.value.trim();
        if (!text && !imageHTML) return;

        // Añadir mensaje del usuario
        let userBubbleContent = '';
        if (imageHTML) userBubbleContent += imageHTML;
        if (text) userBubbleContent += `<p>${text}</p>`;

        addMessage('user', userBubbleContent, true);
        chatInput.value = '';
        chatInput.style.height = '50px';

        // Añadir loader de Charlie
        const loaderId = 'loader-' + Date.now();
        addMessage('charlie', `<div class="typing-indicator" id="${loaderId}"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`, true);

        try {
            const formData = {
                message: text,
                model: modelSelect.value
            };

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            
            // Eliminar loader
            const loaderEl = document.getElementById(loaderId);
            if(loaderEl) loaderEl.closest('.message').remove();

            if (data.success) {
                // Parse markdown
                const parsedText = marked.parse(data.answer);
                addMessage('charlie', parsedText, true);

                // Update badge
                confBadge.textContent = "Conf: " + (data.confidence || "N/A").toUpperCase();
                confBadge.className = 'badge ' + data.confidence;
            } else {
                addMessage('charlie', `<p style="color:#ff7b72;">Error: ${data.error}</p>`, true);
            }
        } catch (err) {
            const loaderEl = document.getElementById(loaderId);
            if(loaderEl) loaderEl.closest('.message').remove();
            addMessage('charlie', `<p style="color:#ff7b72;">Error de conexión.</p>`, true);
        }
    }

    function addMessage(sender, htmlContent, scrollToBottom = true) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = htmlContent;
        msgDiv.appendChild(bubble);
        chatWindow.appendChild(msgDiv);

        if (scrollToBottom) {
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    }

    // Drag & Drop / Paste Handling for OCR
    let dragCounter = 0;
    
    document.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dragCounter++;
        dropOverlay.classList.remove('hidden');
    });

    document.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter === 0) dropOverlay.classList.add('hidden');
    });

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
        if (!file.type.startsWith('image/')) {
            alert('Por favor, sube solo imágenes.');
            return;
        }

        // Preview local
        const reader = new FileReader();
        reader.onload = async (e) => {
            const imgHTML = `<img src="${e.target.result}" class="upload-image-preview" /><br>`;
            
            // UI Feedback
            const loaderId = 'loader-ocr-' + Date.now();
            addMessage('user', imgHTML, true);
            addMessage('charlie', `<div class="typing-indicator" id="${loaderId}"><div class="dot"></div><div class="dot"></div><div class="dot"></div><p style="margin-left:10px;font-size:0.8rem;">Leyendo imagen (OCR)...</p></div>`, true);

            // Subir a OCR
            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/ocr', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                
                const loaderEl = document.getElementById(loaderId);
                if(loaderEl) loaderEl.closest('.message').remove();

                if (data.success && data.text) {
                    // Auto-enviar la pregunta con el texto OCR
                    const ocrMsg = `[Texto extraído por OCR de la captura]:\n"${data.text}"\n\nAnaliza este texto fiscal.`;
                    
                    // Mostramos la respuesta directa en Charlie como confirmación de lectura, o lanzamos al chat
                    addMessage('charlie', `<p><i>✅ He leído el texto de la imagen:</i></p><blockquote style="border-left: 3px solid #58a6ff; padding-left:10px; margin: 10px 0; color:#8b949e;">${data.text.substring(0,200)}...</blockquote><p>Analizando...</p>`, true);
                    
                    // Ahora mandamos el texto de la imagen como query
                    await sendMessage(ocrMsg); 
                } else {
                    addMessage('charlie', `<p style="color:#d29922;">No pude leer texto claro en esa imagen.</p>`, true);
                }
            } catch (err) {
                const loaderEl = document.getElementById(loaderId);
                if(loaderEl) loaderEl.closest('.message').remove();
                addMessage('charlie', `<p style="color:#ff7b72;">Fallo al procesar la imagen (Servidor ocupado o error OCR).</p>`, true);
            }
        };
        reader.readAsDataURL(file);
    }
});
