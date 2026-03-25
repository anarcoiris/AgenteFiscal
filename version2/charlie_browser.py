"""
charlie_browser.py — Controlador Playwright propio para Charlie.

v3 — Nuevos métodos respecto a v2:
  - NUEVO: extract(selector) — lee el texto de un elemento para el agente
  - NUEVO: dismiss_popups() — descarta automáticamente banners de cookies y diálogos
           comunes antes de cada paso del agente
  - MANTENIDO: navigate con networkidle+fallback, click con scroll, fill con triple-click,
               type_char_by_char, highlight con locator, get_text_content priorizado,
               get_interactive_elements con priorización crítica, check_login_status robusto
  - CORREGIDO: nombre de modelo sin guion en comentarios (qwen2.5vl:3b)
"""
import asyncio
import base64
from typing import Callable, Awaitable

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


# Selectores de botones de cookies/popups conocidos (orden de prioridad)
_POPUP_SELECTORS = [
    # Genéricos
    "button:has-text('Aceptar')",
    "button:has-text('Aceptar todo')",
    "button:has-text('Accept all')",
    "button:has-text('Accept')",
    "button:has-text('Agree')",
    "button:has-text('I agree')",
    "button:has-text('Got it')",
    "button:has-text('OK')",
    "button:has-text('Entendido')",
    # IDs comunes de GDPR
    "#onetrust-accept-btn-handler",
    "#accept-all-cookies",
    ".cookie-consent-accept",
    "[data-testid='accept-cookies']",
    "[aria-label='Accept cookies']",
    # LinkedIn específico
    "button.artdeco-global-alert__dismiss",
    "[data-control-name='accept-cookies']",
    # InfoJobs
    "#didomi-notice-agree-button",
    # Tecnoempleo
    ".cc-btn.cc-allow",
]


class CharlieBrowser:
    """
    Navegador propio de Charlie.
    Controla Chrome via Playwright directamente y emite screenshots
    por WebSocket a 2-3 FPS sin conflictos de CDP.
    """

    def __init__(
        self,
        ws_send: Callable[[str, object], Awaitable[None]],
        chrome_path: str,
        user_data_dir: str,
        headless: bool = False,
        fps: float = 2.0,
        jpeg_quality: int = 60,
        viewport: dict = None,
    ):
        self._send        = ws_send
        self._chrome_path = chrome_path
        self._user_data   = user_data_dir
        self._headless    = headless
        self._fps         = fps
        self._quality     = jpeg_quality
        self._viewport    = viewport or {"width": 1280, "height": 720}

        self._pw:      object         = None
        self._browser: Browser        = None
        self._context: BrowserContext = None
        self.page:     Page           = None

        self._screenshot_task: asyncio.Task = None
        self._stop_event = asyncio.Event()
        self._popup_dismissed_urls: set[str] = set()  # Evitar re-dismiss en la misma URL

    # ── Ciclo de vida ──────────────────────────────────────────────────────────

    async def start(self):
        self._pw = await async_playwright().start()
        try:
            self._context = await self._pw.chromium.launch_persistent_context(
                user_data_dir       = self._user_data,
                executable_path     = self._chrome_path,
                headless            = self._headless,
                viewport            = self._viewport,
                java_script_enabled = True,
                args=[
                    "--disable-gpu-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            )
        except Exception as e:
            err = str(e)
            if "Target page, context or browser has been closed" in err or "exitCode=21" in err:
                await self._emit(
                    "log",
                    "❌ ERROR: Perfil de Chrome bloqueado. "
                    "Cierra Chrome o usa una ruta USER_DATA diferente.",
                )
            raise

        self._browser = None
        self.page = (
            self._context.pages[0]
            if self._context.pages
            else await self._context.new_page()
        )
        await asyncio.sleep(1.0)
        self._stop_event.clear()
        self._screenshot_task = asyncio.create_task(self._screenshot_loop())
        await self._emit("log", "🌐 Navegador Charlie v3 iniciado.")

    async def stop(self):
        self._stop_event.set()
        if self._screenshot_task:
            self._screenshot_task.cancel()
            try:
                await self._screenshot_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self._context.close()
        except Exception:
            pass
        try:
            await self._pw.stop()
        except Exception:
            pass

    # ── Acciones del navegador ─────────────────────────────────────────────────

    async def navigate(self, url: str):
        """Navega a una URL. Intenta networkidle; fallback a domcontentloaded."""
        await self._emit("action", {"type": "navigate", "url": url})
        try:
            await self.page.goto(url, wait_until="networkidle", timeout=15_000)
        except Exception:
            try:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                await asyncio.sleep(2.0)
            except Exception as e:
                await self._emit("log", f"⚠️ navigate falló {url}: {e}")
                raise

    async def click(self, selector: str, description: str = ""):
        """Click con scroll previo al elemento (evita 'element not visible')."""
        await self._highlight(selector)
        await self._emit("action", {"type": "click", "selector": selector, "desc": description})
        locator = self.page.locator(selector).first
        await locator.scroll_into_view_if_needed(timeout=5_000)
        await locator.click(timeout=10_000)

    async def click_at(self, x: int, y: int):
        try:
            await self.page.mouse.click(x, y)
            await self._emit("action", {"type": "click", "x": x, "y": y, "desc": "Coordenadas"})
        except Exception as e:
            await self._emit("log", f"⚠️ click_at ({x},{y}): {e}")
            raise

    async def fill(self, selector: str, value: str, description: str = ""):
        """Rellena un campo, limpiando el valor previo con triple-click."""
        await self._highlight(selector)
        await self._emit("action", {"type": "fill", "selector": selector, "value": value, "desc": description})
        locator = self.page.locator(selector).first
        await locator.scroll_into_view_if_needed(timeout=5_000)
        await locator.click(click_count=3, timeout=5_000)
        await self.page.fill(selector, value, timeout=10_000)

    async def type_char_by_char(self, selector: str, value: str, delay_ms: int = 40):
        """Escribe carácter a carácter (para campos que rechazan fill sintético)."""
        await self._highlight(selector)
        await self._emit("action", {"type": "type", "selector": selector, "value": value})
        locator = self.page.locator(selector).first
        await locator.scroll_into_view_if_needed(timeout=5_000)
        await locator.click(click_count=3, timeout=5_000)
        await locator.press("Backspace")
        await self.page.type(selector, value, delay=delay_ms)

    async def press(self, key: str):
        await self.page.keyboard.press(key)

    async def scroll_to_element(self, selector: str):
        try:
            await self.page.locator(selector).first.scroll_into_view_if_needed(timeout=5_000)
        except Exception:
            pass

    async def wait_for_selector(self, selector: str, timeout: int = 10_000):
        return await self.page.wait_for_selector(selector, timeout=timeout)

    async def wait_for_navigation(self, timeout: int = 10_000) -> bool:
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=timeout)
            return True
        except Exception:
            return False

    async def extract(self, selector: str) -> str:
        """
        NUEVO v3: Lee el texto visible de un elemento dado su selector CSS.
        El agente usa esto para extraer datos sin necesidad de click (títulos,
        precios, primeros resultados de búsqueda, etc.).
        """
        try:
            text = await self.page.locator(selector).first.inner_text(timeout=5_000)
            return text.strip()[:500]
        except Exception:
            # Fallback: evaluate en el DOM
            try:
                text = await self.page.evaluate(
                    f"""() => {{
                        const el = document.querySelector({json_str(selector)});
                        return el ? (el.innerText || el.textContent || '').trim() : '';
                    }}"""
                )
                return str(text).strip()[:500]
            except Exception:
                return ""

    async def get_page_map(self) -> str:
        """
        NUEVO v4: Devuelve un mapa semántico de la página con ubicación de elementos.

        En vez de una lista plana de interactivos, el mapa agrupa los elementos
        por su ZONA en el DOM (cabecera, barra-búsqueda, resultados, modal, pie)
        y añade información de posición relativa (arriba/centro/abajo, izq/centro/der)
        y orden dentro de su sección.

        Esto permite al agente razonar sobre el layout:
          "El botón Easy Apply está en la ZONA modal-central"
          "El resultado nº1 de la búsqueda está en ZONA resultados"
          "El campo de salario está en ZONA formulario"

        Formato de salida (una entrada por línea):
          [ZONA: resultados | pos:1] a "Senior Python Dev – Empresa X" @ (400,320)
          [ZONA: resultados | pos:2] a "Backend Engineer – Startup Y" @ (400,450)
          [ZONA: modal | pos:1] button "Solicitud sencilla" @ (640,400) 🌟ACCIÓN_CLAVE
          [ZONA: formulario | pos:1] input[salary] "" @ (500,300) 📝CAMPO_FORM
        """
        try:
            result = await self.page.evaluate("""() => {
                const vw = window.innerWidth;
                const vh = window.innerHeight;

                // ── Determinar la zona semántica de un elemento ────────────────
                function getZone(el) {
                    // Buscar el ancestro semántico más cercano
                    const zoneMap = [
                        // Modales y diálogos (prioridad máxima)
                        { sel: '[role="dialog"], [role="alertdialog"], .modal, .dialog, .artdeco-modal',  zone: 'modal' },
                        // Formularios
                        { sel: 'form, [role="form"]',                                                      zone: 'formulario' },
                        // Cabecera / nav
                        { sel: 'header, nav, [role="banner"], [role="navigation"], .global-nav',           zone: 'cabecera' },
                        // Barra de búsqueda
                        { sel: '.search-bar, .search-form, [role="search"], form[action*="search"]',       zone: 'barra-busqueda' },
                        // Resultados de búsqueda
                        { sel: '.jobs-search-results-list, .search-results, [role="list"], ol, ul.results', zone: 'resultados' },
                        // Detalle de oferta / contenido principal
                        { sel: '.job-view-layout, [role="main"], main, article',                           zone: 'contenido-principal' },
                        // Pie
                        { sel: 'footer, [role="contentinfo"]',                                             zone: 'pie' },
                    ];

                    for (const { sel, zone } of zoneMap) {
                        if (el.closest(sel)) return zone;
                    }

                    // Fallback por posición en el viewport
                    const rect = el.getBoundingClientRect();
                    const cy   = rect.top + rect.height / 2;
                    if (cy < vh * 0.15) return 'cabecera';
                    if (cy > vh * 0.85) return 'pie';
                    return 'cuerpo';
                }

                // ── Posición relativa dentro de la zona (para ordenar) ─────────
                function getRelPos(rect) {
                    const cx = rect.left + rect.width / 2;
                    const cy = rect.top  + rect.height / 2;
                    const hPos = cx < vw * 0.33 ? 'izq' : cx < vw * 0.66 ? 'centro' : 'der';
                    const vPos = cy < vh * 0.33 ? 'arriba' : cy < vh * 0.66 ? 'medio' : 'abajo';
                    return `${vPos}-${hPos}`;
                }

                // ── Visibilidad ────────────────────────────────────────────────
                function isVisible(el) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return false;
                    const style = window.getComputedStyle(el);
                    if (style.visibility === 'hidden' || style.display === 'none'
                        || parseFloat(style.opacity) < 0.1) return false;
                    const cx = rect.left + rect.width / 2;
                    const cy = rect.top  + rect.height / 2;
                    if (cx < 0 || cy < 0 || cx > vw || cy > vh) return false;
                    return true;
                }

                // ── Etiqueta legible de un elemento ───────────────────────────
                function getLabel(el) {
                    return (
                        el.innerText ||
                        el.getAttribute('aria-label') ||
                        el.getAttribute('placeholder') ||
                        el.getAttribute('value') ||
                        el.getAttribute('title') ||
                        el.getAttribute('name') ||
                        ''
                    ).trim().replace(/\\s+/g, ' ').slice(0, 60);
                }

                // ── Criticidad ────────────────────────────────────────────────
                function getCriticality(el, label) {
                    const low = label.toLowerCase();
                    const aria = (el.getAttribute('aria-label') || '').toLowerCase();
                    const name = (el.getAttribute('name') || '').toLowerCase();
                    const ph   = (el.getAttribute('placeholder') || '').toLowerCase();

                    if (low.includes('easy apply') || low.includes('solicitud sencilla') ||
                        low.includes('inscribirme') || low.includes('apply now') ||
                        aria.includes('easy apply')) return '🌟ACCIÓN_CLAVE';

                    if (low.includes('buscar') || low.includes('search') ||
                        name === 'q' || ph.includes('buscar') || ph.includes('search'))
                        return '🌟BUSCAR_AQUÍ';

                    if (low.includes('continuar') || low.includes('siguiente') ||
                        low.includes('enviar') || low.includes('confirmar') ||
                        low.includes('submit') || low.includes('next'))
                        return '🔶ACCIÓN';

                    if (name.includes('salari') || ph.includes('salari') ||
                        low.includes('teléfono') || name.includes('phone') ||
                        low.includes('inglés'))
                        return '📝CAMPO_FORM';

                    return '';
                }

                // ── Recopilar elementos interactivos ──────────────────────────
                const interactives = document.querySelectorAll(
                    'button, a[href], input:not([type="hidden"]):not([type="file"]), ' +
                    'select, textarea, [role="button"], [role="link"], [role="combobox"], ' +
                    '[contenteditable="true"]'
                );

                const byZone = {};
                interactives.forEach((el, globalIdx) => {
                    if (!isVisible(el)) return;
                    const rect     = el.getBoundingClientRect();
                    const tag      = el.tagName.toLowerCase();
                    const label    = getLabel(el);
                    const zone     = getZone(el);
                    const relPos   = getRelPos(rect);
                    const crit     = getCriticality(el, label);
                    const cx       = Math.round(rect.left + rect.width / 2);
                    const cy       = Math.round(rect.top  + rect.height / 2);
                    const name     = el.getAttribute('name') ? `[${el.getAttribute('name')}]` : '';

                    if (!byZone[zone]) byZone[zone] = [];
                    byZone[zone].push({
                        tag, name, label, cx, cy, relPos, crit, globalIdx,
                        // Ordenar dentro de la zona por posición vertical
                        sortKey: Math.round(rect.top)
                    });
                });

                // ── Serializar ────────────────────────────────────────────────
                const lines = [];
                // Orden de zonas: modal primero (más urgente), luego el resto
                const zoneOrder = ['modal','formulario','barra-busqueda','resultados',
                                   'contenido-principal','cuerpo','cabecera','pie'];
                const sortedZones = Object.keys(byZone).sort((a, b) => {
                    const ai = zoneOrder.indexOf(a); const bi = zoneOrder.indexOf(b);
                    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
                });

                for (const zone of sortedZones) {
                    const items = byZone[zone].sort((a, b) => a.sortKey - b.sortKey);
                    items.forEach((item, posInZone) => {
                        const critStr = item.crit ? ` ${item.crit}` : '';
                        lines.push(
                            `[ZONA:${zone} | pos:${posInZone + 1} | ${item.relPos}]` +
                            ` [INDEX:${item.globalIdx}] ${item.tag}${item.name}` +
                            ` "${item.label}" @ (${item.cx},${item.cy})${critStr}`
                        );
                    });
                }

                return lines.join('\\n');
            }""")
            return result or "(página sin elementos interactivos visibles)"
        except Exception as e:
            return f"Error generando mapa: {str(e)}"

    async def dismiss_popups(self):
        """
        NUEVO v3: Intenta descartar automáticamente banners de cookies y
        diálogos comunes antes de cada paso del agente.
        Solo actúa una vez por URL para no desperdiciar tiempo.
        """
        current_url = self.page.url
        # Solo intentar una vez por URL
        if current_url in self._popup_dismissed_urls:
            return
        dismissed = False
        for sel in _POPUP_SELECTORS:
            try:
                locator = self.page.locator(sel).first
                # is_visible tiene timeout muy corto para no bloquear
                if await locator.is_visible(timeout=400):
                    await locator.click(timeout=2_000)
                    await self._emit("log", f"🍪 Popup/cookie descartado: {sel}")
                    dismissed = True
                    break  # Un click suele ser suficiente
            except Exception:
                continue
        if dismissed:
            await asyncio.sleep(0.5)
        self._popup_dismissed_urls.add(current_url)

    async def get_url(self) -> str:
        return self.page.url

    async def get_page_title(self) -> str:
        try:
            return await self.page.title()
        except Exception:
            return ""

    async def get_text_content(self) -> str:
        """Extrae texto visible priorizando la zona de contenido principal."""
        try:
            text = await self.page.evaluate("""() => {
                const mainSelectors = [
                    'main', 'article', '[role="main"]',
                    '.jobs-search-results-list',
                    '.job-view-layout',
                    '.oferta-empleo',
                    '.ij-OfferList',
                ];
                let root = null;
                for (const sel of mainSelectors) {
                    root = document.querySelector(sel);
                    if (root) break;
                }
                if (!root) root = document.body;
                const clone = root.cloneNode(true);
                clone.querySelectorAll(
                    'script, style, code, nav, header, footer, ' +
                    '[aria-hidden="true"], [role="banner"], [role="navigation"]'
                ).forEach(e => e.remove());
                const raw = (clone.innerText || clone.textContent || '')
                    .replace(/[ \\t]+/g, ' ')
                    .replace(/\\n{3,}/g, '\\n\\n')
                    .trim();
                return raw.slice(0, 2500);
            }""")
            return text
        except Exception:
            return ""

    async def get_interactive_elements(self) -> str:
        """Extrae elementos interactivos con coordenadas, priorizando los críticos."""
        try:
            elements = await self.page.evaluate("""() => {
                const results = [];
                const interactives = document.querySelectorAll(
                    'button, a, input, select, ' +
                    '[role="button"], [role="link"], [role="combobox"], ' +
                    '[role="searchbox"], [contenteditable="true"], textarea'
                );

                function isVisible(el) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) return false;
                    const style = window.getComputedStyle(el);
                    if (style.visibility === 'hidden' || style.display === 'none' || parseFloat(style.opacity) < 0.1) return false;
                    const cx = rect.left + rect.width / 2;
                    const cy = rect.top + rect.height / 2;
                    if (cx < 0 || cy < 0 || cx > window.innerWidth || cy > window.innerHeight) return false;
                    const topEl = document.elementFromPoint(cx, cy);
                    if (!topEl) return false;
                    return el.contains(topEl) || topEl.contains(el) || el === topEl;
                }

                interactives.forEach((el, index) => {
                    if (!isVisible(el)) return;
                    const rect        = el.getBoundingClientRect();
                    const tag         = el.tagName.toLowerCase();
                    const inputType   = el.getAttribute('type') || '';
                    const name        = el.getAttribute('name') || '';
                    const ariaLabel   = (el.getAttribute('aria-label') || '').trim();
                    const placeholder = (el.getAttribute('placeholder') || '').trim();
                    const dataAuto    = el.getAttribute('data-automation') || '';
                    const label       = (el.innerText || ariaLabel || placeholder || el.getAttribute('value') || el.getAttribute('title') || '').trim().replace(/\\s+/g, ' ');
                    const cx = Math.round(rect.left + rect.width / 2);
                    const cy = Math.round(rect.top + rect.height / 2);

                    if (tag === 'input' && (inputType === 'hidden' || inputType === 'file') && !label && !name) return;
                    if (!label && !name && !placeholder && !ariaLabel && tag !== 'input' && tag !== 'select') return;

                    let meta = '';
                    if (name)        meta += ` name="${name}"`;
                    if (placeholder) meta += ` placeholder="${placeholder.slice(0, 25)}"`;
                    if (ariaLabel && ariaLabel !== label) meta += ` aria-label="${ariaLabel.slice(0, 25)}"`;
                    if (dataAuto)    meta += ` data-auto="${dataAuto.slice(0, 20)}"`;

                    let item = `[INDEX:${index}] ${tag}${meta} "${label.slice(0, 50)}" @ (${cx},${cy})`;

                    const low      = label.toLowerCase();
                    const lowAria  = ariaLabel.toLowerCase();
                    const lowName  = name.toLowerCase();
                    const lowPlace = placeholder.toLowerCase();

                    const isSearchInput = (
                        tag === 'input' || tag === 'textarea' ||
                        el.getAttribute('role') === 'combobox' ||
                        el.getAttribute('role') === 'searchbox'
                    ) && (
                        low.includes('buscar') || low.includes('search') ||
                        lowName === 'q' || lowName === 'search' || lowName === 'keywords' ||
                        lowAria.includes('buscar') || lowAria.includes('search') ||
                        lowPlace.includes('buscar') || lowPlace.includes('search') ||
                        lowPlace.includes('cargo') || lowPlace.includes('puesto')
                    );

                    const isCriticalAction = (
                        low.includes('solicitud sencilla') || low.includes('easy apply') ||
                        low.includes('inscribirme') || low.includes('inscribirse') ||
                        low.includes('continuar') || low.includes('continue') ||
                        low.includes('siguiente') || low.includes('next') ||
                        low.includes('enviar') || low.includes('submit') ||
                        low.includes('aplicar') || low.includes('apply') ||
                        low.includes('confirmar') || low.includes('confirm') ||
                        low.includes('enviar candidatura') ||
                        lowAria.includes('easy apply') || lowAria.includes('solicitud sencilla')
                    );

                    const isSalaryOrForm = (
                        low.includes('salario') || low.includes('salary') ||
                        lowName.includes('salario') || lowName.includes('salary') ||
                        lowPlace.includes('salario') || lowPlace.includes('salary') ||
                        low.includes('teléfono') || lowName.includes('phone') ||
                        low.includes('inglés') || lowName.includes('english')
                    );

                    if (isSearchInput)     item = `🌟 [BUSCAR_AQUÍ] ${item}`;
                    else if (isCriticalAction) item = `🌟 [ACCIÓN_CLAVE] ${item}`;
                    else if (isSalaryOrForm)   item = `📝 [CAMPO_FORM] ${item}`;

                    results.push(item);
                });

                const critical = results.filter(r => r.startsWith('🌟'));
                const form     = results.filter(r => r.startsWith('📝'));
                const rest     = results.filter(r => !r.startsWith('🌟') && !r.startsWith('📝'));
                return [...critical, ...form, ...rest].slice(0, 80).join(' ||| ');
            }""")
            return elements
        except Exception as e:
            return f"Error extrayendo elementos: {str(e)}"

    async def check_login_status(self) -> str:
        try:
            status = await self.page.evaluate("""() => {
                const url = window.location.href;
                if (url.includes('linkedin.com')) {
                    const loggedIn = !!(
                        document.querySelector('.profile-rail-card') ||
                        document.querySelector('#global-nav-typeahead') ||
                        document.querySelector('.feed-identity-module') ||
                        document.querySelector('[data-control-name="identity_welcome_message"]') ||
                        document.querySelector('.global-nav__me-photo') ||
                        document.querySelector('[aria-label="Foto del perfil"]') ||
                        document.querySelector('a[href*="/in/"][class*="global-nav"]')
                    );
                    const onLoginPage = url.includes('/login') || url.includes('/checkpoint') || !!document.querySelector('#username, #password');
                    if (onLoginPage) return 'NO_LOGUEADO (página login)';
                    return loggedIn ? 'LOGUEADO' : 'ESTADO_DESCONOCIDO';
                }
                if (url.includes('tecnoempleo.com')) {
                    const loggedIn = !!(
                        document.querySelector('a[href*="salir"]') ||
                        document.querySelector('.menu-usuario') ||
                        document.querySelector('[href*="mi-perfil"]') ||
                        document.querySelector('#menu-usuario')
                    );
                    return loggedIn ? 'LOGUEADO' : 'NO_LOGUEADO';
                }
                if (url.includes('infojobs.net')) {
                    const loggedIn = !!(
                        document.querySelector('a[href*="logout"]') ||
                        document.querySelector('.ij-Component-header-user') ||
                        document.querySelector('[data-testid="userMenu"]') ||
                        document.querySelector('.ij-HeaderUser')
                    );
                    return loggedIn ? 'LOGUEADO' : 'NO_LOGUEADO';
                }
                return 'DESCONOCIDO';
            }""")
            return status
        except Exception:
            return "ERROR_AL_COMPROBAR"

    async def take_screenshot_b64(self) -> str:
        try:
            data = await self.page.screenshot(type="jpeg", quality=self._quality)
            return base64.b64encode(data).decode("utf-8")
        except Exception:
            return ""

    # ── Highlight ─────────────────────────────────────────────────────────────

    async def _highlight(self, selector: str):
        try:
            locator = self.page.locator(selector).first
            await locator.evaluate("""el => {
                const prev = el.style.outline;
                el.style.outline = '3px solid #00ff88';
                el.style.boxShadow = '0 0 8px #00ff88';
                setTimeout(() => {
                    try { el.style.outline = prev; el.style.boxShadow = ''; } catch(e) {}
                }, 900);
            }""")
        except Exception:
            pass

    # ── Screenshot loop ───────────────────────────────────────────────────────

    async def _screenshot_loop(self):
        interval = 1.0 / self._fps
        while not self._stop_event.is_set():
            try:
                data = await asyncio.wait_for(
                    self.page.screenshot(type="jpeg", quality=self._quality),
                    timeout=3.0,
                )
                b64 = base64.b64encode(data).decode("utf-8")
                await self._send("screenshot", b64)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            except Exception:
                pass
            await asyncio.sleep(interval)

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _emit(self, msg_type: str, data):
        try:
            await self._send(msg_type, data)
        except Exception:
            pass


def json_str(s: str) -> str:
    """Escapa una cadena para uso seguro en f-strings con JS."""
    import json
    return json.dumps(s)
