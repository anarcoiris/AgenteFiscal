"""
charlie_self_healer.py — Sistema de auto-reparación por prueba-y-error para Charlie.

Problema que resuelve:
  El LLM de 3B alucina selectores CSS (e.g. "input[btnK]", "input[name='search_box']")
  porque razona desde texto, no desde el DOM real. Cuando un selector falla no hay
  aprendizaje: el siguiente paso repite el mismo error.

Solución:
  CharlieSelfHealer intercepta los fallos de fill/type/click y:
    1. Extrae candidatos del mapa semántico de la página actual.
    2. Los prueba en orden de probabilidad (timeout cortísimo: 800 ms).
    3. Guarda el ganador en CharlieMemory bajo una clave derivada de la URL + intención.
    4. La próxima vez que Charlie intente la misma acción en la misma URL, carga
       el selector guardado y lo intenta primero, sin necesitar al LLM.

Rendimiento:
  - Prueba hasta 6 candidatos × 800 ms = máximo 4.8 s extra por fallo.
  - En la práctica, el primer o segundo candidato suele funcionar.
  - Desde la segunda sesión, 0 pruebas extra (carga directo de memoria).
"""
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from charlie_browser import CharlieBrowser
    from charlie_memory  import CharlieMemory


# ── Selectores candidatos por intención ───────────────────────────────────────
# Para cada "tipo de acción", lista de selectores a probar en orden.

SEARCH_FILL_CANDIDATES = [
    "textarea[name='q']",
    "input[name='q']",
    "input[name='search']",
    "input[name='query']",
    "input[name='keywords']",
    "[role='searchbox']",
    "[role='combobox']",
    "input[type='search']",
    "textarea",
]

APPLY_BUTTON_CANDIDATES = [
    "button[aria-label*='Easy Apply']",
    "button[aria-label*='Solicitud sencilla']",
    ".jobs-apply-button",
    "button:has-text('Easy Apply')",
    "button:has-text('Solicitud sencilla')",
    "button:has-text('Inscribirme')",
    "button:has-text('Apply')",
    "a:has-text('Inscribirme en esta oferta')",
]

SUBMIT_BUTTON_CANDIDATES = [
    "button[aria-label*='Enviar']",
    "button:has-text('Enviar candidatura')",
    "button:has-text('Enviar')",
    "button:has-text('Submit')",
    "button:has-text('Confirmar')",
    "button[type='submit']",
    "input[type='submit']",
]

CONTINUE_BUTTON_CANDIDATES = [
    "button:has-text('Continuar')",
    "button:has-text('Siguiente')",
    "button:has-text('Next')",
    "button:has-text('Continue')",
    "[aria-label*='Continuar']",
]


def _classify_intent(desc: str, value: str = "") -> str:
    """Clasifica la intención de una acción a partir de su descripción."""
    d = (desc + " " + value).lower()
    if any(w in d for w in ["buscar", "search", "query", "python", "búsqueda"]):
        return "search_fill"
    if any(w in d for w in ["solicitud", "easy apply", "inscribir", "apply"]):
        return "apply_button"
    if any(w in d for w in ["enviar", "submit", "confirmar", "send"]):
        return "submit_button"
    if any(w in d for w in ["continuar", "siguiente", "next", "continue"]):
        return "continue_button"
    return "generic"


def _candidates_for_intent(intent: str) -> list[str]:
    return {
        "search_fill":      SEARCH_FILL_CANDIDATES,
        "apply_button":     APPLY_BUTTON_CANDIDATES,
        "submit_button":    SUBMIT_BUTTON_CANDIDATES,
        "continue_button":  CONTINUE_BUTTON_CANDIDATES,
    }.get(intent, [])


def _extract_candidates_from_map(page_map: str, intent: str) -> list[str]:
    """
    Extrae selectores concretos del mapa semántico de la página.
    Prioriza elementos marcados como 🌟 o 📝 según la intención.
    Devuelve selectores CSS reales (name=, role=) extraídos del mapa.
    """
    candidates = []

    for line in page_map.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Solo líneas relevantes según intención
        if intent == "search_fill":
            if "BUSCAR_AQUÍ" not in line and "name=\"q\"" not in line and "searchbox" not in line:
                continue
        elif intent == "apply_button":
            if "ACCIÓN_CLAVE" not in line and "easy apply" not in line.lower() and "inscribir" not in line.lower():
                continue
        elif intent in ("submit_button", "continue_button"):
            if "ACCIÓN" not in line:
                continue

        # Extraer name= del mapa: "[INDEX:5] textarea[q]" → textarea[name='q']
        m_name = re.search(r'\b(input|textarea|select|button)\[([^\]]+)\]', line)
        if m_name:
            tag  = m_name.group(1)
            attr = m_name.group(2)
            candidates.append(f"{tag}[name='{attr}']")

        # Extraer INDEX:N
        m_idx = re.search(r'INDEX:(\d+)', line)
        if m_idx:
            candidates.append(f"INDEX:{m_idx.group(1)}")

        # Extraer coordenadas como fallback
        m_coords = re.search(r'@\s*\((\d+),\s*(\d+)\)', line)
        if m_coords:
            candidates.append(f"@coords:{m_coords.group(1)},{m_coords.group(2)}")

    return candidates


class CharlieSelfHealer:
    """
    Sistema de auto-reparación por prueba-y-error.

    Uso típico (en el manejador de fill/type/click del agente):

        healer = CharlieSelfHealer(browser, memory)

        # Intento principal
        try:
            await browser.fill(selector, value)
        except Exception:
            winner = await healer.heal_fill(
                failed_selector=selector,
                value=value,
                desc=desc,
                page_map=page_map,
                url=url,
            )
            if winner:
                history.append(f"✅ fill [healed] '{winner}'='{value}'")
            else:
                raise  # No hay nada más que hacer
    """

    def __init__(self, browser: "CharlieBrowser", memory: "CharlieMemory | None" = None):
        self._browser = browser
        self._memory  = memory

    def _memory_key(self, url: str, intent: str) -> str:
        """Genera una clave estable de memoria para una URL + intención."""
        # Normalizar URL: solo dominio + primer segmento de path
        m = re.match(r'https?://([^/]+)(/[^/?#]*)?', url)
        domain = m.group(1) if m else url[:40]
        path   = (m.group(2) or "").strip("/")[:20]
        return f"healer.{domain}.{path}.{intent}".replace(" ", "_")

    def _load_from_memory(self, url: str, intent: str) -> str | None:
        """Recupera el selector ganador guardado para esta URL + intención."""
        if not self._memory:
            return None
        key = self._memory_key(url, intent)
        entries = self._memory._long_term.get("healer", {})
        entry   = entries.get(key)
        return entry["value"] if entry else None

    def _save_to_memory(self, url: str, intent: str, selector: str):
        """Guarda el selector ganador."""
        if not self._memory:
            return
        key = self._memory_key(url, intent)
        self._memory.learn("healer", key, selector, source="auto")

    async def _try_fill(self, selector: str, value: str, timeout_ms: int = 800) -> bool:
        """Intenta un fill con timeout corto. True = éxito."""
        try:
            if selector.startswith("@coords:"):
                x, y = map(int, selector[8:].split(","))
                await self._browser.click_at(x, y)
                await self._browser.press("End")
                # Para coordenadas, usar type para no perder el foco
                await self._browser.page.keyboard.type(value)
                return True

            if selector.startswith("INDEX:"):
                from charlie_agent import CharlieAgent  # evitar importación circular
                resolved = f":is(input, textarea, [contenteditable='true'], select) >> nth={selector[6:]}"
            else:
                resolved = selector

            locator = self._browser.page.locator(resolved).first
            await locator.wait_for(state="visible", timeout=timeout_ms)
            await locator.scroll_into_view_if_needed(timeout=timeout_ms)
            await locator.click(click_count=3, timeout=timeout_ms)
            await self._browser.page.fill(resolved, value, timeout=timeout_ms * 2)
            return True
        except Exception:
            return False

    async def _try_click(self, selector: str, timeout_ms: int = 800) -> bool:
        """Intenta un click con timeout corto. True = éxito."""
        try:
            if selector.startswith("@coords:"):
                x, y = map(int, selector[8:].split(","))
                await self._browser.click_at(x, y)
                return True

            if selector.startswith("INDEX:"):
                resolved = (
                    f":is(button, a, input, select, [role='button'], [role='link'], "
                    f"[contenteditable='true'], textarea) >> nth={selector[6:]}"
                )
            else:
                resolved = selector

            locator = self._browser.page.locator(resolved).first
            await locator.wait_for(state="visible", timeout=timeout_ms)
            await locator.scroll_into_view_if_needed(timeout=timeout_ms)
            await locator.click(timeout=timeout_ms * 2)
            return True
        except Exception:
            return False

    async def heal_fill(
        self,
        failed_selector: str,
        value: str,
        desc: str,
        page_map: str,
        url: str,
        log_fn=None,
    ) -> str | None:
        """
        Intenta recuperar un fill fallido probando selectores alternativos.
        Devuelve el selector ganador (str) o None si todo falló.
        """
        intent = _classify_intent(desc, value)
        winner = await self._probe(
            intent=intent, page_map=page_map, url=url,
            try_fn=lambda sel: self._try_fill(sel, value),
            log_fn=log_fn,
        )
        return winner

    async def heal_click(
        self,
        failed_selector: str,
        desc: str,
        page_map: str,
        url: str,
        log_fn=None,
    ) -> str | None:
        """
        Intenta recuperar un click fallido probando selectores alternativos.
        Devuelve el selector ganador (str) o None si todo falló.
        """
        intent = _classify_intent(desc)
        winner = await self._probe(
            intent=intent, page_map=page_map, url=url,
            try_fn=self._try_click,
            log_fn=log_fn,
        )
        return winner

    async def _probe(self, intent: str, page_map: str, url: str, try_fn, log_fn) -> str | None:
        """
        Núcleo del sistema de prueba-y-error.
        Construye la lista de candidatos, los prueba en orden y guarda el ganador.
        """
        async def _log(msg):
            if log_fn:
                await log_fn(msg)

        # 1. Primero: selector guardado en memoria (0 tiempo si existe)
        saved = self._load_from_memory(url, intent)
        if saved:
            await _log(f"🧠 Healer: probando selector guardado '{saved}'...")
            if await try_fn(saved):
                await _log(f"✅ Healer: selector de memoria funcionó '{saved}'")
                return saved
            await _log(f"⚠️ Healer: selector de memoria obsoleto, descartando.")
            # Borrar el obsoleto
            if self._memory:
                key = self._memory_key(url, intent)
                self._memory._long_term.get("healer", {}).pop(key, None)
                self._memory.save()

        # 2. Candidatos extraídos del mapa de la página (más específicos)
        map_candidates = _extract_candidates_from_map(page_map, intent)

        # 3. Candidatos genéricos por intención
        generic_candidates = _candidates_for_intent(intent)

        # Combinar sin duplicados, mapa primero
        seen     = set()
        all_candidates: list[str] = []
        for c in map_candidates + generic_candidates:
            if c not in seen:
                seen.add(c)
                all_candidates.append(c)

        await _log(f"🔬 Healer [{intent}]: probando {len(all_candidates)} candidatos...")

        for i, candidate in enumerate(all_candidates[:8]):  # Máximo 8 intentos
            await _log(f"   [{i+1}] '{candidate}'")
            if await try_fn(candidate):
                await _log(f"✅ Healer: '{candidate}' funcionó. Guardando en memoria.")
                self._save_to_memory(url, intent, candidate)
                return candidate

        await _log(f"❌ Healer: ningún candidato funcionó para intent='{intent}'")
        return None
