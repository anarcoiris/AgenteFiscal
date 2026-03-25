import asyncio
from typing import Dict, Any, Optional, TYPE_CHECKING
import platform

if TYPE_CHECKING:
    from charlie_browser import CharlieBrowser

class CharliePerception:
    """
    Capa de percepción (Fase 1: separar observación/acción).
    Provee el "entorno mixto" a Charlie: DOM + Screenshots + OS (Desktop).
    """

    def __init__(self, browser: Optional["CharlieBrowser"] = None, use_desktop: bool = False):
        self._browser = browser
        self._use_desktop = use_desktop

    async def observe(self, needs_vision: bool = False) -> Dict[str, Any]:
        """Extrae el estado del entorno de las fuentes activas."""
        state = {
            "browser": None,
            "desktop": None,
            "system": self._get_system_context()
        }

        # --- 1) Navegador ---
        if self._browser:
            try:
                url = await self._browser.get_url()
                text = await self._browser.get_text_content()
                page_map = await self._browser.get_page_map()
                login = await self._browser.check_login_status()

                screenshot = ""
                if needs_vision:
                    screenshot = await self._browser.take_screenshot_b64()

                state["browser"] = {
                    "url": url,
                    "text": text,
                    "map": page_map,
                    "login": login,
                    "screenshot_b64": screenshot
                }
            except Exception as e:
                print(f"⚠️ Error leyendo entorno browser: {e}")

        # --- 2) Escritorio ---
        if self._use_desktop:
            # TODO: Implemetar con pyautogui / mss / win32gui
            state["desktop"] = {
                "active_window": "unknown", # e.g. "Visual Studio Code"
                "focused_element_text": "",
                "screenshot_b64": "",       # mss
                "ocr_map": ""               # Tesseract / OCR local
            }

        return state

    def _get_system_context(self) -> Dict[str, str]:
        """Info del SO."""
        return {
            "os": platform.system(),
            "release": platform.release()
        }
