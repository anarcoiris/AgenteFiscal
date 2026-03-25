import asyncio
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Callable

# Intentar importar librerías de escritorio. Si fallan, el desktop controller estará inactivo.
try:
    import pyautogui
    # Configuración de seguridad
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.5
    import mss
    import mss.tools
    DESKTOP_READY = True
except ImportError:
    DESKTOP_READY = False

class CharlieDesktopController:
    """
    Controlador de Escritorio (Fase 5).
    Ejecuta acciones a nivel SO: ratón, teclado, apps.
    """

    def __init__(self, log_fn: Optional[Callable] = None):
        self._log = log_fn or print
        self._desktop_ready = DESKTOP_READY
        self._sct = mss.mss() if DESKTOP_READY else None

    async def _async_log(self, msg: str):
        if asyncio.iscoroutinefunction(self._log):
            await self._log(msg)
        else:
            self._log(msg)

    async def is_ready(self) -> bool:
        return self._desktop_ready

    async def capture_screen_b64(self) -> str:
        """Captura toda la pantalla y devuelve base64 JPEG."""
        if not self._desktop_ready:
            return ""
        
        try:
            # Ejecutamos sct en un thread para no bloquear asyncio
            monitor = self._sct.monitors[0]  # Todas las pantallas
            sct_img = self._sct.grab(monitor)
            
            # Convertimos la imagen raw a base64 (requiere Pillow, pero mss puede usar PIL)
            from PIL import Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Resize a max 1280x720 para no saturar Ollama
            img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=60)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            await self._async_log(f"⚠️ Error capturando pantalla: {e}")
            return ""

    async def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción sobre el escritorio."""
        if not self._desktop_ready:
            return {"stop": True, "error": "Desktop libraries (pyautogui/mss) not installed."}

        kind = action.get("action", "")
        
        try:
            if kind == "click_at":
                x = action.get("x", 0)
                y = action.get("y", 0)
                await self._async_log(f"🖱️ Desktop Click: ({x}, {y})")
                pyautogui.click(int(x), int(y))
                
            elif kind == "type":
                value = action.get("value", "")
                await self._async_log(f"⌨️ Desktop Type: '{value}'")
                pyautogui.write(value, interval=0.05)
                
            elif kind == "press":
                key = action.get("key", "enter").lower()
                # Adaptación de nombres Playwright a pyautogui
                key_map = {
                    "enter": "enter", "escape": "esc", "tab": "tab",
                    "backspace": "backspace", "arrowdown": "down", "arrowup": "up"
                }
                pg_key = key_map.get(key, key)
                await self._async_log(f"⌨️ Desktop Press: {pg_key}")
                pyautogui.press(pg_key)
                
            elif kind == "open_app":
                # Abre bash o menú inicio y escribe para buscar
                app_name = action.get("app_name", "")
                await self._async_log(f"🖥️ Abriendo app: {app_name}")
                pyautogui.press("win")
                await asyncio.sleep(0.5)
                pyautogui.write(app_name, interval=0.05)
                await asyncio.sleep(0.5)
                pyautogui.press("enter")
                await asyncio.sleep(2.0)
                
            elif kind == "done":
                await self._async_log("✅ Desktop Tarea Completada")
                return {"stop": True, "payload": action}
                
            else:
                await self._async_log(f"⚠️ Desktop Action desconocida: {kind}")

        except Exception as e:
            await self._async_log(f"❌ Error en Desktop Action '{kind}': {e}")
            return {"stop": False, "error": str(e)}

        return {"stop": False}
