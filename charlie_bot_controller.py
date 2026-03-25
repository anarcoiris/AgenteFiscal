import asyncio
from typing import Callable, Awaitable
from charlie_browser import CharlieBrowser

class CharlieBotController:
    """
    Motor de automatización determinista y ultrarrápido (Bot Mode).
    Ejecuta scripts predefinidos en formato JSON sin usar LLM.
    Levanta advertencias si falla un selector, permitiendo la intervención de Curación de Charlie.
    """
    def __init__(self, browser: CharlieBrowser, ws_send: Callable[[str, any], Awaitable[None]]):
        self._browser = browser
        self._send = ws_send

    async def _log(self, text: str):
        await self._send("log", f"⚡ [BOT] {text}")

    async def run_skill(self, skill_name: str, steps: list[dict]) -> dict:
        """
        Ejecuta sincrónicamente una secuencia de pasos en el navegador.
        Retorna success o lanza flag_error identificando exactamente dónde falló.
        """
        await self._log(f"Iniciando macro '{skill_name}' ({len(steps)} pasos operativos)")
        page = self._browser.page
        if not page:
            return {"action": "error", "reason": "El navegador no está activo."}

        try:
            for i, step in enumerate(steps):
                action = step.get("action")
                selector = step.get("selector")
                
                await self._log(f"Paso {i+1}: {action} {selector or step.get('url', '')}")
                
                if action == "navigate":
                    url = step.get("url")
                    if url:
                        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                
                elif action == "click":
                    if selector:
                        await page.wait_for_selector(selector, state="visible", timeout=6000)
                        await page.click(selector, timeout=5000)
                
                elif action == "fill":
                    value = str(step.get("value", ""))
                    if selector:
                        await page.wait_for_selector(selector, state="visible", timeout=6000)
                        await page.fill(selector, value, timeout=5000)
                        
                elif action == "press":
                    key = step.get("key", step.get("value", "Enter"))
                    await page.keyboard.press(key)
                
                elif action == "done":
                    return {"action": "done", "reason": "Macro finalizada con éxito explícito."}
                
                # Estabilización mínima tras cada acción para que el DOM fluya
                await asyncio.sleep(0.7)
                
            return {"action": "done", "reason": "Se ejecutaron todos los pasos de la secuencia."}
            
        except Exception as e:
            err_msg = str(e).split('\\n')[0][:100]
            await self._log(f"💥 Fallo crítico en el paso {i+1} [{action} en '{selector}']: {err_msg}")
            
            return {
                "action": "error", 
                "reason": "La secuencia bot falló y requiere auto-healing.",
                "failed_step_index": i,
                "failed_step": step,
                "exception": str(e)
            }
