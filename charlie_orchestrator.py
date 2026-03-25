import asyncio
import time
from typing import Callable, Awaitable, Optional, Dict, Any

from charlie_browser import CharlieBrowser
from charlie_planner import CharliePlanner
from charlie_memory import CharlieMemory
from charlie_self_healer import CharlieSelfHealer
from charlie_perception import CharliePerception
from charlie_classifier import CharlieClassifier
from charlie_models import Task, Session, Trace, Action
from charlie_skills import CharlieSkillManager
from charlie_bot_controller import CharlieBotController

class CharlieOrchestrator:
    """
    Nuevo Orchestrator híbrido (Fase 6 de Target.md).
    Remplaza parcialmente a charlie_agent.py (v5) pero con arquitectura "Antigravity".
    
    Flujo dinámico:
    1. Classifier clasifica la tarea (Familia, Entorno).
    2. Modos: Browser / Desktop / Híbrido.
    3. Loop Plan-and-Act: Perceive -> Plan/Infer -> Execute Controller.
    """

    def __init__(
        self,
        ws_send: Callable[[str, object], Awaitable[None]],
        ollama_url: str,
        model: str,
        task_str: str,
        max_steps: int = 35,
        browser: Optional[CharlieBrowser] = None,
        planner: Optional[CharliePlanner] = None,
        memory: Optional[CharlieMemory]  = None,
        answer_queue: Optional[asyncio.Queue] = None,
    ):
        self._send = ws_send
        self._ollama = ollama_url.rstrip("/")
        self._model = model
        self._raw_task = task_str
        self._max_steps = max_steps
        
        # Componentes Antigravity
        self._browser = browser
        self._planner = planner
        self._memory = memory
        self._classifier = CharlieClassifier(self._ollama, self._model)
        
        # Perception
        self._perception = CharliePerception(browser=self._browser, use_desktop=False) # OS dev true later
        
        # Controladores locales
        # self._desktop_ctrl = ...  # (Fase 5)
        self._healer = None
        
        self._step = 0

    async def _log(self, *args):
        # Mismo formato de logs del agent original
        level = "info"
        msg = ""
        if len(args) == 2:
            level, msg = args
        elif len(args) == 1:
            msg = args[0]
        try:
            await self._send("log", {"level": level, "msg": msg})
        except Exception:
            print(f"[{level.upper()}] {msg}")

    async def run(self) -> dict:
        await self._log("🧠 Antigravity Orchestrator V6 | Iniciando Clasificación...")
        
        # 1. Fase 3: Clasificación Inicial
        classification = await self._classifier.classify_task(self._raw_task)
        family = classification.get("task_family", "general")
        env = classification.get("environment", "browser")
        atoms = classification.get("expected_ui_atoms", [])
        
        await self._log(f"🔎 Clasificación: Familia={family} | Entorno={env} | Atoms={atoms}")
        
        # Activar el entorno adecuado (si es desktop hay que usar pyautogui / mss / etc)
        if env == "desktop" or env == "hybrid":
            await self._log("⚠️ Entorno Desktop requerido. Habilitando Perception Desktop...")
            self._perception._use_desktop = True # Para Phase 5 Desktop Controller

        # Iniciar Session Memory (Episódica)
        session = Session(session_id=str(int(time.time())), task_description=self._raw_task)
        
        # 2. Reutilización de charlie_agent.py (BrowserController legacy) por ahora
        # Para cumplir la "Fase 1: Base sólida y fallback"
        if env == "browser":
            await self._log("🌐 Delegando ejecución web al agente principal de Browser...")
            # Aquí idealmente usaríamos el BrowserController refactorizado. 
            # Como fallback de transición interactuamos con el CharlieAgent orgánico 
            # o el self._browser directamente.
            
            # Vamos a simular un loop de orquestación donde el Orchestrator
            # coordina la percepción y luego decide el controlador.
            return await self._run_browser_orchestrated()
        
        elif env == "desktop":
            await self._log("🖥️ Ejecutando Desktop Controller (Stub)...")
            return {"action": "done", "result": "Desktop actions not implemented fully yet."}
            
        return {"action": "error", "reason": "Entorno híbrido en desarrollo"}

    async def _run_browser_orchestrated(self) -> dict:
        """Loop web-first (Fase 4 Browser Controller) integrado con Bot Mode y Perception Layer."""
        from charlie_agent import CharlieAgent
        
        # 1. Intentar Bot Mode rápido primero
        skill_manager = CharlieSkillManager()
        # Generamos un identificador único para la tarea, ej: "infojobs_administrador_de_sistemas"
        task_sig = self._raw_task.split("===")[0].replace(" ", "_").lower().strip()[:50]
        if "google_test" in self._raw_task.lower():
            task_sig = "google_test"
            
        bot_sequence = skill_manager.get_skill(task_sig)
        
        if bot_sequence:
            await self._log(f"🤖 [BOT MODE] Encontrada macro procedural para '{task_sig}'. Ejecutando a velocidad luz...")
            bot = CharlieBotController(self._browser, self._send)
            result = await bot.run_skill(task_sig, bot_sequence)
            
            if result.get("action") == "done":
                # Ensure the summary doesn't break
                result["summary"] = result.get("reason", "Ejecutado por Bot Ultrarrápido.")
                return result
            else:
                await self._log("⚠️ [HEALING REQUIRED] El Bot Mode falló en un paso. Delegando al LLM Agent (Slow Mode) para recuperar...")
                # Aquí iría el Self-Healer explícito. Por ahora delegamos toda la tarea de nuevo al Agente.
        else:
            await self._log("⚙️ [AGENT MODE] No hay macro de bot conocida para esta tarea. El Agente explorará y grabará la ruta.")

        # 2. Inicializamos el sub-agente explorador (Slow Mode)
        browser_agent = CharlieAgent(
            browser=self._browser,
            ws_send=self._send,
            ollama_url=self._ollama,
            model=self._model,
            task=self._raw_task,
            max_steps=self._max_steps,
            planner=self._planner,
            memory=self._memory
        )
        
        result = await browser_agent.run()
        
        # 3. Consolidación de memoria procedimental
        # Si el agente logró terminar ("done"), extraemos sus clicks exitosos y los guardamos
        if result.get("action") == "done" and getattr(browser_agent, "successful_sequence", None):
            await self._log("💾 Consolidando secuencia exitosa en el Bot Engine (SkillManager)...")
            skill_manager.save_skill(task_sig, browser_agent.successful_sequence)
            
        return result
