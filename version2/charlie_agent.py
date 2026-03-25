"""
charlie_agent.py — Ejecutor LLM de Charlie usando Ollama directamente.

v5 — Visión lazy + Auto-reparación por prueba-y-error:
  - NUEVO: visión lazy — screenshot solo cuando realmente es necesario.
           En el resto de pasos el mapa semántico reemplaza la imagen.
           Reducción esperada de latencia: de 40-80 s/paso a 5-15 s/paso.
  - NUEVO: CharlieSelfHealer — cuando fill/type/click falla, prueba candidatos
           alternativos con timeout de 800 ms y guarda el ganador en CharlieMemory.
  - NUEVO: Search Shield ampliado — intercepta cualquier click en página de búsqueda,
           no solo los que contienen "textarea" en el selector.
  - NUEVO: extracción de selector real del mapa — cuando el LLM envía un selector
           ambiguo (INDEX:N sin contexto), se busca el atributo name= en el mapa.
  - MANTENIDO: Plan-and-Act, inferencia de paso, máquina de fases,
               temperatura dinámica, reintentos, validación done, memoria.
"""
import asyncio
import json
import re
import time
from enum import Enum
from typing import Callable, Awaitable, TYPE_CHECKING

import httpx

from charlie_browser      import CharlieBrowser
from charlie_self_healer  import CharlieSelfHealer

if TYPE_CHECKING:
    from charlie_planner import CharliePlanner
    from charlie_memory  import CharlieMemory


# ── Máquina de estados ─────────────────────────────────────────────────────────

class Phase(str, Enum):
    INIT        = "INIT"
    SEARCHING   = "SEARCHING"
    SELECTING   = "SELECTING"
    APPLYING    = "APPLYING"
    FILLING     = "FILLING"
    SUBMITTING  = "SUBMITTING"
    DONE        = "DONE"
    ERROR       = "ERROR"

PHASE_GUIDES: dict[Phase, str] = {
    Phase.INIT:       "Arrancando. Navega a la URL de inicio.",
    Phase.SEARCHING:  "Estás en resultados. Localiza la primera oferta relevante.",
    Phase.SELECTING:  "Haz clic en una oferta concreta para ver su detalle.",
    Phase.APPLYING:   "Encuentra el botón de solicitud (Easy Apply / Inscribirme) y púlsalo.",
    Phase.FILLING:    "Rellena los campos del formulario uno a uno con los datos de la tarea.",
    Phase.SUBMITTING: "Confirma el envío. Busca 'Enviar', 'Confirmar' o 'Submit'.",
}

PHASE_ADVANCE_SIGNALS: dict[Phase, list[str]] = {
    Phase.SEARCHING:  ["solicitud sencilla", "easy apply", "inscribirme", "apply now"],
    Phase.SELECTING:  ["solicitud sencilla", "easy apply", "inscribirme", "salario", "salary"],
    Phase.APPLYING:   ["continuar", "siguiente", "next", "salario deseado", "nivel de inglés"],
    Phase.FILLING:    ["confirmar", "enviar candidatura", "submit application", "aplicación enviada"],
}

# Prompt para la llamada ligera de inferencia de paso del plan
STEP_INFER_SYSTEM = """\
Eres el supervisor de Charlie. Tu única tarea es determinar en qué paso del plan está el agente.
Responde SOLO con JSON: {"step_index": <número entero comenzando en 0>, "reason": "<una línea>"}
"""


# ── Prompt de sistema del ejecutor ─────────────────────────────────────────────

EXECUTOR_SYSTEM = """\
Eres el Ejecutor de Charlie, un agente web autónomo.
Recibes el estado actual de la página (mapa semántico + texto) y UN sub-paso a ejecutar.
Responde SOLO con un objeto JSON válido. Sin texto extra.

ACCIONES DISPONIBLES:
  navigate   → {"action":"navigate","url":"<url>"}
  click      → {"action":"click","selector":"<css, text=..., o INDEX:N>","desc":"<por qué>"}
  click_at   → {"action":"click_at","x":<int>,"y":<int>,"desc":"<por qué>"}
  fill       → {"action":"fill","selector":"<css o INDEX:N>","value":"<valor>","desc":"<campo>"}
  type       → {"action":"type","selector":"<css>","value":"<valor>","desc":"<campo>"}
  press      → {"action":"press","key":"<Enter|Tab|Escape>"}
  scroll     → {"action":"scroll","direction":"<down|up>","amount":<px>}
  wait       → {"action":"wait","ms":<ms>}
  extract    → {"action":"extract","selector":"<css>","desc":"<qué se extrae>"}
  note       → {"action":"note","text":"<apunta algo importante>"}
  human_ask  → {"action":"human_ask","question":"<pregunta clara al usuario>","context":"<por qué necesitas ayuda>"}
  done       → {"action":"done","result":"<resumen>","job_url":"<url exacta>","company":"<empresa>"}
  error      → {"action":"error","reason":"<qué falló>"}

NOTAS SOBRE EL MAPA DE PÁGINA:
- El mapa agrupa elementos por ZONA: modal (urgente), formulario, resultados, cabecera.
- Usa 'pos:N' dentro de cada zona para referirte al N-ésimo elemento de esa sección.
- Los elementos con 🌟 son críticos para la tarea actual.
- Los elementos con 📝 son campos de formulario que debes rellenar.
- Si ves [ZONA:modal], lo que hay dentro tiene prioridad sobre el resto.

REGLAS:
1. Para buscar: 'fill' en el elemento 🌟BUSCAR_AQUÍ, luego 'press' Enter.
2. Para formularios: lee 'name' del campo en el mapa. No pongas salario en teléfono.
3. Si no ves el botón: 'scroll' down para revelar más de la página.
4. NUNCA inventes URLs.
5. 'done' solo tras confirmación visual de envío.
6. Usa 'human_ask' si llevas 3+ pasos sin progresar o si la página es inesperada.
7. Incluye "thought" con tu razonamiento antes de "action".
"""


class CharlieAgent:
    """
    Ejecutor LLM que opera sobre un CharlieBrowser.
    Acepta un CharliePlanner y un CharlieMemory opcionales.
    Bucle: [planificar] → observar → inferir-paso → pensar → actuar → verificar → repetir.
    """

    def __init__(
        self,
        browser: CharlieBrowser,
        ws_send: Callable[[str, object], Awaitable[None]],
        ollama_url: str,
        model: str,
        task: str,
        max_steps: int = 35,
        planner: "CharliePlanner | None" = None,
        memory:  "CharlieMemory | None"  = None,
        answer_queue: "asyncio.Queue | None" = None,
    ):
        self._browser      = browser
        self._send         = ws_send
        self._ollama       = ollama_url.rstrip("/")
        self._model        = model
        self._task         = task
        self._max_steps    = max_steps
        self._planner      = planner
        self._memory       = memory
        # Cola para recibir respuestas del usuario a human_ask
        self._answer_queue: asyncio.Queue = answer_queue or asyncio.Queue()

        self._step: int               = 0
        self._phase: Phase            = Phase.INIT
        self._history: list[str]      = []
        self._failed_selectors: set[str] = set()
        self._coordinate_memory: dict[str, tuple[int, int]] = {}
        self._system_hint: str        = ""
        self._scratchpad: str         = ""
        self._temperature: float      = 0.1
        self._last_url: str           = ""
        self._same_url_count: int     = 0
        self._last_action_sigs: list[str] = []

        self._plan: list[str]         = []
        self._plan_idx: int           = 0
        self._plan_step_iters: int    = 0
        self._replan_count: int       = 0
        self._step_times: list[float] = []

        # ── Visión lazy ────────────────────────────────────────────────────────
        # Solo se envía screenshot al LLM cuando _needs_vision es True.
        # Se activa automáticamente ante cambios de URL, fallos, y cada 5 pasos.
        self._needs_vision: bool      = True   # Siempre true en el primer paso
        self._last_vision_url: str    = ""
        self._last_vision_step: int   = 0
        self._last_action_failed: bool = False
        self._page_map: str           = ""   # Último mapa semántico — accesible desde _execute()

        # ── Self-healer (se instancia en run() cuando hay memoria disponible) ──
        self._healer: CharlieSelfHealer | None = None

    # ── Punto de entrada ───────────────────────────────────────────────────────

    async def run(self) -> dict:
        mode = "Plan-and-Act" if self._planner else "Directo"
        await self._log(f"🧠 Charlie v5 [{mode}] | Modelo: {self._model}")

        # Instanciar self-healer
        self._healer = CharlieSelfHealer(self._browser, self._memory)

        for step in range(1, self._max_steps + 1):
            self._step = step
            t0 = time.monotonic()

            url        = await self._browser.get_url()
            page_text  = await self._browser.get_text_content()
            page_map   = await self._browser.get_page_map()
            self._page_map = page_map  # disponible en _execute() y helpers
            login_stat = await self._browser.check_login_status()

            # ── Visión lazy: solo capturar imagen cuando sea necesario ────────
            # Se activa cuando: URL cambió, acción anterior falló, modal presente,
            # o han pasado 5 pasos sin imagen.
            url_changed   = url != self._last_vision_url
            modal_present = "ZONA:modal" in page_map
            steps_since   = step - self._last_vision_step
            self._needs_vision = (
                url_changed or self._last_action_failed or steps_since >= 5 or modal_present
            )

            if self._needs_vision:
                screenshot = await self._browser.take_screenshot_b64()
                self._last_vision_url  = url
                self._last_vision_step = step
                vision_marker = "👁️"
            else:
                screenshot   = ""
                vision_marker = "⚡"

            await self._browser.dismiss_popups()
            self._last_action_failed = False

            # ── INIT: planificar y navegar ────────────────────────────────────
            if self._phase == Phase.INIT:
                if self._planner and not self._plan:
                    await self._log("📋 Generando plan...")
                    self._plan, start_url = await self._planner.generate_plan(
                        self._task, url, login_stat
                    )
                    # Filtrar pasos inválidos que el LLM a veces genera
                    _bad = {"error", "login", "verificar login", "comprobar login"}
                    self._plan = [
                        p for p in self._plan
                        if p.strip().lower() not in _bad
                        and not p.strip().lower().startswith("error")
                    ]
                    if not self._plan:
                        self._plan = self._planner._fallback_plan(self._task, url)

                    await self._log(f"📋 Plan ({len(self._plan)} pasos):")
                    for i, p in enumerate(self._plan, 1):
                        await self._log(f"   {i}. {p}")

                    # Navegar siempre al start_url si es distinto de la URL actual
                    if start_url and start_url not in ("about:blank", url, ""):
                        await self._execute({"action": "navigate", "url": start_url})
                        self._phase = Phase.SEARCHING
                        continue
                else:
                    target = self._extract_start_url()
                    if target:
                        await self._execute({"action": "navigate", "url": target})
                        self._phase = Phase.SEARCHING
                        continue

            self._update_coordinate_memory(page_map)
            self._maybe_advance_phase(page_text, url)
            self._track_url(url)

            # ── Replanning ────────────────────────────────────────────────────
            if self._planner and self._plan:
                self._plan_step_iters += 1
                if self._plan_step_iters > 8 and self._replan_count < 2:
                    await self._replan(url, login_stat)

            # ── Inferencia contextual de paso del plan (cada 3 pasos) ─────────
            if self._plan and step % 3 == 0:
                inferred = await self._infer_plan_step(url, page_text, page_map)
                if inferred is not None and inferred != self._plan_idx:
                    await self._log(
                        f"🔍 Inferencia: paso real es {inferred + 1} "
                        f"(era {self._plan_idx + 1}): '{self._plan[inferred]}'"
                    )
                    self._plan_idx        = inferred
                    self._plan_step_iters = 0

            plan_guide = self._current_plan_step()
            phase_str  = f"{self._phase.value}" + (f" → '{plan_guide}'" if plan_guide else "")
            await self._log(f"📍 Paso {step}/{self._max_steps} [{phase_str}] {vision_marker}")

            # ── Contexto de memoria ───────────────────────────────────────────
            mem_context = url + " " + (plan_guide or self._phase.value)
            mem_block   = self._memory.prompt_block(mem_context) if self._memory else ""

            action = await self._think_with_retry(
                url, page_text, page_map, login_stat, screenshot, plan_guide, mem_block
            )
            if action is None:
                await self._log("❌ LLM sin JSON válido. Abortando.")
                return {"action": "error", "reason": "JSON inválido del LLM"}

            await self._log(f"🐾 {action.get('action')} — {action.get('desc', action.get('reason',''))}")

            result = await self._execute(action)

            elapsed = time.monotonic() - t0
            self._step_times.append(elapsed)
            avg = sum(self._step_times) / len(self._step_times)
            await self._log("debug", f"⏱️ Paso {step}: {elapsed:.1f}s | Media: {avg:.1f}s | visión={self._needs_vision}")

            if result.get("stop"):
                return result.get("payload", action)

        avg_final = sum(self._step_times) / max(1, len(self._step_times))
        await self._log(f"⚠️ Máximo de pasos alcanzado. Media: {avg_final:.1f}s/paso")
        return {"action": "error", "reason": "Máximo de pasos alcanzado"}

    # ── LLM con reintentos ─────────────────────────────────────────────────────

    async def _think_with_retry(self, url, page_text, page_map, login_stat, screenshot, plan_guide, mem_block="") -> dict | None:
        for attempt in range(3):
            result = await self._think(url, page_text, page_map, login_stat, screenshot, plan_guide, mem_block, attempt)
            if result is not None:
                return result
            await self._log(f"⚠️ Intento {attempt + 1}/3 fallido...")
            await asyncio.sleep(1.0)
        return None

    async def _think(self, url, page_text, page_map, login_stat, screenshot_b64, plan_guide, mem_block="", attempt=0) -> dict | None:
        history_str  = "\n".join(self._history[-5:]) if self._history else "Ninguna."
        hint_block   = f"\n🚨 CONSEJO: {self._system_hint}" if self._system_hint else ""
        pad_block    = f"\nNOTAS: {self._scratchpad}"        if self._scratchpad  else ""
        failed_block = f"\nFALLADOS: {', '.join(self._failed_selectors)}" if self._failed_selectors else ""
        mem_section  = f"\n{mem_block}"                      if mem_block         else ""

        # El mapa ya es el contexto primario; get_text_content es backup
        map_limit  = [3000, 1500, 500][min(attempt, 2)]
        text_limit = [ 600,  300, 150][min(attempt, 2)]

        active_guide = plan_guide if plan_guide else PHASE_GUIDES.get(self._phase, "")

        user_content = f"""\
### SUB-PASO: {active_guide}
### CONTEXTO: Paso {self._step}/{self._max_steps} | Fase {self._phase.value} | URL: {url} | Login: {login_stat}
{hint_block}{pad_block}{failed_block}{mem_section}

ESTADO DE LA URL:{"⚠️ BÚSQUEDA YA REALIZADA — q= está en la URL. No vuelvas a rellenar ni a pulsar Enter. Tu tarea es observar los resultados." if ("q=" in url.lower() or "/search" in url.lower()) and any(w in url.lower() for w in ["google", "bing", "duckduckgo"]) else ""}

HISTORIAL:
{history_str}

MAPA SEMÁNTICO (usa los atributos name= y INDEX:N que aparecen aquí, NO inventes CSS):
{page_map[:map_limit]}

TEXTO:
{page_text[:text_limit]}

INSTRUCCIÓN SOBRE SELECTORES:
- USA el atributo name= exacto del mapa: e.g. "textarea[name='q']", "input[name='salary']"
- O usa INDEX:N del mapa: e.g. "INDEX:5"
- NO inventes clases CSS como "input[btnK]" o ".search-box"
- Para botones con texto visible usa: text='Texto exacto'

Observa el screenshot (si hay) y el mapa. Responde SOLO con JSON."""

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": EXECUTOR_SYSTEM},
                {"role": "user",   "content": user_content,
                 "images": [screenshot_b64] if screenshot_b64 else []},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_ctx":     16384,
                "num_batch":   32,
                "num_gpu":     12,
                "top_p":       0.3,
                "num_thread":  8,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self._ollama}/api/chat", json=payload)
                if resp.status_code != 200:
                    await self._log(f"🛑 Ollama {resp.status_code}: {resp.text[:200]}")
                    return None
                content = resp.json().get("message", {}).get("content", "")
                await self._log("debug", f"🧠 RAW [{attempt}]: {content[:300]}")
                parsed = self._parse_json(content)
                # Solo borrar la pista si el modelo la recibió correctamente
                # (parsed no es None). Si falló el parse, mantenerla para el reintento.
                if parsed is not None:
                    self._system_hint = ""
                return parsed
        except Exception as e:
            await self._log(f"❌ Error Ollama: {e}")
            return None

    # ── Ejecutar acción ────────────────────────────────────────────────────────

    async def _execute(self, action: dict) -> dict:
        kind = action.get("action", "")
        desc = action.get("desc", action.get("reason", ""))

        sig = f"{kind}:{action.get('selector','')}{action.get('x','')}{action.get('y','')}{action.get('url','')}"
        self._last_action_sigs.append(sig)
        if len(self._last_action_sigs) > 8:
            self._last_action_sigs.pop(0)

        if len(self._last_action_sigs) >= 3 and len(set(self._last_action_sigs[-3:])) == 1:
            self._temperature = min(self._temperature + 0.15, 0.8)
            msg = f"Llevas 3 veces repitiendo '{kind}'. CAMBIA estrategia."
            self._history.append(f"⚠️ BUCLE: {msg}")
            self._system_hint = msg
            await self._log(f"🔄 Bucle. T={self._temperature:.2f}")
            await asyncio.sleep(1.0)
            return {"stop": False}

        if len(self._last_action_sigs) >= 4:
            s = self._last_action_sigs
            if s[-1] == s[-3] and s[-2] == s[-4]:
                self._temperature = min(self._temperature + 0.1, 0.8)
                self._system_hint = "Bucle alternante. Prueba scroll, wait, o ruta diferente."
                self._history.append("⚠️ BUCLE ALT")
                await asyncio.sleep(1.0)
                return {"stop": False}

        url_before = await self._browser.get_url()

        try:
            if kind == "navigate":
                url = action.get("url", "").strip()
                if not url:
                    return {"stop": False}
                await self._browser.navigate(url)
                await asyncio.sleep(2.0)
                self._history.append(f"✅ navigate → {url}")
                self._phase = self._infer_phase_from_url(url)
                if self._try_advance_plan_step():
                    await self._log(f"🏁 Plan completado tras navigate → {url}")
                    return await self._auto_done(url)

            elif kind in ["click", "click_at"]:
                selector = action.get("selector")
                x = action.get("x")
                y = action.get("y")
                for field in ["coordinate", "point", "coords"]:
                    val = action.get(field)
                    if isinstance(val, (list, tuple)) and len(val) == 2:
                        x, y = int(val[0]), int(val[1]); break
                    elif isinstance(val, dict):
                        x, y = val.get("x"), val.get("y"); break

                cur_url = await self._browser.get_url()
                # 🛡️ Search Shield — intercepta clicks en buscadores cuando hay input de búsqueda.
                # Guarda: si q= ya está en la URL, la búsqueda ya se realizó → no disparar de nuevo.
                _is_search_engine = any(d in cur_url.lower() for d in ["google", "bing", "duckduckgo"])
                _has_search_in_map = "BUSCAR_AQUÍ" in self._page_map
                _search_already_done = (
                    "q=" in cur_url.lower()
                    or "/search" in cur_url.lower()
                    or any("press Enter [shield]" in h for h in self._history[-6:])
                )
                if _is_search_engine and _has_search_in_map and not _search_already_done and kind != "click_at":
                    # ── Extraer la query de búsqueda ──────────────────────────
                    # Prioridad: 1) comillas en el paso del plan actual
                    #            2) comillas en la descripción de la acción
                    #            3) regex flexible sobre la tarea
                    #            4) el texto completo de la tarea (fallback último)
                    _query = ""

                    # 1. El planificador ya puso la query entre comillas en el paso del plan:
                    #    "Rellenar el campo de búsqueda con el texto 'Haberman pdf'"
                    _plan_step = self._current_plan_step()
                    _m = re.search(r"'([^']{3,})'", _plan_step) or re.search(r'"([^"]{3,})"', _plan_step)
                    if _m:
                        _query = _m.group(1).strip()

                    # 2. Comillas en la descripción de la acción del LLM
                    if not _query and desc:
                        _m = re.search(r"'([^']{3,})'", desc) or re.search(r'"([^"]{3,})"', desc)
                        if _m:
                            _query = _m.group(1).strip()

                    # 3. Regex flexible sobre la tarea: cubre "Búscame", "Busca el",
                    #    "Encuentra", "Busca en google", "Buscame", "Búsca", etc.
                    if not _query:
                        _patterns = [
                            r"(?:b[uú]sca(?:me|r)?(?:\s+en\s+\w+)?)\s+(.+?)(?:\s+en\s+(?:pdf|google|internet|la web))?$",
                            r"(?:encuentra|find|search(?:\s+for)?)\s+(.+?)(?:\s+(?:en|in|on)\s+\w+)?$",
                            r"(?:descarga|download)\s+(.+?)(?:\s+en\s+pdf)?$",
                        ]
                        for _pat in _patterns:
                            _m = re.search(_pat, self._task, re.IGNORECASE)
                            if _m:
                                _query = _m.group(1).strip()
                                # Limpiar artículos iniciales: "el libro de" → mantener
                                _query = re.sub(r"^(el|la|los|las|un|una)\s+", "", _query, flags=re.IGNORECASE)
                                break

                    # 4. Fallback: usar la tarea completa truncada (mejor que "Python")
                    if not _query:
                        _query = self._task.strip()[:80]

                    await self._log(f"🛡️ Search Shield: fill('{_query}') + Enter")
                    if self._healer:
                        winner = await self._healer.heal_fill(
                            failed_selector="", value=_query, desc="search",
                            page_map=self._page_map, url=cur_url, log_fn=self._log
                        )
                        _fill_sel = winner if (winner and not winner.startswith("@coords:")) else "textarea[name='q']"
                    else:
                        _fill_sel = "textarea[name='q']"
                    try:
                        await self._browser.fill(_fill_sel, _query, description="búsqueda")
                    except Exception:
                        await self._browser.page.type(_fill_sel, _query)
                    self._history.append(f"✅ fill [shield] búsqueda='{_query}'")
                    self._last_action_failed = False
                    await asyncio.sleep(0.4)
                    await self._browser.press("Enter")
                    self._history.append("✅ press Enter [shield]")
                    await asyncio.sleep(1.5)
                    url_after = await self._browser.get_url()
                    self._history.append(f"   → {url_after}")
                    # Marcar como COMPLETADO en el historial para que el LLM no re-intente
                    self._history.append(
                        f"⚠️ BÚSQUEDA YA EJECUTADA. q='{_query}' en URL. "
                        f"NO vuelvas a rellenar ni a pulsar Enter. Observa los resultados."
                    )
                    self._needs_vision = True
                    self._last_vision_url = ""   # Forzar screenshot fresco en el siguiente paso

                    # El Shield ejecuta DOS pasos del plan (fill + press Enter).
                    # Avanzamos el plan dos veces para reflejar ambos pasos consumidos.
                    self._plan_step_iters = max(self._plan_step_iters, 1)
                    _done1 = self._try_advance_plan_step()   # fill → siguiente
                    if _done1:
                        await self._log(f"🏁 Búsqueda completada (fill era último paso). URL: {url_after}")
                        return await self._auto_done(url_after, result=f"Búsqueda '{_query}' completada")
                    # Segundo avance: press Enter → siguiente
                    self._plan_step_iters = 1
                    _done2 = self._try_advance_plan_step()
                    if _done2:
                        await self._log(f"🏁 Búsqueda completada (Enter era último paso). URL: {url_after}")
                        return await self._auto_done(url_after, result=f"Búsqueda '{_query}' completada")
                    return {"stop": False}

                if kind == "click_at" or (x is not None and y is not None and not selector):
                    await self._browser.click_at(int(x), int(y))
                    self._history.append(f"✅ click_at ({x},{y})")
                elif selector:
                    resolved = self._resolve_selector(selector)
                    try:
                        await self._browser.click(resolved, description=desc)
                        self._history.append(f"✅ click '{resolved}'")
                    except Exception as e:
                        self._failed_selectors.add(selector)
                        self._last_action_failed = True
                        # 1. Fallback por coordenadas del mapa
                        fb = self._find_coord_fallback(desc + " " + selector)
                        if fb:
                            lbl, cx, cy = fb
                            await self._browser.click_at(cx, cy)
                            self._history.append(f"✅ click_at fallback coord '{lbl}'")
                            self._last_action_failed = False
                        else:
                            # 2. Self-healer para clicks de acción (apply, submit, etc.)
                            healed = False
                            if self._healer:
                                winner = await self._healer.heal_click(
                                    failed_selector=selector, desc=desc,
                                    page_map=self._page_map, url=url_before, log_fn=self._log
                                )
                                if winner:
                                    self._history.append(f"✅ click [healed] '{winner}'")
                                    healed = True
                                    self._last_action_failed = False
                            if not healed:
                                self._history.append(f"❌ click FALLÓ '{selector}': {str(e)[:60]}")
                                raise
                else:
                    fb = self._find_coord_fallback(desc)
                    if fb:
                        lbl, cx, cy = fb
                        await self._browser.click_at(cx, cy)
                        self._history.append(f"✅ click_at coord '{lbl}'")
                    else:
                        return {"stop": False}

                await asyncio.sleep(1.2)
                url_after = await self._browser.get_url()
                if url_after != url_before:
                    self._history.append(f"   → {url_after}")
                    self._maybe_advance_phase(await self._browser.get_text_content(), url_after)
                    if self._try_advance_plan_step():
                        await self._log(f"🏁 Plan completado tras click+navigate → {url_after}")
                        return await self._auto_done(url_after)

            elif kind in ("fill", "type"):
                selector = action.get("selector") or "input:not([type='hidden'])"
                value    = action.get("value") or action.get("text") or ""
                resolved = self._resolve_selector(selector, force_fillable=True)

                # Intentar primero con el selector del LLM
                _fill_ok = False
                try:
                    if kind == "fill":
                        await self._browser.fill(resolved, value, description=desc)
                    else:
                        await self._browser.type_char_by_char(resolved, value)
                    self._history.append(f"✅ {kind} '{desc}'='{value[:40]}'")
                    _fill_ok = True
                except Exception as _fill_err:
                    self._failed_selectors.add(selector)
                    self._last_action_failed = True
                    await self._log(f"⚠️ {kind} falló con '{resolved}': {str(_fill_err)[:60]}")

                    # Activar self-healer
                    if self._healer:
                        winner = await self._healer.heal_fill(
                            failed_selector=selector, value=value, desc=desc,
                            page_map=self._page_map, url=url_before, log_fn=self._log
                        )
                        if winner:
                            self._history.append(f"✅ {kind} [healed] '{winner}'='{value[:40]}'")
                            _fill_ok = True
                            self._last_action_failed = False
                    if not _fill_ok:
                        self._history.append(f"❌ {kind} FALLÓ: '{selector}'")
                        raise _fill_err
                await asyncio.sleep(0.5)

            elif kind == "press":
                key = action.get("key", "Enter")
                await self._browser.press(key)
                self._history.append(f"✅ press {key}")
                await asyncio.sleep(0.8)
                url_after = await self._browser.get_url()
                if url_after != url_before:
                    self._history.append(f"   → {url_after}")
                    if self._try_advance_plan_step():
                        await self._log("🏁 Plan completado tras press+URL change")
                        return await self._auto_done(url_after)

            elif kind == "scroll":
                direction = action.get("direction", "down")
                amount    = int(action.get("amount", 500))
                delta     = amount if direction == "down" else -amount
                await self._browser.page.evaluate(f"window.scrollBy(0, {delta})")
                self._history.append(f"✅ scroll {direction} {amount}px")
                await asyncio.sleep(0.5)

            elif kind == "wait":
                ms = max(200, min(int(action.get("ms", 1000)), 10000))
                await asyncio.sleep(ms / 1000)
                self._history.append(f"✅ wait {ms}ms")

            elif kind == "extract":
                selector = action.get("selector", "body")
                text     = await self._browser.extract(selector)
                note     = f"extract[{desc}]: {text[:200]}"
                self._scratchpad = (self._scratchpad + " | " + note).strip(" | ")[-400:]
                self._history.append(f"✅ extract '{selector}' → '{text[:60]}'")
                await self._log(f"📤 Extract: {text[:200]}")
                if self._try_advance_plan_step():
                    await self._log("🏁 Plan completado tras extract")
                    return await self._auto_done(await self._browser.get_url(), result=text[:300])

            elif kind == "note":
                text = action.get("text", "")
                self._scratchpad = (self._scratchpad + " | " + text).strip(" | ")[-400:]
                self._history.append(f"✅ note: {text[:60]}")
                await self._log(f"📝 Nota: {text}")

            elif kind == "done":
                job_url = action.get("job_url", "")
                company = action.get("company", "")

                # Detectar si es una tarea laboral real (necesita validación estricta)
                # o una tarea genérica (búsqueda, test, extracción de datos)
                _job_task_keywords = ["linkedin", "tecnoempleo", "infojobs", "solicitud", "candidatura", "inscribirme", "easy apply"]
                _is_job_task = any(w in self._task.lower() for w in _job_task_keywords)

                if _is_job_task:
                    # Tareas laborales: exigen job_url real y fase de envío
                    if not job_url or job_url in ("N/A", "<url exacta>", ""):
                        self._system_hint = "No puedes usar 'done' sin job_url real. Copia la URL exacta de la oferta."
                        self._history.append("⚠️ done rechazado: falta job_url")
                        return {"stop": False}
                    if self._phase not in (Phase.SUBMITTING, Phase.FILLING, Phase.APPLYING):
                        self._system_hint = f"Fase {self._phase.value}: confirma el envío antes de usar done."
                        self._history.append(f"⚠️ done rechazado: fase {self._phase.value}")
                        return {"stop": False}
                else:
                    # Tareas genéricas: aceptar done con job_url = URL actual si no se proporcionó
                    if not job_url or job_url in ("N/A", "<url exacta>", ""):
                        job_url = await self._browser.get_url()
                        action["job_url"] = job_url
                    if not company:
                        action["company"] = "N/A"

                await self._log(f"✅ Completado: {action.get('result','')}")
                self._phase = Phase.DONE
                return {"stop": True, "payload": action}

            elif kind == "error":
                await self._log(f"🛑 Error: {action.get('reason','')}")
                self._phase = Phase.ERROR
                return {"stop": True, "payload": action}

            elif kind == "human_intervention":
                reason = action.get("reason", "")
                await self._log(f"📢 INTERVENCIÓN: {reason}")
                await self._send("action", {"type": "human_intervention", "reason": reason})
                await asyncio.sleep(30.0)
                return {"stop": False}

            elif kind == "human_ask":
                # Charlie pide ayuda al usuario y espera su respuesta
                question = action.get("question", "")
                context  = action.get("context", "")
                await self._log(f"🙋 Charlie pregunta: {question}")
                await self._send("human_ask", {
                    "question": question,
                    "context":  context,
                    "step":     self._step,
                })
                # Esperar respuesta con timeout de 120 s
                try:
                    answer = await asyncio.wait_for(self._answer_queue.get(), timeout=120.0)
                    await self._log(f"💬 Respuesta del usuario: {answer}")
                    self._history.append(f"✅ human_ask: '{question}' → '{answer}'")
                    self._scratchpad = (self._scratchpad + f" | Usuario dijo: {answer}").strip(" | ")[-400:]
                    # Persistir en memoria si está disponible
                    if self._memory:
                        self._memory.learn_from_answer(question, answer, context=context)
                        await self._log("💾 Respuesta guardada en memoria.")
                except asyncio.TimeoutError:
                    await self._log("⏰ Sin respuesta del usuario (120 s). Continuando.")
                    self._history.append("⚠️ human_ask: timeout sin respuesta")
                return {"stop": False}

            else:
                await self._log(f"⚠️ Acción desconocida: {kind}")

        except Exception as e:
            await self._log(f"⚠️ Error en '{kind}': {e}")
            if "Timeout" in str(e) and action.get("selector"):
                self._failed_selectors.add(action["selector"])
            self._history.append(f"❌ '{kind}' falló: {str(e)[:80]}")
            self._last_action_failed = True

        return {"stop": False}

    # ── Gestión del plan ───────────────────────────────────────────────────────

    def _current_plan_step(self) -> str:
        if self._plan and self._plan_idx < len(self._plan):
            return self._plan[self._plan_idx]
        return ""

    async def _auto_done(self, url: str, result: str = "") -> dict:
        """
        Genera un payload done automático cuando el plan se completa sin
        que el LLM haya emitido done explícitamente.
        Para tareas laborales sigue requiriendo validación manual del LLM —
        este método solo se usa para tareas genéricas (búsquedas, tests, etc.)
        """
        _job_keywords = ["linkedin", "tecnoempleo", "infojobs", "solicitud", "candidatura"]
        _is_job = any(w in self._task.lower() for w in _job_keywords)
        if _is_job:
            # No auto-done en tareas laborales: el LLM debe confirmar el envío
            self._system_hint = (
                "El plan indica que hemos terminado. "
                "Si la candidatura fue enviada, usa 'done' con job_url y company reales."
            )
            return {"stop": False}

        # Para tareas genéricas: cerrar limpiamente
        summary = result or f"Tarea completada. URL final: {url}"
        self._phase = Phase.DONE
        await self._log(f"✅ Auto-done: {summary[:120]}")
        return {"stop": True, "payload": {
            "action":  "done",
            "result":  summary,
            "job_url": url,
            "company": "N/A",
        }}

    def _try_advance_plan_step(self) -> bool:
        """
        Avanza al siguiente paso del plan si es posible.
        Devuelve True cuando el plan está COMPLETO (señal de done automático).
        """
        if not self._plan or self._plan_step_iters < 1:
            return False
        # Último paso ya ejecutado → plan terminado
        if self._plan_idx >= len(self._plan) - 1:
            self._history.append(f"🏁 Plan completado ({len(self._plan)} pasos)")
            return True
        # Avanzar al siguiente
        self._plan_idx       += 1
        self._plan_step_iters = 0
        self._history.append(f"📌 Plan → paso {self._plan_idx + 1}: {self._plan[self._plan_idx]}")
        return False

    async def _replan(self, url: str, login_status: str):
        if not self._planner:
            return
        self._replan_count += 1
        context = (
            f"Completados: {self._plan[:self._plan_idx]}. "
            f"Atascado en: '{self._current_plan_step()}' tras {self._plan_step_iters} pasos. "
            f"Historial: {self._history[-4:]}"
        )
        await self._log(f"🔁 Replanificando ({self._replan_count}/2)...")
        new_plan, _ = await self._planner.generate_plan(
            self._task, url, login_status, failure_context=context
        )
        self._plan, self._plan_idx, self._plan_step_iters = new_plan, 0, 0
        self._temperature = min(self._temperature + 0.1, 0.7)
        await self._log(f"📋 Nuevo plan ({len(new_plan)} pasos): {new_plan}")
        self._history.append(f"🔁 Replan: {new_plan}")

    async def _infer_plan_step(self, url: str, page_text: str, page_map: str) -> int | None:
        """
        Llamada LLM ligera (sin screenshot) que determina en qué paso del plan
        se encuentra realmente el agente según el contenido visible.

        Devuelve el índice del paso (0-based) o None si no puede determinarlo.
        Es rápida porque no incluye imagen y usa un contexto mínimo.
        """
        if not self._plan or len(self._plan) <= 1:
            return None

        plan_str = "\n".join(f"{i}. {p}" for i, p in enumerate(self._plan))

        # Señales semánticas para anclar la inferencia
        _recent_ok = [h for h in self._history if h.startswith(("✅", "📌", "🏁"))][-5:]
        _recent_str = " | ".join(_recent_ok) if _recent_ok else "ninguna"
        _has_filled  = any("fill" in h and "shield" in h for h in self._history[-10:])
        _has_entered = any("press Enter [shield]" in h for h in self._history[-10:])
        _url_results = "q=" in url.lower() or "/search" in url.lower()
        _hints = []
        if _has_filled:   _hints.append("ya se rellenó el campo de búsqueda")
        if _has_entered:  _hints.append("ya se pulsó Enter")
        if _url_results:  _hints.append("URL muestra resultados (q= en URL)")

        user_msg = f"""\
Plan actual:
{plan_str}

URL: {url}
Señales: {'; '.join(_hints) or 'ninguna'}
Acciones exitosas recientes: {_recent_str}

Mapa (primeras líneas):
{self._page_map[:500]}

Texto:
{page_text[:300]}

REGLA: Si ya se ejecutó fill+Enter y la URL tiene q= o /search, el agente está
en el último paso (o más allá). No retrocedas a pasos ya ejecutados.

¿Índice del paso actual (0-based)?"""

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": STEP_INFER_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 4096, "num_thread": 8},
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self._ollama}/api/chat", json=payload)
                if resp.status_code != 200:
                    return None
                content = resp.json().get("message", {}).get("content", "")
                parsed  = self._parse_json(content)
                if parsed is None:
                    return None
                idx = int(parsed.get("step_index", -1))
                if 0 <= idx < len(self._plan):
                    return idx
        except Exception:
            pass
        return None

    # ── Helpers de fase ────────────────────────────────────────────────────────

    def _maybe_advance_phase(self, page_text: str, url: str):
        text_low    = page_text.lower()
        transitions = {
            Phase.SEARCHING: Phase.SELECTING,
            Phase.SELECTING: Phase.APPLYING,
            Phase.APPLYING:  Phase.FILLING,
            Phase.FILLING:   Phase.SUBMITTING,
        }
        if self._phase in PHASE_ADVANCE_SIGNALS:
            if any(s in text_low for s in PHASE_ADVANCE_SIGNALS[self._phase]):
                new_phase = transitions[self._phase]
                self._phase = new_phase
                self._history.append(f"📌 Fase → {new_phase.value}")

    def _infer_phase_from_url(self, url: str) -> Phase:
        if self._phase in (Phase.DONE, Phase.ERROR): return self._phase
        if self._phase == Phase.INIT:                return Phase.SEARCHING
        return self._phase

    def _track_url(self, url: str):
        if url == self._last_url:
            self._same_url_count += 1
            if self._same_url_count >= 5 and not self._system_hint:
                self._system_hint = (
                    f"Llevas {self._same_url_count} pasos en la misma URL. "
                    "Prueba scroll, busca botones sin pulsar, o navega directo."
                )
        else:
            self._same_url_count = 0
        self._last_url = url

    def _extract_start_url(self) -> str | None:
        match = re.search(r'https?://[^\s\'"]+', self._task)
        if match: return match.group(0)
        tl = self._task.lower()
        if "tecnoempleo" in tl: return "https://www.tecnoempleo.com"
        if "infojobs"    in tl: return "https://www.infojobs.net"
        if "google"      in tl: return "https://www.google.com"
        return "https://www.linkedin.com/jobs/"

    # ── Helpers de selectores ─────────────────────────────────────────────────

    def _resolve_selector(self, selector: str, force_fillable: bool = False) -> str:
        """
        Normaliza el selector antes de enviarlo a Playwright:
        - Guarda contra selector vacío
        - text= y css= pasan sin tocar (nativos Playwright)
        - :contains(X) jQuery → :has-text(X) Playwright (Playwright no soporta :contains)
        - Descripciones completas como "[INDEX:13] textarea name='q' 'Buscar'" → extrae INDEX:13
        """
        if not selector:
            return selector

        # Nativos Playwright — pasar sin tocar
        if selector.startswith("text=") or selector.startswith("css="):
            return selector

        # :contains() no es CSS válido; Playwright usa :has-text()
        selector = re.sub(r':contains\(([^)]+)\)', r':has-text(\1)', selector)

        # Extraer INDEX:N aunque venga en medio de una descripción larga
        # El LLM a veces pega la descripción entera del elemento como selector
        match = re.search(r'INDEX:(\d+)', selector)
        if not match:
            return selector
        idx  = match.group(1)
        base = (
            ":is(input, textarea, [contenteditable='true'], select)"
            if force_fillable else
            ":is(button, a, input, select, [role='button'], [role='link'], [contenteditable='true'], textarea)"
        )
        return f"{base} >> nth={idx}"

    def _find_coord_fallback(self, hint: str) -> tuple[str, int, int] | None:
        if not hint: return None
        words = [w.lower() for w in re.split(r'\W+', hint) if len(w) > 2]
        for label, (cx, cy) in self._coordinate_memory.items():
            if any(w in label for w in words):
                return label, cx, cy
        return None

    # ── Memoria de coordenadas ─────────────────────────────────────────────────

    def _update_coordinate_memory(self, elements_str: str):
        if not elements_str or "Error" in elements_str:
            return
        self._coordinate_memory = {}
        for item in elements_str.split(" ||| "):
            item      = item.strip()
            m_label   = re.search(r'"([^"]+)"\s*@\s*\((\d+),\s*(\d+)\)', item)
            m_name    = re.search(r'name="([^"]+)"',                      item)
            m_coords  = re.search(r'@\s*\((\d+),\s*(\d+)\)',              item)
            if m_label:
                self._coordinate_memory[m_label.group(1).lower()] = (
                    int(m_label.group(2)), int(m_label.group(3))
                )
            if m_name and m_coords:
                self._coordinate_memory[m_name.group(1).lower()] = (
                    int(m_coords.group(1)), int(m_coords.group(2))
                )

    # ── Parse JSON ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':   depth += 1
                elif text[i] == '}': depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: i + 1])
                    except Exception:
                        break
        return None

    # ── Log ──────────────────────────────────────────────────────────────────

    async def _log(self, type_or_msg: str, msg: str = None):
        try:
            m_type  = "log"  if msg is None else type_or_msg
            content = type_or_msg if msg is None else msg
            await self._send(m_type, content)
        except Exception:
            pass
