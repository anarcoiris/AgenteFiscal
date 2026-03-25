"""
charlie_planner.py — Planificador de alto nivel para Charlie.

Separa el "qué hay que hacer" (Planificador) del "cómo ejecutar cada paso" (Ejecutor = CharlieAgent).

Arquitectura Plan-and-Act:
  1. El Planificador recibe el objetivo + estado inicial y genera una lista de
     3-5 pasos abstractos (e.g. "Navegar a LinkedIn", "Buscar ofertas de sysadmin").
  2. CharlieAgent recibe esa lista y ejecuta cada paso en orden, acción a acción.
  3. Si un paso falla repetidamente, el Planificador regenera el plan desde el
     estado actual (replanning dinámico).

Por qué funciona mejor con modelos pequeños (3B):
  - El planificador hace UNA sola llamada para razonar a alto nivel, con contexto mínimo.
  - El ejecutor hace llamadas cortas con el sub-paso actual, sin necesidad de recordar
    el objetivo global ni el historial largo.
  - Divide la carga cognitiva: ninguna llamada individual satura el modelo.
"""
import json
import re

import httpx


# ── Prompt del planificador ────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
Eres el Planificador de Charlie, un agente web autónomo.
Tu único trabajo es descomponer una tarea en una lista ordenada de pasos concretos.

Responde SOLO con un JSON válido:
{
  "plan": ["<paso 1>", "<paso 2>", ...],
  "start_url": "<URL donde debe empezar el agente>",
  "reason": "<por qué este plan>"
}

REGLAS:
- Máximo 6 pasos. Mínimo 2.
- Cada paso es una instrucción corta y ejecutable (verbo + objeto).
  Ejemplos válidos:
    "Navegar a google.com"
    "Rellenar el campo de búsqueda con el texto indicado"
    "Pulsar Enter para buscar"
    "Hacer clic en el primer resultado relevante"
    "Pulsar el botón Solicitud sencilla o Easy Apply"
    "Rellenar el campo salario con el valor indicado"
    "Confirmar el envío de la candidatura"
- start_url: la URL exacta donde debe arrancar el agente. Si la tarea menciona una URL, úsala.
  Si la tarea menciona Google, usa "https://www.google.com".
  Si no hay URL clara, usa "about:blank".
- NUNCA incluyas "error" como paso del plan. Los pasos son siempre acciones positivas.
- NUNCA incluyas pasos de verificación de login — el ejecutor lo maneja solo.
- NO incluyas pasos de navegación si la tarea ya da la URL de inicio.
- Responde SOLO con JSON. Nada más.
"""


class CharliePlanner:
    """
    Genera un plan de alto nivel a partir de la tarea y el estado actual.
    Puede replanificar si el ejecutor reporta que está atascado.
    """

    def __init__(self, ollama_url: str, model: str):
        self._ollama = ollama_url.rstrip("/")
        self._model  = model

    async def generate_plan(
        self,
        task: str,
        current_url: str,
        login_status: str,
        failure_context: str = "",
    ) -> tuple[list[str], str]:
        """
        Genera (o regenera) un plan.

        Args:
            task:            La tarea completa tal como la recibió el dashboard.
            current_url:     URL actual del navegador.
            login_status:    Resultado de check_login_status().
            failure_context: Descripción de por qué el plan anterior falló (para replan).

        Returns:
            (lista_de_pasos, start_url)
            Si la llamada falla, devuelve un plan de emergencia mínimo.
        """
        replan_block = ""
        if failure_context:
            replan_block = f"\n⚠️ REPLANIFICACIÓN: El plan anterior falló. Motivo: {failure_context}\nURL actual: {current_url}\nGenera un plan alternativo."

        user_msg = f"""\
TAREA:
{task}

ESTADO ACTUAL:
- URL: {current_url}
- Login: {login_status}
{replan_block}

Genera el plan."""

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system",  "content": PLANNER_SYSTEM},
                {"role": "user",    "content": user_msg},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.2,   # Un poco más alto que el ejecutor: permite creatividad táctica
                "num_ctx":     8192,  # El planificador no necesita contexto de screenshot
                "top_p":       0.5,
                "num_thread":  8,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self._ollama}/api/chat", json=payload)
                if resp.status_code != 200:
                    return self._fallback_plan(task, current_url), current_url

                raw     = resp.json().get("message", {}).get("content", "")
                parsed  = self._parse_json(raw)
                if not parsed:
                    return self._fallback_plan(task, current_url), current_url

                plan      = parsed.get("plan", [])
                start_url = parsed.get("start_url", current_url).strip()

                # Validación mínima
                if not isinstance(plan, list) or len(plan) == 0:
                    return self._fallback_plan(task, current_url), current_url

                # Limpiar pasos vacíos
                plan = [str(p).strip() for p in plan if str(p).strip()]
                return plan, start_url

        except Exception:
            return self._fallback_plan(task, current_url), current_url

    def _fallback_plan(self, task: str, url: str) -> list[str]:
        """
        Plan de emergencia mínimo cuando el LLM no responde o devuelve JSON inválido.
        Intenta inferir los pasos más probables a partir de palabras clave en la tarea.
        """
        steps: list[str] = []
        task_low = task.lower()

        # Detectar URL de inicio
        match = re.search(r'https?://[^\s\'"]+', task)
        if match:
            steps.append(f"Navegar a {match.group(0)}")

        if "linkedin" in task_low or "tecnoempleo" in task_low or "infojobs" in task_low:
            steps += [
                "Verificar que estás logueado en el portal",
                "Buscar la primera oferta relevante en la lista",
                "Abrir el detalle de la oferta haciendo clic en ella",
                "Pulsar el botón de solicitud (Solicitud sencilla / Inscribirme)",
                "Rellenar los campos del formulario con los datos de la tarea",
                "Confirmar el envío de la candidatura",
            ]
        elif "google" in task_low:
            steps += [
                "Navegar a https://www.google.com",
                "Rellenar el campo de búsqueda con la consulta",
                "Pulsar Enter para buscar",
                "Hacer clic en el primer resultado relevante",
            ]
        else:
            steps += [
                "Navegar a la URL indicada en la tarea",
                "Completar la tarea según las instrucciones",
            ]

        return steps

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
