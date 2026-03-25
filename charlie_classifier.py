import json
import re
import httpx
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class CharlieClassifier:
    """
    Motor de clasificación (Fase 3).
    No resuelve tareas. Clasifica:
    - task family (e.g. job_search, coding, info_retrieval, system_config)
    - env: browser, desktop, hybrid
    - ui_atoms: lista de elementos esperados (search, filter, apply_button)
    - skill_candidate: posible skill a usar
    """

    def __init__(self, ollama_url: str, model: str):
        self._ollama = ollama_url.rstrip("/")
        self._model = model

    async def classify_task(self, task: str) -> Dict[str, Any]:
        """Clasifica la tarea inicial del usuario."""
        system_prompt = """Eres el Clasificador del Agente Antigravity (Charlie).
Tu objetivo es analizar la solicitud del usuario y devolver un JSON estricto con la siguiente estructura:
{
    "task_family": "<ejemplo: job_search, code_edit, data_extraction, system_config, web_navigation>",
    "environment": "<browser, desktop, o hybrid>",
    "expected_ui_atoms": ["<atom1>", "<atom2>"],
    "risk_level": "<low, medium, high>"
}

Reglas:
1. environment = 'browser' si solo requiere web.
2. environment = 'desktop' si requiere apps locales (IDE, terminal, archivos locales).
3. environment = 'hybrid' si requiere buscar información en web y luego usar apps locales (o viceversa).
4. risk_level = 'high' si involucra borrar archivos, enviar emails a terceros reales sin revisión, etc. De lo contrario 'low' o 'medium'.
Responde SOLO con JSON válido."""

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Tarea: {task}"}
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096}
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self._ollama}/api/chat", json=payload)
                if resp.status_code == 200:
                    content = resp.json().get("message", {}).get("content", "")
                    return self._parse_json(content)
        except Exception as e:
            print(f"⚠️ Error en clasificador: {e}")
        
        return self._fallback_classification(task)

    def _fallback_classification(self, task: str) -> Dict[str, Any]:
        t = task.lower()
        if any(w in t for w in ["linkedin", "infojobs", "tecnoempleo", "oferta", "trabajo"]):
            return {"task_family": "job_search", "environment": "browser", "expected_ui_atoms": ["search", "filter", "apply_button"], "risk_level": "medium"}
        return {"task_family": "general", "environment": "browser", "expected_ui_atoms": [], "risk_level": "low"}

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except Exception:
            return {}
