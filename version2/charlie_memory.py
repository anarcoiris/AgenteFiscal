"""
charlie_memory.py — Sistema de memoria persistente para Charlie.

Tres capas de memoria:
  1. Memoria a largo plazo (JSON en disco): aprende de las respuestas del usuario.
     Se carga al arrancar y se inyecta en el prompt como contexto.
  2. Memoria de sesión: hechos aprendidos en la sesión actual (se pierde al cerrar).
  3. Memoria de portal: patrones específicos por sitio web (selectores que funcionan,
     campos que el portal tiene, mensajes de confirmación conocidos).

El agente puede:
  - Consultar la memoria antes de cada paso (recall).
  - Guardar respuestas del usuario como aprendizajes nuevos (learn).
  - Obtener un bloque de texto listo para inyectar en el prompt (prompt_block).
"""
import json
import os
import time
from pathlib import Path


class CharlieMemory:
    """
    Gestiona la memoria persistente y de sesión de Charlie.
    """

    # Ruta por defecto (override con env CHARLIE_MEMORY_PATH)
    DEFAULT_PATH = Path(os.getenv(
        "CHARLIE_MEMORY_PATH",
        r"C:\Users\soyko\Documents\LaAgencia\charlie_memory.json"
    ))

    def __init__(self, path: Path | None = None):
        self._path: Path           = path or self.DEFAULT_PATH
        self._long_term: dict      = {}   # Persiste en disco
        self._session: list[str]   = []   # Solo esta sesión
        self._load()

    # ── Carga y guardado ──────────────────────────────────────────────────────

    def _load(self):
        """Carga la memoria desde disco. Si no existe, arranca vacía."""
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._long_term = raw if isinstance(raw, dict) else {}
        except Exception:
            self._long_term = {}

    def save(self):
        """Persiste la memoria a disco."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._long_term, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"⚠️ charlie_memory: no se pudo guardar: {e}")

    # ── Aprender ──────────────────────────────────────────────────────────────

    def learn(self, category: str, key: str, value: str, source: str = "user"):
        """
        Guarda un aprendizaje en la memoria a largo plazo.

        Args:
            category: Categoría temática (portal, acción, campo, general).
            key:      Identificador del hecho (p.ej. "linkedin.easy_apply_selector").
            value:    Lo que se aprendió (p.ej. "button[aria-label*='Easy Apply']").
            source:   Quién lo aprendió ('user', 'agent', 'auto').
        """
        if category not in self._long_term:
            self._long_term[category] = {}
        self._long_term[category][key] = {
            "value":   value,
            "source":  source,
            "ts":      int(time.time()),
        }
        self.save()

    def learn_from_answer(self, question: str, answer: str, context: str = ""):
        """
        Registra una pregunta hecha al usuario y su respuesta.
        Las categoriza automáticamente según palabras clave.
        """
        category = "general"
        q_low = question.lower()
        for portal in ["linkedin", "tecnoempleo", "infojobs", "google"]:
            if portal in q_low:
                category = portal
                break
        if any(w in q_low for w in ["selector", "botón", "campo", "input"]):
            category = category + ".ui"

        key = question[:60].strip().replace(" ", "_").lower()
        self.learn(category, key, answer, source="user")
        self._session.append(f"[Aprendido] {question} → {answer}")

    def remember_session(self, fact: str):
        """Guarda un hecho en la memoria de sesión (no persiste)."""
        self._session.append(fact)

    # ── Consultar ────────────────────────────────────────────────────────────

    def recall(self, context: str, max_items: int = 6) -> list[str]:
        """
        Devuelve los recuerdos más relevantes dado un contexto.
        Hace matching por palabras clave entre el contexto y las claves guardadas.
        """
        context_words = set(context.lower().split())
        scored: list[tuple[int, str]] = []

        for category, entries in self._long_term.items():
            for key, entry in entries.items():
                # Puntuación simple: palabras del contexto que aparecen en la clave
                key_words  = set(key.lower().replace("_", " ").split())
                cat_words  = set(category.lower().split("."))
                score      = len(context_words & (key_words | cat_words))
                if score > 0:
                    text = f"[{category}] {key.replace('_',' ')}: {entry['value']}"
                    scored.append((score, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:max_items]]

    def prompt_block(self, context: str = "") -> str:
        """
        Devuelve un bloque de texto listo para inyectar en el prompt del LLM.
        Combina memoria a largo plazo relevante + hechos de la sesión actual.
        """
        parts: list[str] = []

        relevant = self.recall(context, max_items=5)
        if relevant:
            parts.append("MEMORIA (aprendizajes previos):\n" + "\n".join(f"  • {r}" for r in relevant))

        if self._session:
            recent = self._session[-4:]  # Solo los últimos 4 de sesión
            parts.append("SESIÓN ACTUAL:\n" + "\n".join(f"  • {s}" for s in recent))

        return "\n".join(parts) if parts else ""

    # ── Introspección ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Devuelve un resumen de cuánto sabe Charlie."""
        total = sum(len(v) for v in self._long_term.values())
        return {
            "categorias":      list(self._long_term.keys()),
            "total_entradas":  total,
            "sesion_actual":   len(self._session),
            "ruta":            str(self._path),
        }

    def all_entries(self) -> list[dict]:
        """Devuelve todas las entradas para mostrarlas en el dashboard."""
        rows = []
        for category, entries in self._long_term.items():
            for key, entry in entries.items():
                rows.append({
                    "category": category,
                    "key":      key,
                    "value":    entry.get("value", ""),
                    "source":   entry.get("source", ""),
                    "ts":       entry.get("ts", 0),
                })
        return sorted(rows, key=lambda r: r["ts"], reverse=True)
