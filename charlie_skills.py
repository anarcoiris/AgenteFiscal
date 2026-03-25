import json
import os
from pathlib import Path

class CharlieSkillManager:
    """
    Gestiona las configuraciones de bots (Skills) en formato JSON.
    Permite guardar secuencias exitosas generadas por la IA y cargarlas para el modo Bot rápido.
    """
    DEFAULT_PATH = Path(os.getenv(
        "CHARLIE_SKILLS_PATH",
        r"C:\Users\soyko\Documents\LaAgencia\charlie_skills.json"
    ))

    def __init__(self, path: Path | None = None):
        self._path = path or self.DEFAULT_PATH
        self._skills: dict = {}
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._skills = raw if isinstance(raw, dict) else {}
        except Exception:
            self._skills = {}

    def save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._skills, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"⚠️ charlie_skills: no se pudo guardar: {e}")

    def get_skill(self, task_signature: str) -> list[dict] | None:
        """Devuelve la secuencia de acciones para una tarea conocida, si existe."""
        return self._skills.get(task_signature)

    def save_skill(self, task_signature: str, sequence: list[dict]):
        """
        Guarda una secuencia de acciones como un bot predefinido.
        Filtra acciones innecesarias como 'wait', 'human_ask' o 'extract' (si el bot no las necesita),
        o simplemente las guarda todas limpiando ruido.
        """
        clean_sequence = []
        for step in sequence:
            action_type = step.get("action")
            if action_type in ("navigate", "click", "fill", "press", "done"):
                clean_sequence.append({
                    "action": action_type,
                    "url": step.get("url"),
                    "selector": step.get("selector"),
                    "value": step.get("value"),
                    "key": step.get("key"),
                })
        
        self._skills[task_signature] = clean_sequence
        self.save()
        print(f"💾 Skill guardada para signature: {task_signature} ({len(clean_sequence)} pasos)")

