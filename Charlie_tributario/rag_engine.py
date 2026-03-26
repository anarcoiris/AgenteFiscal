"""
rag_engine.py — Motor RAG reutilizable para el proyecto tributario.

v3 — Mejoras:
  - GPT-4.1 como motor primario (OpenAI) con local Ollama como contraste asíncrono
  - Caché con retroalimentación de calidad (good/bad) — las malas no se sirven
  - Contraste local: detecta discrepancias entre OpenAI y modelo local
  - Embedding multilingüe (BAAI/bge-m3) para español
  - Indicador de confianza basado en distancia FAISS
  - Timeout Ollama extendido a 300s para DeepSeek R1
  - OCR vía pytesseract (CPU, sin VRAM)
  - Metadata de páginas en las fuentes
"""
import os
import re
import hashlib
import pickle
import shelve
import threading
import uuid
import faiss
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True

    # Configuración automática de Tesseract en Windows si no está en el PATH
    if os.name == 'nt':
        common_tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Tesseract-OCR', 'tesseract.exe'),
        ]
        # Solo lo configuramos si el comando actual no apunta a un ejecutable válido
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            for path in common_tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Umbral de distancia FAISS para indicador de confianza
CONFIDENCE_HIGH = 1.0     # distancia media < 1.0 → alta confianza
CONFIDENCE_MEDIUM = 2.0   # distancia media < 2.0 → confianza media


class RAGEngine:
    """Motor RAG: GPT-4.1 (OpenAI) primario + Ollama local como contraste asíncrono."""

    def __init__(
        self,
        index_path="faiss_index.bin",
        chunks_path="chunks.pkl",
        cache_path="rag_cache",
        embedding_model="BAAI/bge-m3",
        ollama_url="http://localhost:11434",
        llm_model="qwen2.5:3b",
        openai_model="gpt-4.1",
        top_k=7,
        on_status=None,
    ):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.cache_path = cache_path
        self.embedding_model_name = embedding_model
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.openai_model = openai_model
        self.top_k = top_k
        self._on_status = on_status

        self._embedding_model = None
        self._index = None
        self._chunks = None
        self._loaded = False
        self._history = []  # Últimas N interacciones para contexto conversacional

        # Almacén de contrastes asíncronos: { contrast_id: { status, local_answer, discrepancy, ... } }
        self._contrasts = {}
        self._contrast_lock = threading.Lock()

    # ── Status callback ─────────────────────────────────────────────────────
    def _status(self, msg: str):
        if self._on_status:
            try:
                self._on_status(msg)
            except Exception:
                pass  # Evitar crash si el callback falla (ej. ventana cerrada)

    # ── Load ────────────────────────────────────────────────────────────────
    def load(self):
        """Carga el modelo de embeddings, el índice FAISS y los chunks."""
        if self._loaded:
            return

        self._status("Cargando modelo de embeddings multilingüe...")
        self._embedding_model = SentenceTransformer(self.embedding_model_name)

        self._status("Cargando índice FAISS...")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Índice FAISS no encontrado: {self.index_path}")
        self._index = faiss.read_index(self.index_path)

        self._status("Cargando chunks...")
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks no encontrados: {self.chunks_path}")
        with open(self.chunks_path, "rb") as f:
            self._chunks = pickle.load(f)

        self._loaded = True
        self._status(f"RAG listo — {len(self._chunks)} chunks, {self._index.ntotal} vectores")

    # ── Query (texto → respuesta) ───────────────────────────────────────────
    def query(self, question: str, debug: bool = False, use_model: str = None) -> dict:
        """
        Pipeline RAG completo con caché e indicador de confianza.
        GPT-4.1 es el motor primario. Local se lanza como contraste asíncrono.

        use_model: fuerza un motor específico:
          - "gpt-4.1" / "gpt-4.1-mini"  → OpenAI directo
          - "local:xxx"                  → Ollama directo (sin contraste)
          - None (default)               → OpenAI primario + contraste

        Retorna: {
            "answer": str,
            "sources": list[dict],
            "confidence": "alta"|"media"|"baja",
            "cached": bool,
            "model_used": str,
            "contrast_id": str|None,
            "debug_chunks": list[str] (solo si debug=True),
            "error": str|None
        }
        """
        if not self._loaded:
            self.load()

        # 1. Comprobar caché (skip si quality == bad)
        cache_key = self._cache_key(question)
        cached = self._cache_get(cache_key)
        if cached:
            self._status("⚡ Respuesta desde caché")
            cached["cached"] = True
            return cached

        # 2. Buscar contexto
        self._status("Buscando contexto relevante...")
        query_emb = self._embedding_model.encode([question]).astype("float32")
        distances, indices = self._index.search(query_emb, self.top_k)

        sources = []
        context_parts = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._chunks):
                chunk = self._chunks[idx]
                fuente = chunk.get("fuente", "Desconocida")
                paginas = chunk.get("paginas", "")
                texto = chunk["texto"]
                ref = f"{fuente} ({paginas})" if paginas else fuente
                sources.append({
                    "fuente": fuente,
                    "paginas": paginas,
                    "texto": texto,
                    "distancia": float(distances[0][i])
                })
                context_parts.append(f"[Fuente: {ref}]\n{texto}")

        # 3. Calcular confianza
        avg_dist = np.mean(distances[0]) if len(distances[0]) > 0 else 999
        if avg_dist < CONFIDENCE_HIGH:
            confidence = "alta"
        elif avg_dist < CONFIDENCE_MEDIUM:
            confidence = "media"
        else:
            confidence = "baja"

        # 4. Construir prompt con historial
        prompt = self._build_prompt(question, context_parts)

        # 5. Generar respuesta según motor seleccionado
        self._status(f"Generando respuesta (confianza: {confidence})...")
        answer, model_used = self._call_llm_primary(prompt, use_model)

        if answer is None:
            return {
                "answer": "Error: no se pudo obtener respuesta del modelo.",
                "sources": sources,
                "confidence": confidence,
                "cached": False,
                "model_used": "N/A",
                "contrast_id": None,
                "error": "LLM unavailable"
            }

        # 6. Lanzar contraste local asíncrono (solo si la respuesta principal fue OpenAI)
        contrast_id = None
        is_openai = model_used.startswith("GPT")
        if is_openai:
            contrast_id = self._start_contrast(prompt, answer, question)

        result = {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "cached": False,
            "model_used": model_used,
            "contrast_id": contrast_id,
            "error": None
        }

        # Debug: incluir chunks recuperados
        if debug:
            result["debug_chunks"] = context_parts

        # 7. Guardar en caché (quality=None = sin evaluar)
        self._cache_set(cache_key, result)

        # 8. Actualizar historial
        self._history.append({"q": question, "a": answer[:200]})
        if len(self._history) > 3:
            self._history.pop(0)

        return result

    # ── OCR (imagen → texto) ────────────────────────────────────────────────
    def image_to_text(self, pil_image) -> str:
        """Extrae texto de imagen PIL vía pytesseract (CPU, sin VRAM)."""
        if not TESSERACT_AVAILABLE:
            return "[Error: librería pytesseract no disponible en Python]"

        self._status("Extrayendo texto de la captura (OCR)...")
        try:
            # Intentar con español primero, fallback a solo inglés
            try:
                text = pytesseract.image_to_string(pil_image, lang="spa+eng")
            except Exception:
                text = pytesseract.image_to_string(pil_image, lang="eng")

            text = text.strip()
            if not text:
                return "[No se pudo extraer texto de la captura]"
            return text
        except Exception as e:
            err = str(e).lower()
            if "tesseract is not installed" in err or "not in your path" in err:
                return ("[Error: Tesseract-OCR no detectado]. "
                        "Descárgalo aquí: https://github.com/UB-Mannheim/tesseract/wiki "
                        "e instálalo marcando 'Additional language data (Spanish)'.")
            return f"[Error OCR: {e}]"

    # ── Prompt builder ──────────────────────────────────────────────────────
    def _build_prompt(self, question: str, context_parts: list) -> str:
        context_text = "\n\n".join(context_parts)

        # Incluir historial si hay
        history_text = ""
        if self._history:
            history_lines = []
            for h in self._history[-3:]:
                history_lines.append(f"  P: {h['q']}\n  R: {h['a']}")
            history_text = f"\nHistorial de conversación reciente:\n" + "\n".join(history_lines) + "\n"

        return f"""Eres un asesor fiscal experto. Basa tu respuesta en el contexto proporcionado.

REGLAS:
1. Usa la información del contexto para responder. NO inventes datos, normativas ni supuestos que no se deriven EXPLÍCITAMENTE del contexto.
2. Si el contexto no tiene información para validar o descartar una afirmación, indica que no hay datos y NO asumas la respuesta.
3. Cita fuentes y páginas cuando sea posible (ej: "Según el Manual, p.45...").
4. Analiza paso a paso cada opción o punto de la pregunta basándote estrictamente en el texto antes de dar una conclusión final.
5. Responde en español usando el Euro (€).
6. Sé preciso y directo.
{history_text}
CONTEXTO:
{context_text}

PREGUNTA:
{question}
"""

    # ── LLM calls ───────────────────────────────────────────────────────────
    def _call_llm_primary(self, prompt: str, use_model: str = None) -> tuple[str | None, str]:
        """
        Motor primario: GPT-4.1 (OpenAI) por defecto.
        Si use_model empieza con 'local:', usa Ollama directo.
        Retorna (answer, model_used_label).
        """
        # Forzar modelo local si se solicita
        if use_model and use_model.startswith("local:"):
            local_model = use_model.split(":", 1)[1]
            old = self.llm_model
            self.llm_model = local_model
            self._status(f"Usando modelo local: {local_model}")
            answer = self._call_ollama(prompt)
            self.llm_model = old
            return (answer, f"{local_model} (Local)")

        # Determinar modelo OpenAI a usar
        openai_model = use_model if use_model and use_model.startswith("gpt") else self.openai_model
        api_key = os.environ.get("OPENAI_API_KEY")

        if api_key and OPENAI_AVAILABLE:
            self._status(f"Consultando {openai_model} (OpenAI)...")
            answer = self._call_openai(prompt, api_key, model=openai_model)
            if answer:
                label = f"GPT-4.1 (OpenAI)" if openai_model == "gpt-4.1" else f"GPT-4.1-mini (OpenAI)"
                return (answer, label)
            else:
                self._status("⚠️ OpenAI falló, intentando fallback local...")

        # Fallback a local si OpenAI no disponible
        self._status(f"Fallback: usando modelo local {self.llm_model}")
        answer = self._call_ollama(prompt)
        if answer:
            return (answer, f"{self.llm_model} (Local Fallback)")

        self._status("⚠️ Ningún modelo disponible")
        return (None, "N/A")

    def _call_ollama(self, prompt: str) -> str | None:
        url = f"{self.ollama_url}/api/generate"
        try:
            resp = requests.post(url, json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }, timeout=300)  # 300s para DeepSeek R1 chain-of-thought
            resp.raise_for_status()
            return resp.json().get("response")
        except Exception as e:
            self._status(f"⚠️ Error Ollama: {e}")
            return None

    def _call_openai(self, prompt: str, api_key: str, model: str = None) -> str | None:
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model or self.openai_model,
                messages=[
                    {"role": "system", "content": "Eres un asesor fiscal experto en el sistema tributario español. Responde siempre en español, usando el Euro (€)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return resp.choices[0].message.content
        except Exception as e:
            self._status(f"⚠️ Error OpenAI: {e}")
            return None

    # ── Contraste asíncrono ─────────────────────────────────────────────────
    def _start_contrast(self, prompt: str, openai_answer: str, question: str) -> str:
        """Lanza el modelo local en un hilo separado para contrastar."""
        contrast_id = str(uuid.uuid4())[:8]
        with self._contrast_lock:
            self._contrasts[contrast_id] = {
                "status": "running",
                "local_answer": None,
                "discrepancy": None,
                "question": question,
            }

        def _run_contrast():
            try:
                self._status(f"Contraste local en progreso ({self.llm_model})...")
                local_answer = self._call_ollama(prompt)
                discrepancy = self._detect_discrepancy(openai_answer, local_answer)
                with self._contrast_lock:
                    self._contrasts[contrast_id].update({
                        "status": "done",
                        "local_answer": local_answer,
                        "discrepancy": discrepancy,
                    })
            except Exception as e:
                with self._contrast_lock:
                    self._contrasts[contrast_id].update({
                        "status": "error",
                        "local_answer": None,
                        "discrepancy": {"differs": False, "reason": f"Error: {e}"},
                    })

        thread = threading.Thread(target=_run_contrast, daemon=True)
        thread.start()
        return contrast_id

    def get_contrast(self, contrast_id: str) -> dict | None:
        """Consulta estado de un contraste asíncrono."""
        with self._contrast_lock:
            return self._contrasts.get(contrast_id)

    def _detect_discrepancy(self, openai_answer: str, local_answer: str | None) -> dict:
        """Detecta si las respuestas difieren significativamente."""
        if not local_answer:
            return {"differs": False, "reason": "Local no disponible"}

        oa = openai_answer.lower().strip()
        la = local_answer.lower().strip()

        # 1. Detectar si el local fue evasivo (no tenía info)
        evasive_phrases = ["no contiene información", "no puedo asegurar",
                           "no se especifican detalles", "no se dispone de", "no se menciona"]
        local_evasive = any(p in la for p in evasive_phrases)
        if local_evasive:
            return {"differs": False, "reason": "Local fue evasivo (sin contexto suficiente)"}

        # 2. Comparar cifras numéricas mencionadas
        oa_numbers = set(re.findall(r'\d+[\.,]?\d*', oa))
        la_numbers = set(re.findall(r'\d+[\.,]?\d*', la))
        key_oa = {n for n in oa_numbers if len(n) >= 2}
        key_la = {n for n in la_numbers if len(n) >= 2}
        if key_oa and key_la:
            common = key_oa & key_la
            if len(common) < min(len(key_oa), len(key_la)) * 0.3:
                return {
                    "differs": True,
                    "reason": f"Discrepancia numérica: OpenAI cita {key_oa - common}, Local cita {key_la - common}"
                }

        # 3. Comparar letras de opción múltiple (A/B/C/D)
        oa_letter = re.search(r'\b([A-D])\)', oa)
        la_letter = re.search(r'\b([A-D])\)', la)
        if oa_letter and la_letter:
            if oa_letter.group(1) != la_letter.group(1):
                return {
                    "differs": True,
                    "reason": f"OpenAI seleccionó {oa_letter.group(1)}, Local seleccionó {la_letter.group(1)}"
                }

        # 4. Comparar longitud relativa (si una es muy corta vs la otra)
        len_ratio = len(la) / max(len(oa), 1)
        if len_ratio < 0.15:
            return {"differs": True, "reason": "Local dio respuesta muy breve comparada con OpenAI"}

        return {"differs": False, "reason": "Respuestas consistentes"}

    # ── Caché con retroalimentación de calidad ──────────────────────────────
    def _cache_key(self, question: str) -> str:
        try:
            mtime = os.path.getmtime(self.chunks_path)
            version_str = str(mtime)
        except OSError:
            version_str = "v1"
            
        normalized = f"{question.strip().lower()}||{version_str}"
        return hashlib.md5(normalized.encode()).hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        try:
            with shelve.open(self.cache_path, 'r') as db:
                entry = db.get(key)
                if entry and entry.get("quality") == "bad":
                    return None  # No servir respuestas flaggeadas como malas
                return entry
        except Exception:
            return None

    def _cache_set(self, key: str, value: dict):
        try:
            # Preservar quality si ya existía una evaluación
            existing_quality = None
            try:
                with shelve.open(self.cache_path, 'r') as db:
                    existing = db.get(key)
                    if existing:
                        existing_quality = existing.get("quality")
            except Exception:
                pass

            value["quality"] = existing_quality  # None = sin evaluar
            with shelve.open(self.cache_path) as db:
                db[key] = value
        except Exception as e:
            self._status(f"⚠️ Error guardando caché: {e}")

    def flag_cache(self, question: str, quality: str) -> bool:
        """
        Marca una respuesta en caché como 'good' o 'bad'.
        quality: 'good' | 'bad' | None (reset)
        """
        cache_key = self._cache_key(question)
        try:
            with shelve.open(self.cache_path) as db:
                entry = db.get(cache_key)
                if entry:
                    entry["quality"] = quality
                    db[cache_key] = entry
                    self._status(f"Cache entry flagged as '{quality}'")
                    return True
            return False
        except Exception as e:
            self._status(f"⚠️ Error flagging cache: {e}")
            return False

    def get_cache_stats(self) -> dict:
        """Estadísticas de la caché."""
        stats = {"total": 0, "good": 0, "bad": 0, "unrated": 0}
        try:
            with shelve.open(self.cache_path, 'r') as db:
                for key in db:
                    stats["total"] += 1
                    q = db[key].get("quality")
                    if q == "good":
                        stats["good"] += 1
                    elif q == "bad":
                        stats["bad"] += 1
                    else:
                        stats["unrated"] += 1
        except Exception:
            pass
        return stats

    def clear_bad_cache(self) -> int:
        """Elimina solo las entradas marcadas como malas. Retorna nro eliminadas."""
        removed = 0
        try:
            with shelve.open(self.cache_path) as db:
                bad_keys = [k for k in db if db[k].get("quality") == "bad"]
                for k in bad_keys:
                    del db[k]
                    removed += 1
            self._status(f"🗑️ {removed} respuestas malas eliminadas")
        except Exception:
            pass
        return removed

    def clear_cache(self):
        """Limpia toda la caché de respuestas."""
        try:
            with shelve.open(self.cache_path) as db:
                db.clear()
            self._status("🗑️ Caché limpiada")
        except Exception:
            pass

    def clear_history(self):
        """Limpia el historial de conversación."""
        self._history.clear()
