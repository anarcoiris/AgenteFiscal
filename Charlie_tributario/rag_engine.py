"""
rag_engine.py — Motor RAG reutilizable para el proyecto tributario.

v2 — Mejoras:
  - Embedding multilingüe (BAAI/bge-m3) para español
  - Caché de respuestas (shelve) para evitar re-consultas
  - Indicador de confianza basado en distancia FAISS
  - Timeout Ollama extendido a 300s para DeepSeek R1
  - Fallback OpenAI actualizado a gpt-4.1-mini
  - OCR vía pytesseract (CPU, sin VRAM)
  - Metadata de páginas en las fuentes
"""
import os
import hashlib
import pickle
import shelve
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
    """Motor RAG local: FAISS + DeepSeek (Ollama) con fallback OpenAI."""

    def __init__(
        self,
        index_path="faiss_index.bin",
        chunks_path="chunks.pkl",
        cache_path="rag_cache",
        embedding_model="BAAI/bge-m3",
        ollama_url="http://localhost:11434",
        llm_model="qwen2.5:3b",
        top_k=7,
        on_status=None,
    ):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.cache_path = cache_path
        self.embedding_model_name = embedding_model
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.top_k = top_k
        self._on_status = on_status

        self._embedding_model = None
        self._index = None
        self._chunks = None
        self._loaded = False
        self._history = []  # Últimas N interacciones para contexto conversacional

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
    def query(self, question: str, debug: bool = False) -> dict:
        """
        Pipeline RAG completo con caché e indicador de confianza.

        Retorna: {
            "answer": str,
            "sources": list[dict],
            "confidence": "alta"|"media"|"baja",
            "cached": bool,
            "debug_chunks": list[str] (solo si debug=True),
            "error": str|None
        }
        """
        if not self._loaded:
            self.load()

        # 1. Comprobar caché
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

        # 5. Generar respuesta
        self._status(f"Generando respuesta (confianza: {confidence})...")
        answer = self._call_llm(prompt)

        if answer is None:
            return {
                "answer": "Error: no se pudo obtener respuesta del modelo.",
                "sources": sources,
                "confidence": confidence,
                "cached": False,
                "error": "LLM unavailable"
            }

        result = {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "cached": False,
            "error": None
        }

        # Debug: incluir chunks recuperados
        if debug:
            result["debug_chunks"] = context_parts

        # 6. Guardar en caché
        self._cache_set(cache_key, result)

        # 7. Actualizar historial
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
1. Usa la información del contexto para responder. Puedes sintetizar y explicar, pero NO inventes datos, artículos ni procedimientos que no se deriven del contexto.
2. Si el contexto no tiene información relevante, di: "El contexto no contiene información sobre este tema."
3. Cita fuentes y páginas cuando sea posible (ej: "Según el Manual, p.45...").
4. Responde en español usando el Euro (€).
5. Si el contexto trata sobre otro país (ej: República Dominicana), indícalo.
6. Sé preciso y directo.
{history_text}
CONTEXTO:
{context_text}

PREGUNTA:
{question}
"""

    # ── LLM calls ───────────────────────────────────────────────────────────
    def _call_llm(self, prompt: str) -> str | None:
        """Intenta modelo local primero. Ejecuta fallback inteligente si la respuesta es evasiva o nula."""
        answer = self._call_ollama(prompt)
        
        # Detectar si el modelo local dio una "no-respuesta" por falta de contexto (usando heurísticas)
        is_evasive = False
        if answer:
            evasive_phrases = ["no contiene información", "no puedo asegurar", "no se especifican detalles", "no se dispone de", "no se menciona"]
            if any(phrase in answer.lower() for phrase in evasive_phrases):
                is_evasive = True
                self._status("Local model lacked confidence, falling back to OpenAI...")
        
        if answer and not is_evasive:
            return answer

        # Fallback a OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key and OPENAI_AVAILABLE:
            self._status("Fallback inteligente: consultando a OpenAI (GPT-4o mini)...")
            openai_answer = self._call_openai(prompt, api_key)
            if openai_answer:
                return f"*(Respuesta ampliada mediante OpenAI GPT-4o-mini debido a falta de contexto local)*\n\n{openai_answer}"
            elif answer: # Si OpenAI falla, devuelve la respuesta evasiva local al menos
                return answer

        if not answer:
            self._status("⚠️ Ollama no disponible y OPENAI_API_KEY no configurada")
        
        return answer

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

    def _call_openai(self, prompt: str, api_key: str) -> str | None:
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
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

    # ── Caché ───────────────────────────────────────────────────────────────
    def _cache_key(self, question: str) -> str:
        normalized = question.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        try:
            with shelve.open(self.cache_path, 'r') as db:
                return db.get(key)
        except Exception:
            return None

    def _cache_set(self, key: str, value: dict):
        try:
            with shelve.open(self.cache_path) as db:
                db[key] = value
        except Exception as e:
            self._status(f"⚠️ Error guardando caché: {e}")

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
