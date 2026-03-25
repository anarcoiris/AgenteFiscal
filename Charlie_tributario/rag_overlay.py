"""
rag_overlay.py — Ventana flotante RAG Tributario.

v2 — Mejoras:
  - Indicador visual de confianza (🟢🟡🔴)
  - Citación de páginas en fuentes
  - Botón "Limpiar" para resetear historial y vista
  - TclError safety en todos los callbacks de hilo
  - Crash-safe al cerrar ventana durante consulta
"""
import os
import sys
import threading
import re
import customtkinter as ctk

# Añadir el directorio del script al path para imports locales
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from rag_engine import RAGEngine
from screen_capture import ScreenCapture


# ── Configuración visual ────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

FONT_FAMILY = "Segoe UI"
COLOR_BG = "#1a1a2e"
COLOR_SURFACE = "#16213e"
COLOR_ACCENT = "#0f3460"
COLOR_HIGHLIGHT = "#00d2ff"
COLOR_TEXT = "#e0e0e0"
COLOR_MUTED = "#888888"

WINDOW_WIDTH = 520
WINDOW_HEIGHT_COMPACT = 160
WINDOW_HEIGHT_EXPANDED = 600
DEFAULT_ALPHA = 0.92

CONFIDENCE_ICONS = {
    "alta": "🟢",
    "media": "🟡",
    "baja": "🔴",
}


class RAGOverlay(ctk.CTk):
    """Ventana flotante RAG Tributario v2."""

    def __init__(self):
        super().__init__()

        # ── Ventana ──────────────────────────────────────────────────────────
        self.title("🧾 RAG Tributario")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT_COMPACT}")
        self.minsize(400, 140)
        self.attributes("-topmost", True)
        self.attributes("-alpha", DEFAULT_ALPHA)
        self.configure(fg_color=COLOR_BG)

        # Posicionar en esquina superior derecha
        screen_w = self.winfo_screenwidth()
        x_pos = screen_w - WINDOW_WIDTH - 20
        self.geometry(f"+{x_pos}+40")

        self._expanded = False
        self._settings_open = False
        self._debug_mode = False
        self._alive = True
        self._engine = RAGEngine(on_status=self._set_status_safe)
        self._capture = ScreenCapture()

        # Manejar cierre seguro
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._load_engine_async()

    def _on_close(self):
        """Cierre seguro: marca como no-vivo y destruye."""
        self._alive = False
        self.destroy()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header bar
        header = ctk.CTkFrame(self, fg_color=COLOR_SURFACE, corner_radius=0, height=36)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🧾 RAG Tributario",
            font=(FONT_FAMILY, 13, "bold"), text_color=COLOR_HIGHLIGHT
        ).pack(side="left", padx=12)

        # Botón configurar (Gear)
        self._settings_btn = ctk.CTkButton(
            header, text="⚙️", width=30, height=24,
            font=(FONT_FAMILY, 12), fg_color="transparent",
            hover_color=COLOR_ACCENT, command=self._toggle_settings
        )
        self._settings_btn.pack(side="right", padx=2)

        # Botón debug
        self._debug_btn = ctk.CTkButton(
            header, text="🔍", width=30, height=24,
            font=(FONT_FAMILY, 12), fg_color="transparent",
            hover_color=COLOR_ACCENT, command=self._toggle_debug
        )
        self._debug_btn.pack(side="right", padx=2)

        # Botón limpiar
        self._clear_btn = ctk.CTkButton(
            header, text="🗑️", width=30, height=24,
            font=(FONT_FAMILY, 12), fg_color="transparent",
            hover_color=COLOR_ACCENT, command=self._on_clear
        )
        self._clear_btn.pack(side="right", padx=4)

        # Opacity slider
        self._opacity_slider = ctk.CTkSlider(
            header, from_=0.4, to=1.0, width=70, height=14,
            button_color=COLOR_HIGHLIGHT, button_hover_color="#33ddff",
            progress_color=COLOR_ACCENT,
            command=self._on_opacity_change
        )
        self._opacity_slider.set(DEFAULT_ALPHA)
        self._opacity_slider.pack(side="right", padx=4)

        ctk.CTkLabel(
            header, text="α", font=(FONT_FAMILY, 11), text_color=COLOR_MUTED
        ).pack(side="right")

        # Input row
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=(10, 4))

        self._query_entry = ctk.CTkEntry(
            input_frame, placeholder_text="Escribe tu pregunta fiscal...",
            font=(FONT_FAMILY, 13), height=38,
            fg_color=COLOR_SURFACE, border_color=COLOR_ACCENT,
            text_color=COLOR_TEXT
        )
        self._query_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._query_entry.bind("<Return>", lambda e: self._on_send())

        self._send_btn = ctk.CTkButton(
            input_frame, text="➤", width=42, height=38,
            font=(FONT_FAMILY, 16), fg_color=COLOR_ACCENT,
            hover_color=COLOR_HIGHLIGHT, command=self._on_send
        )
        self._send_btn.pack(side="left", padx=(0, 4))

        self._capture_btn = ctk.CTkButton(
            input_frame, text="📸", width=42, height=38,
            font=(FONT_FAMILY, 16), fg_color=COLOR_ACCENT,
            hover_color=COLOR_HIGHLIGHT, command=self._on_capture
        )
        self._capture_btn.pack(side="left")

        # Status bar
        self._status_label = ctk.CTkLabel(
            self, text="Inicializando...",
            font=(FONT_FAMILY, 11), text_color=COLOR_MUTED, anchor="w"
        )
        self._status_label.pack(fill="x", padx=14, pady=(2, 6))

        # Settings panel (hidden)
        self._settings_frame = ctk.CTkFrame(self, fg_color=COLOR_SURFACE, height=0)
        self._build_settings_ui()

        # Response area (hidden initially)
        self._response_frame = ctk.CTkFrame(self, fg_color="transparent")

        # Confidence indicator + sources on same row
        info_frame = ctk.CTkFrame(self._response_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=10, pady=(4, 2))

        self._confidence_label = ctk.CTkLabel(
            info_frame, text="",
            font=(FONT_FAMILY, 11, "bold"), text_color=COLOR_TEXT, anchor="w"
        )
        self._confidence_label.pack(side="left")

        self._cached_label = ctk.CTkLabel(
            info_frame, text="",
            font=(FONT_FAMILY, 10), text_color=COLOR_MUTED, anchor="e"
        )
        self._cached_label.pack(side="right")

        self._response_text = ctk.CTkTextbox(
            self._response_frame, font=(FONT_FAMILY, 12),
            fg_color=COLOR_SURFACE, text_color=COLOR_TEXT,
            border_color=COLOR_ACCENT, border_width=1,
            corner_radius=8, wrap="word"
        )
        self._response_text.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        # Sources label
        self._sources_label = ctk.CTkLabel(
            self._response_frame, text="",
            font=(FONT_FAMILY, 10), text_color=COLOR_MUTED, anchor="w",
            wraplength=WINDOW_WIDTH - 40
        )
        self._sources_label.pack(fill="x", padx=14, pady=(0, 4))

        # Debug area (hidden by default)
        self._debug_frame = ctk.CTkFrame(self._response_frame, fg_color="transparent")

        ctk.CTkLabel(
            self._debug_frame, text="🔍 Chunks recuperados:",
            font=(FONT_FAMILY, 11, "bold"), text_color="#ff9900", anchor="w"
        ).pack(fill="x", padx=10)

        self._debug_text = ctk.CTkTextbox(
            self._debug_frame, font=(FONT_FAMILY, 10),
            fg_color="#0d1117", text_color="#8b949e",
            border_color="#ff9900", border_width=1,
            corner_radius=6, wrap="word", height=150
        )
        self._debug_text.pack(fill="both", expand=True, padx=10, pady=(2, 8))

    def _build_settings_ui(self):
        # Model Selection
        ctk.CTkLabel(self._settings_frame, text="Modelo LLM:", font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=14, pady=(8, 2))
        self._model_var = ctk.StringVar(value=self._engine.llm_model)
        self._model_menu = ctk.CTkOptionMenu(
            self._settings_frame, values=["qwen2.5:3b", "deepseek-r1:8b", "qwen2.5vl:3b"],
            variable=self._model_var, command=self._update_engine_params,
            fg_color=COLOR_ACCENT, button_color=COLOR_ACCENT,
            button_hover_color=COLOR_HIGHLIGHT
        )
        self._model_menu.pack(fill="x", padx=14, pady=(0, 8))

        # Top-K
        ctk.CTkLabel(self._settings_frame, text="Fragmentos (Top-K):", font=(FONT_FAMILY, 11, "bold")).pack(anchor="w", padx=14, pady=(4, 2))
        self._topk_slider = ctk.CTkSlider(
            self._settings_frame, from_=3, to=15, number_of_steps=12,
            fg_color=COLOR_ACCENT, progress_color=COLOR_HIGHLIGHT,
            command=self._update_engine_params
        )
        self._topk_slider.set(self._engine.top_k)
        self._topk_slider.pack(fill="x", padx=14, pady=(0, 8))

        # Cache control
        self._clear_cache_btn = ctk.CTkButton(
            self._settings_frame, text="Limpiar Caché 🗑️", font=(FONT_FAMILY, 11),
            fg_color="#444", hover_color="#661111", command=self._engine.clear_cache
        )
        self._clear_cache_btn.pack(fill="x", padx=14, pady=(4, 12))

    def _update_engine_params(self, *args):
        self._engine.llm_model = self._model_var.get()
        self._engine.top_k = int(self._topk_slider.get())
        self._set_status(f"Config: {self._engine.llm_model} | Top-K: {self._engine.top_k}")

    def _toggle_settings(self):
        self._settings_open = not self._settings_open
        if self._settings_open:
            self._settings_frame.pack(fill="x", after=self._status_label, pady=(0, 10))
            self._settings_btn.configure(fg_color=COLOR_ACCENT)
        else:
            self._settings_frame.pack_forget()
            self._settings_btn.configure(fg_color="transparent")

    # ── Expand / Collapse ────────────────────────────────────────────────────
    def _expand(self):
        if not self._expanded:
            self._response_frame.pack(fill="both", expand=True)
            self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT_EXPANDED}")
            self._expanded = True

    def _collapse(self):
        if self._expanded:
            self._response_frame.pack_forget()
            self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT_COMPACT}")
            self._expanded = False

    # ── Actions ──────────────────────────────────────────────────────────────
    def _on_opacity_change(self, value):
        self.attributes("-alpha", value)

    def _toggle_debug(self):
        """Activa/desactiva modo debug."""
        self._debug_mode = not self._debug_mode
        if self._debug_mode:
            self._debug_btn.configure(fg_color=COLOR_ACCENT)
            self._set_status("Modo debug activado")
        else:
            self._debug_btn.configure(fg_color="transparent")
            self._debug_frame.pack_forget()
            self._set_status("Modo debug desactivado")

    def _on_clear(self):
        """Limpia respuesta, historial y caché."""
        self._collapse()
        self._response_text.delete("1.0", "end")
        self._engine.clear_history()
        self._set_status("Historial limpiado")

    def _on_send(self):
        question = self._query_entry.get().strip()
        if not question:
            return
        self._query_entry.delete(0, "end")
        self._run_query(question)

    def _on_capture(self):
        """Activa el modo de selección asíncrono sin ocultar la ventana."""
        self._set_status("Dibuja un rectángulo para capturar...")
        self._set_buttons_enabled(False)
        self._capture.capture_async(parent=self, on_complete=self._on_capture_done)

    def _on_capture_done(self, img):
        if not self._alive:
            return
            
        if img is None:
            self._set_status("Captura cancelada.")
            self._set_buttons_enabled(True)
            return

        self._set_status("Extrayendo texto de la captura...")

        def ocr_then_query():
            text = self._engine.image_to_text(img)
            if text.startswith("[Error") or text.startswith("[No se"):
                self._safe_after(lambda: self._set_status(f"OCR: {text}"))
                self._safe_after(lambda: self._set_buttons_enabled(True))
                return
            self._safe_after(lambda: self._set_status(f"Texto extraído: {text[:80]}..."))
            self._run_query_in_thread(text)

        threading.Thread(target=ocr_then_query, daemon=True).start()

    def _run_query(self, question: str):
        """Ejecuta la consulta RAG en un hilo secundario."""
        self._set_buttons_enabled(False)
        self._set_status(f"Consultando: {question[:60]}...")
        self._expand()
        
        # Insertar pregunta del usuario
        self._response_text.insert("end", f"🧑‍💻 Tú:\n", "user")
        self._response_text.insert("end", f"{question}\n\n")
        
        # Marcador de procesamiento
        self._pending_marker = "⏳ Procesando respuesta...\n"
        self._response_text.insert("end", self._pending_marker, "think")
        self._response_text.see("end")
        
        self._sources_label.configure(text="")
        self._confidence_label.configure(text="")

        threading.Thread(target=self._run_query_in_thread, args=(question,), daemon=True).start()

    def _run_query_in_thread(self, question: str):
        try:
            result = self._engine.query(question, debug=self._debug_mode)
            answer = result["answer"]
            sources = result["sources"]
            confidence = result.get("confidence", "media")
            cached = result.get("cached", False)
            debug_chunks = result.get("debug_chunks", [])

            # Formatear fuentes con páginas
            unique_sources = []
            seen = set()
            for s in sources:
                ref = f"{s['fuente']}"
                if s.get("paginas"):
                    ref += f" ({s['paginas']})"
                if ref not in seen:
                    unique_sources.append(ref)
                    seen.add(ref)
            sources_text = "📚 " + " · ".join(unique_sources) if unique_sources else ""

            conf_icon = CONFIDENCE_ICONS.get(confidence, "⚪")
            conf_text = f"{conf_icon} Confianza: {confidence}"
            cached_text = "⚡ Caché" if cached else ""

            self._safe_after(lambda: self._show_response(answer, sources_text, conf_text, cached_text, debug_chunks))
        except Exception as e:
            self._safe_after(lambda: self._show_response(f"Error: {e}", "", "", "", []))
        finally:
            self._safe_after(lambda: self._set_buttons_enabled(True))

    def _show_response(self, answer: str, sources_text: str, conf_text: str, cached_text: str, debug_chunks: list = None):
        self._expand()
        
        # Eliminar marcador de procesamiento
        try:
            content = self._response_text.get("1.0", "end")
            if self._pending_marker in content:
                # Buscar posición del marcador y borrarlo
                idx = content.find(self._pending_marker)
                if idx != -1:
                    # Contar líneas y cols para el índice de Tkinter
                    # Pero es más fácil buscar desde el final
                    self._response_text.delete("end-2c linestart", "end")
        except: pass

        self._response_text.insert("end", f"🤖 Charlie:\n", "assistant")
        
        # Aplicar Renderizado Markdown
        self._render_markdown(answer)
        
        # Añadir fuentes al final de la respuesta en el texto
        if sources_text:
            self._response_text.insert("end", f"\n{sources_text}\n", "think")
        
        self._response_text.insert("end", f"\n{'─'*40}\n\n", "think")
        self._response_text.see("end")
        
        self._sources_label.configure(text=sources_text)
        # Debug chunks
        if self._debug_mode and debug_chunks:
            self._debug_frame.pack(fill="both", expand=True)
            self._debug_text.delete("1.0", "end")
            for i, chunk in enumerate(debug_chunks, 1):
                self._debug_text.insert("end", f"--- Chunk {i} ---\n{chunk}\n\n")
        else:
            self._debug_frame.pack_forget()
        self._set_status("Listo.")

    def _render_markdown(self, text: str):
        """Simula renderizado markdown usando tags de Tkinter."""
        txt = self._response_text
        
        # Definir tags (usar _textbox para permitir 'font' que ctk prohíbe)
        txt._textbox.tag_config("bold", font=(FONT_FAMILY, 12, "bold"))
        txt._textbox.tag_config("header", font=(FONT_FAMILY, 14, "bold"), spacing1=10, spacing3=5, foreground=COLOR_HIGHLIGHT)
        txt._textbox.tag_config("list", lmargin1=10, lmargin2=20)
        txt._textbox.tag_config("think", font=(FONT_FAMILY, 11, "italic"), foreground=COLOR_MUTED)
        txt._textbox.tag_config("user", font=(FONT_FAMILY, 11, "bold"), foreground=COLOR_HIGHLIGHT, spacing1=15)
        txt._textbox.tag_config("assistant", font=(FONT_FAMILY, 11, "bold"), foreground="#00ff88", spacing1=10)

        # Limpiar tags previos no es necesario porque borramos todo en _show_response
        
        lines = text.split("\n")
        in_think = False
        
        for line in lines:
            line_str = line.strip()
            
            # Bloque <think> (Oud de DeepSeek)
            if "<think>" in line: 
                in_think = True
                continue
            if "</think>" in line: 
                in_think = False
                continue
            
            if in_think:
                txt.insert("end", line + "\n", "think")
                continue

            # Headers (# Header)
            if line_str.startswith("#"):
                clean = line_str.lstrip("#").strip()
                txt.insert("end", clean + "\n", "header")
                continue
            
            # Listas (- o *)
            if line_str.startswith("- ") or line_str.startswith("* ") or (line_str[0:1].isdigit() and ". " in line_str[:4]):
                self._insert_with_bold(line + "\n", "list")
                continue
            
            # Línea normal
            self._insert_with_bold(line + "\n")

    def _insert_with_bold(self, text, base_tag=None):
        """Inserta texto detectando **bold**."""
        txt = self._response_text
        parts = re.split(r"(\*\*.*?\*\*)", text)
        
        for part in parts:
            tags = []
            if base_tag: tags.append(base_tag)
            
            if part.startswith("**") and part.endswith("**"):
                clean = part[2:-2]
                tags.append("bold")
                txt.insert("end", clean, tuple(tags))
            else:
                txt.insert("end", part, tuple(tags))

    # ── Engine loading ───────────────────────────────────────────────────────
    def _load_engine_async(self):
        def load():
            try:
                self._engine.load()
                self._safe_after(lambda: self._set_buttons_enabled(True))
            except Exception as e:
                self._safe_after(lambda: self._set_status(f"❌ Error: {e}"))

        self._set_buttons_enabled(False)
        threading.Thread(target=load, daemon=True).start()

    # ── Helpers (thread-safe UI updates) ─────────────────────────────────────
    def _safe_after(self, callback):
        """Thread-safe UI update que no crashea si la ventana fue cerrada."""
        if self._alive:
            try:
                self.after(0, callback)
            except Exception:
                pass

    def _set_status(self, msg: str):
        try:
            self._status_label.configure(text=msg)
        except Exception:
            pass

    def _set_status_safe(self, msg: str):
        self._safe_after(lambda: self._set_status(msg))

    def _set_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        try:
            self._send_btn.configure(state=state)
            self._capture_btn.configure(state=state)
            self._query_entry.configure(state=state)
        except Exception:
            pass


if __name__ == "__main__":
    app = RAGOverlay()
    app.mainloop()
