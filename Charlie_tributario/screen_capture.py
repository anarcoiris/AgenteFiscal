"""
screen_capture.py — Módulo de captura de región de pantalla.

v2 — Mejoras:
  - Compensación de DPI scaling para pantallas HiDPI
  - Delay antes de captura para asegurar que el overlay desaparece
  - Mejor feedback visual durante la selección
"""
import ctypes
import time
import tkinter as tk
from PIL import Image
import mss
import mss.tools


def _get_dpi_scale():
    """Obtiene el factor de escala DPI en Windows."""
    try:
        # Activar DPI awareness para obtener coordenadas reales
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    try:
        # Obtener DPI del monitor principal
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0  # 96 DPI = 100% scaling
    except Exception:
        return 1.0


class ScreenCapture:
    """Overlay fullscreen para seleccionar una región de pantalla."""

    def __init__(self):
        self._result_image = None
        self._dpi_scale = _get_dpi_scale()

    def capture_async(self, parent, on_complete):
        """
        No bloquea. Llama a on_complete(Image) cuando termina, o on_complete(None) si cancela.
        """
        self._result_image = None
        self._start_x = 0
        self._start_y = 0
        self._on_complete = on_complete

        root = tk.Toplevel(parent)
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.3)
        root.configure(bg="black")
        root.config(cursor="crosshair")
        root.overrideredirect(True) 
        root.focus_force() 

        canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        self._rect_id = None
        self._size_label = None
        self._root = root
        self._canvas = canvas

        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release_async)
        root.bind("<Escape>", self._on_cancel_async)

        # Instrucciones
        canvas.create_text(
            root.winfo_screenwidth() // 2,
            30,
            text="Dibuja un rectángulo sobre el área a capturar  ·  ESC para cancelar",
            fill="white",
            font=("Segoe UI", 14),
        )

    def _on_cancel_async(self, event):
        self._root.destroy()
        if self._on_complete:
            self._on_complete(None)

    def _on_release_async(self, event):
        x1 = min(self._start_x, event.x)
        y1 = min(self._start_y, event.y)
        x2 = max(self._start_x, event.x)
        y2 = max(self._start_y, event.y)

        # Mínimo 20x20 px para evitar clics accidentales
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return

        self._root.withdraw()
        self._root.update()
        time.sleep(0.15)

        scale = self._dpi_scale
        region = {
            "left": int(x1 * scale),
            "top": int(y1 * scale),
            "width": int((x2 - x1) * scale),
            "height": int((y2 - y1) * scale),
        }

        try:
            with mss.mss() as sct:
                sct_img = sct.grab(region)
                self._result_image = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except Exception as e:
            print(f"Error en captura de pantalla: {e}")
            self._result_image = None
        finally:
            self._root.destroy()
            if self._on_complete:
                self._on_complete(self._result_image)


    # --- Métodos síncronos antiguos ---
    def capture(self, parent=None) -> Image.Image | None:
        """
        Bloquea hasta que el usuario seleccione un área o pulse Escape.
        Retorna PIL.Image del recorte, o None si se cancela.
        """
        self._result_image = None
        self._start_x = 0
        self._start_y = 0

        if parent:
            root = tk.Toplevel(parent)
        else:
            root = tk.Tk()
        
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.3)
        root.configure(bg="black")
        root.config(cursor="crosshair")
        root.overrideredirect(True) 

        canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        self._rect_id = None
        self._size_label = None
        self._root = root
        self._canvas = canvas

        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        root.bind("<Escape>", lambda e: root.destroy())

        # Instrucciones
        canvas.create_text(
            root.winfo_screenwidth() // 2,
            30,
            text="Dibuja un rectángulo sobre el área a capturar  ·  ESC para cancelar",
            fill="white",
            font=("Segoe UI", 14),
        )

        if parent:
            parent.wait_window(root)
        else:
            root.mainloop()
        
        return self._result_image

    def _on_press(self, event):
        self._start_x = event.x
        self._start_y = event.y
        if self._rect_id:
            self._canvas.delete(self._rect_id)
        if self._size_label:
            self._canvas.delete(self._size_label)
        self._rect_id = self._canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#00ff88", width=2, dash=(4, 4)
        )

    def _on_drag(self, event):
        if self._rect_id:
            self._canvas.coords(self._rect_id, self._start_x, self._start_y, event.x, event.y)
            # Mostrar dimensiones en píxeles
            w = abs(event.x - self._start_x)
            h = abs(event.y - self._start_y)
            if self._size_label:
                self._canvas.delete(self._size_label)
            self._size_label = self._canvas.create_text(
                event.x + 10, event.y + 15,
                text=f"{w}×{h}", fill="#00ff88", font=("Segoe UI", 10), anchor="nw"
            )

    def _on_release(self, event):
        x1 = min(self._start_x, event.x)
        y1 = min(self._start_y, event.y)
        x2 = max(self._start_x, event.x)
        y2 = max(self._start_y, event.y)

        # Mínimo 20x20 px para evitar clics accidentales
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return

        # Cerrar overlay ANTES de capturar
        self._root.withdraw()
        self._root.update()
        
        # Pequeño delay para que el overlay desaparezca completamente del framebuffer
        time.sleep(0.15)

        # Aplicar compensación DPI para captura real
        scale = self._dpi_scale
        region = {
            "left": int(x1 * scale),
            "top": int(y1 * scale),
            "width": int((x2 - x1) * scale),
            "height": int((y2 - y1) * scale),
        }

        try:
            with mss.mss() as sct:
                sct_img = sct.grab(region)
                self._result_image = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except Exception as e:
            print(f"Error en captura de pantalla: {e}")
            self._result_image = None
        finally:
            self._root.destroy()


if __name__ == "__main__":
    sc = ScreenCapture()
    img = sc.capture()
    if img:
        img.save("_test_capture.png")
        print(f"Captura guardada: {img.size}")
    else:
        print("Captura cancelada.")

