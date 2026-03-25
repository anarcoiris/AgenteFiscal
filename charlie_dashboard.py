"""
charlie_dashboard.py — Backend FastAPI para Charlie v4.

v4 — Cambios respecto a v3:
  - NUEVO: integración con CharlieMemory — aprendizajes persisten entre sesiones.
  - NUEVO: WebSocket bidireccional — el cliente puede enviar respuestas a human_ask
           mientras la sesión está en curso (no solo la config inicial).
  - NUEVO: endpoint GET /api/memory para ver qué ha aprendido Charlie.
  - NUEVO: endpoint DELETE /api/memory/{category}/{key} para borrar entradas.
  - MANTENIDO: Plan-and-Act toggle, ping periódico, CSV, /api/plan, /api/candidaturas.
"""
import asyncio
import csv
import json
import os
import urllib.request
from datetime import date
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from charlie_browser  import CharlieBrowser
from charlie_agent    import CharlieAgent
from charlie_planner  import CharliePlanner
from charlie_memory   import CharlieMemory
from charlie_orchestrator import CharlieOrchestrator

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

def _read_cv_pdf(pdf_paths=["cv.pdf", "cv_english.pdf"]) -> str:
    if not PyPDF2:
        return "PyPDF2 no está instalado."
    text = []
    for path in pdf_paths:
        p = Path(path)
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text.append(page.extract_text() or '')
            except Exception as e:
                text.append(f"Error leyendo {path}: {e}")
    if not text:
        return "No hay CV adjunto."
    return "\\n".join(text)[:4000]


# ─── Constantes ───────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
CHROME_PATH  = os.getenv("CHROME_PATH",  r"C:\Program Files\Google\Chrome\Application\chrome.exe")
USER_DATA    = os.getenv("USER_DATA",    r"C:\Users\soyko\AppData\Local\Google\Chrome\User Data\Charlie")
CSV_LOG      = os.getenv("CSV_LOG",      r"C:\Users\soyko\Documents\LaAgencia\mis_candidaturas_bot.csv")
HTML_FILE    = Path(__file__).parent / "charlie_dashboard.html"

SEARCH_URLS: dict[str, str] = {
    "linkedin":    "https://www.linkedin.com/jobs/search/?keywords={role}&location=Madrid&f_AL=true",
    "tecnoempleo": "https://www.tecnoempleo.com/ofertas-trabajo/?te={role}&pr=29",
    "infojobs":    "https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword={role}&province=MADRID",
}
CSV_HEADERS = ["Fecha", "Empresa", "Rol", "Portal", "URL Oferta", "Resumen"]

# Memoria global compartida entre sesiones
_memory = CharlieMemory()

# ─── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Charlie Dashboard v4")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if HTML_FILE.exists():
        return HTMLResponse(content=HTML_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>charlie_dashboard.html not found</h1>", status_code=404)


@app.get("/api/status")
async def ollama_status():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            tags = json.loads(r.read())
        models = [m["name"] for m in tags.get("models", [])]
        return {
            "ollama":        "online",
            "model":         OLLAMA_MODEL,
            "model_present": any(OLLAMA_MODEL in m for m in models),
            "available":     models,
            "memory":        _memory.summary(),
        }
    except Exception as e:
        return {"ollama": "offline", "error": str(e)}


@app.get("/api/candidaturas")
async def get_candidaturas():
    csv_path = Path(CSV_LOG)
    if not csv_path.exists():
        return JSONResponse({"candidaturas": []})
    try:
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return JSONResponse({"candidaturas": rows})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/memory")
async def get_memory():
    """Devuelve todos los aprendizajes guardados en CharlieMemory."""
    return JSONResponse({
        "summary": _memory.summary(),
        "entries": _memory.all_entries(),
    })


@app.delete("/api/memory/{category}/{key}")
async def delete_memory_entry(category: str, key: str):
    """Elimina una entrada concreta de la memoria."""
    if category in _memory._long_term and key in _memory._long_term[category]:
        del _memory._long_term[category][key]
        _memory.save()
        return JSONResponse({"deleted": True})
    return JSONResponse({"deleted": False, "reason": "No encontrado"}, status_code=404)


@app.post("/api/plan")
async def preview_plan(body: dict):
    task       = body.get("task", "")
    model_name = body.get("model", OLLAMA_MODEL)
    url        = body.get("url", "about:blank")
    login      = body.get("login", "DESCONOCIDO")
    if not task:
        return JSONResponse({"error": "Falta 'task'"}, status_code=400)
    planner = CharliePlanner(ollama_url=OLLAMA_URL, model=model_name)
    plan, start_url = await planner.generate_plan(task, url, login)
    return JSONResponse({"plan": plan, "start_url": start_url})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    browser: CharlieBrowser | None = None
    ping_task:   asyncio.Task | None = None
    listen_task: asyncio.Task | None = None

    # Cola para pasar respuestas del usuario al agente en ejecución
    answer_queue: asyncio.Queue = asyncio.Queue()

    async def send(msg_type: str, data):
        try:
            await ws.send_text(json.dumps({"type": msg_type, "data": data}))
        except Exception:
            pass

    async def ping_loop():
        while True:
            await asyncio.sleep(20)
            try:
                await ws.send_text(json.dumps({"type": "ping"}))
            except Exception:
                break

    async def listen_loop():
        """
        Escucha mensajes entrantes del cliente durante la sesión.
        Tipos esperados post-config:
          {"type": "answer", "data": "<texto>"}  → respuesta a human_ask
          {"type": "teach",  "category": "...", "key": "...", "value": "..."}  → enseñar directamente
        """
        while True:
            try:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                t   = msg.get("type", "")
                if t == "answer":
                    await answer_queue.put(msg.get("data", ""))
                elif t == "teach":
                    _memory.learn(
                        category = msg.get("category", "general"),
                        key      = msg.get("key", ""),
                        value    = msg.get("value", ""),
                        source   = "user_direct",
                    )
                    await send("log", f"💾 Aprendido: [{msg.get('category')}] {msg.get('key')} = {msg.get('value')}")
                elif t == "teach_click":
                    if browser is not None:
                        data = msg.get("data", {})
                        x, y = int(data.get("x", 0)), int(data.get("y", 0))
                        elem = await browser.get_element_at(x, y)
                        if elem:
                            await send("teach_suggestion", elem)
                        else:
                            await send("log", f"⚠️ No se detectó un selector útil en ({x}, {y})")
            except Exception:
                break

    try:
        raw = await ws.receive_text()
        cfg = json.loads(raw)

        platform      = cfg.get("platform", "linkedin").strip().lower()
        role          = cfg.get("role", "Administrador de Sistemas").strip()
        salary        = cfg.get("salary", "26000").strip()
        model_name    = cfg.get("model", OLLAMA_MODEL).strip()
        task_override = cfg.get("task_override", "").strip() or None
        use_planner   = cfg.get("use_planner", True)

        if platform not in SEARCH_URLS and not task_override:
            await send("log",    f"❌ Portal desconocido: '{platform}'")
            await send("status", "error")
            return
        if not role and not task_override:
            await send("log",    "❌ Falta el campo 'role'.")
            await send("status", "error")
            return

        mode_str = "Plan-and-Act" if use_planner else "Directo"
        await send("log",    f"🐾 Charlie v4 [{mode_str}] | {platform.capitalize()} | {role} | {salary}€")
        await send("log",    f"🤖 Modelo: {model_name} | Memoria: {_memory.summary()['total_entradas']} entradas")
        if task_override:
            await send("log", f"🧪 Override: {task_override}")
        await send("status", "running")

        ping_task   = asyncio.create_task(ping_loop())
        listen_task = asyncio.create_task(listen_loop())

        browser = CharlieBrowser(
            ws_send=send, chrome_path=CHROME_PATH, user_data_dir=USER_DATA,
            headless=False, fps=2.0, jpeg_quality=60, viewport={"width": 1280, "height": 720},
        )
        await browser.start()

        # ── Tarea ─────────────────────────────────────────────────────────────
        if task_override:
            if task_override == "google_test":
                task = (
                    "OBJETIVO: Busca 'Charlie agent test' en Google y reporta el título del primer resultado.\n"
                    "1. Navega a https://www.google.com\n"
                    "2. Usa 'fill' en el input 🌟BUSCAR_AQUÍ con el texto 'Charlie agent test'\n"
                    "3. Usa 'press' Enter\n"
                    "4. Usa 'extract' en el primer resultado y 'done'.\n"
                    "   job_url = URL del resultado | company = 'Google'"
                )
            else:
                task = task_override
        else:
            role_encoded = role.replace(" ", "+")
            start_url    = SEARCH_URLS[platform].format(role=role_encoded)
            task = f"""\
=== OBJETIVO ===
Aplicar a una oferta de '{role}' en {platform.capitalize()}.

=== MI CURRÍCULUM ===
{_read_cv_pdf()}

=== CONTEXTO ===
Portal: {platform.capitalize()}
URL de búsqueda: {start_url}
Comprueba primero si estás logueado. Si no lo estás, usa 'error'.

=== PASOS ===
1. Navega a la URL de búsqueda.
2. Selecciona la primera oferta relevante que NO hayas aplicado ya.
   IMPORTANTE: Verifica que los requisitos encajen mínimamente con mi CV y PRIORIZA las ofertas que tengan "Solicitud Sencilla" o "Easy Apply".
3. Abre el detalle haciendo clic en la oferta.
4. Pulsa el botón de solicitud (LinkedIn → 'Easy Apply'; Tecnoempleo/InfoJobs → 'Inscribirme').
5. Rellena el formulario con los datos de la sección FORMULARIO.
6. Confirma el envío.
7. Usa 'done' con la URL exacta y el nombre de la empresa.

=== FORMULARIO ===
- Salario deseado: {salary} (solo números)
- Nivel de inglés: B2 o C1
- Años de experiencia: 5
- Carta: "Hola, soy apasionado de la automatización con base en Física. Expectativa: {salary} € brutos/año."
- Teléfono: si ya está relleno, déjalo; si pide uno nuevo: 600000000.

=== REGLAS ===
- NUNCA inventes URLs.
- Si la oferta NO tiene Easy Apply o te saca de la plataforma permanentemente tras intentarlo, cancela la oferta y sigue con otra.
- NO uses 'done' hasta confirmación visible de envío.
- Usa 'human_ask' si algo es ambiguo o inesperado.
"""

        planner = CharliePlanner(ollama_url=OLLAMA_URL, model=model_name) if use_planner else None

        agent = CharlieOrchestrator(
            browser      = browser,
            ws_send      = send,
            ollama_url   = OLLAMA_URL,
            model        = model_name,
            task_str     = task,
            max_steps    = 35,
            planner      = planner,
            memory       = _memory,
            answer_queue = answer_queue,
        )
        result = await agent.run()
        await browser.stop()
        browser = None

        action  = result.get("action", "error")
        job_url = result.get("job_url", "N/A")
        company = result.get("company", "N/A")
        summary = result.get("result", result.get("reason", str(result)))

        if action == "done":
            _append_csv(date.today(), company, role, platform.capitalize(), job_url, summary)
            await send("log",    f"📝 Registrado → {company} | {job_url}")
            await send("result", {"job_url": job_url, "company": company, "summary": summary})
            await send("status", "done")
        else:
            await send("log",    f"🛑 {action} — {summary}")
            await send("status", "error")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await send("log",    f"❌ Error fatal: {e}")
            await send("status", "error")
        except Exception:
            pass
    finally:
        for t in [ping_task, listen_task]:
            if t:
                t.cancel()
        if browser is not None:
            try:
                await browser.stop()
            except Exception:
                pass



# ─── Helpers ──────────────────────────────────────────────────────────────────

def _append_csv(fecha, empresa, rol, portal, url_oferta, resumen):
    csv_path = Path(CSV_LOG)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(CSV_HEADERS)
            writer.writerow([fecha, empresa, rol, portal, url_oferta, resumen])
    except Exception as e:
        print(f"⚠️ CSV error: {e}")


if __name__ == "__main__":
    import webbrowser
    print("🐾 Charlie Dashboard v3 en http://localhost:8083")
    webbrowser.open("http://localhost:8083")
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="warning")
