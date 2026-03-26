import os
import io
import json
import time
import uvicorn
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Importar el motor RAG
from rag_engine import RAGEngine

app = FastAPI(title="Charlie Tributario Web API")

# Permitir CORS por si acaso
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar motor RAG — GPT-4.1 como primario por defecto
engine = RAGEngine(
    llm_model="qwen2.5:3b",
    openai_model="gpt-4.1",
    top_k=7,
)
print("Cargando RAGEngine...")
engine.load()
print("RAGEngine cargado.")


class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4.1"  # Default: GPT-4.1 (OpenAI)


class FeedbackRequest(BaseModel):
    question: str
    quality: str  # "good" | "bad"


class ClearRequest(BaseModel):
    clear_cache: bool = False  # Solo limpia historial por defecto


@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Determinar qué modelo/motor usar
    use_model = None
    model_val = request.model

    if model_val.startswith("local:"):
        # Modelos locales: "local:qwen2.5:3b" etc.
        use_model = model_val
    elif model_val in ("gpt-4.1", "gpt-4.1-mini"):
        use_model = model_val
    else:
        # Legacy: modelos locales sin prefijo → convertir
        use_model = f"local:{model_val}"

    try:
        t0 = time.time()
        response = engine.query(request.message, debug=True, use_model=use_model)
        elapsed = round(time.time() - t0, 2)

        answer_text = response["answer"] or ""
        model_used = response.get("model_used", "N/A")
        contrast_id = response.get("contrast_id")

        # Serializar sources (limpiar distancias a 4 decimales)
        clean_sources = []
        for s in response.get("sources", []):
            clean_sources.append({
                "fuente": s.get("fuente", "Desconocida"),
                "paginas": s.get("paginas", ""),
                "distancia": round(s.get("distancia", 0), 4)
            })

        return JSONResponse(content={
            "success": True,
            "answer": answer_text,
            "sources": clean_sources,
            "confidence": response.get("confidence", "baja"),
            "model_used": model_used,
            "contrast_id": contrast_id,
            "elapsed_seconds": elapsed,
            "cached": response.get("cached", False),
        })
    except Exception as e:
        print(f"Error en chat: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/contrast/{contrast_id}")
async def get_contrast(contrast_id: str):
    """Consulta el estado de un contraste asíncrono local."""
    result = engine.get_contrast(contrast_id)
    if result is None:
        return JSONResponse(status_code=404, content={"success": False, "error": "Contraste no encontrado"})

    return JSONResponse(content={
        "success": True,
        "status": result["status"],
        "discrepancy": result.get("discrepancy"),
        "local_answer_preview": (result.get("local_answer") or "")[:300] if result["status"] == "done" else None,
    })


@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    """Marca una respuesta como buena o mala en la caché."""
    if request.quality not in ("good", "bad"):
        return JSONResponse(status_code=400, content={"success": False, "error": "quality must be 'good' or 'bad'"})

    ok = engine.flag_cache(request.question, request.quality)
    return {"success": ok}


@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        text = engine.image_to_text(image)
        return JSONResponse(content={"success": True, "text": text})
    except Exception as e:
        print(f"Error en OCR: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/clear")
async def clear_history(request: ClearRequest = ClearRequest()):
    """Limpia historial de conversación. Opcionalmente limpia caché completa."""
    engine.clear_history()
    if request.clear_cache:
        engine.clear_cache()
    return {"success": True, "cache_cleared": request.clear_cache}


@app.post("/api/cache/clear-bad")
async def clear_bad_cache():
    """Elimina solo las entradas de caché marcadas como malas."""
    removed = engine.clear_bad_cache()
    return {"success": True, "removed": removed}


@app.get("/api/cache/stats")
async def cache_stats():
    """Estadísticas de la caché."""
    stats = engine.get_cache_stats()
    return {"success": True, **stats}


@app.get("/api/config")
async def get_config():
    return {
        "success": True,
        "current_model": engine.openai_model,
        "local_model": engine.llm_model,
        "top_k": engine.top_k,
        "openai_available": bool(os.environ.get("OPENAI_API_KEY")),
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": engine.openai_model, "local": engine.llm_model}

# Servir archivos estáticos indexados
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    # Escucha en todas las interfaces para permitir acceso por LAN
    print("\nIniciando servidor en http://0.0.0.0:8084")
    uvicorn.run("web_server:app", host="0.0.0.0", port=8084, reload=True)
