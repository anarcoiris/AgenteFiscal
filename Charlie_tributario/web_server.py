import os
import io
import json
import uvicorn
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

# Inicializar motor RAG
# Puedes ajustar el modelo inicial aquí
engine = RAGEngine(llm_model="qwen2.5:3b", top_k=6)
print("Cargando RAGEngine...")
engine.load()
print("RAGEngine cargado.")

class ChatRequest(BaseModel):
    message: str
    model: str = "qwen2.5:3b"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Si el modelo ha cambiado en el request, actualizamos el modelo del RAGEngine
    if request.model and request.model != engine.llm_model:
        print(f"Cambio de modelo a: {request.model}")
        engine.llm_model = request.model
    
    try:
        response = engine.query(request.message, debug=True)
        return JSONResponse(content={
            "success": True,
            "answer": response["answer"],
            "sources": response.get("sources", []),
            "confidence": response.get("confidence", "baja"),
            "debug_chunks": response.get("debug_chunks", [])
        })
    except Exception as e:
        print(f"Error en chat: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

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
async def clear_history():
    engine.clear_cache()
    engine._history = []
    # Usando clear_history en RAGEngine si existe
    if hasattr(engine, 'clear_history'):
        engine.clear_history()
    return {"success": True}

@app.get("/api/config")
async def get_config():
    return {
        "success": True,
        "current_model": engine.llm_model,
        "top_k": engine.top_k,
    }

# Servir archivos estáticos indexados
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    # Escucha en todas las interfaces para permitir acceso por LAN
    print("\nIniciando servidor en http://0.0.0.0:8084")
    uvicorn.run("web_server:app", host="0.0.0.0", port=8084, reload=True)
