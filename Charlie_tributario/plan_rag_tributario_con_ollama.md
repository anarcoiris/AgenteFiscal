# 🧾 Plan completo: RAG tributario con Ollama (Qwen / DeepSeek)

## 🎯 Objetivo
Construir un sistema local que:
- Procese un PDF tributario (sin OCR)
- Lo convierta en conocimiento estructurado
- Permita consultas tipo chatbot
- Sirva para aprender y responder cuestionarios fiscales

---

# 🧱 Fase 1: Ingesta del PDF

## 1.1 Extraer texto

Herramientas recomendadas:
- `pypdf` (rápido, básico)
- `pdfplumber` (mejor estructura)

```python
import pdfplumber

text = ""
with pdfplumber.open("manual.pdf") as pdf:
    for page in pdf.pages:
        text += page.extract_text() + "\n"
```

## 1.2 Limpieza básica

- Eliminar saltos de línea innecesarios
- Quitar encabezados/pies repetidos
- Normalizar espacios

---

# ✂️ Fase 2: Chunking inteligente

## 2.1 Objetivo
Dividir el texto en fragmentos útiles para recuperación.

## 2.2 Estrategia recomendada

- Tamaño: 300–800 tokens
- Solapamiento: 50–100 tokens
- Separar por:
  - Secciones
  - Artículos
  - Tipos de renta

## 2.3 Ejemplo

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_text(text)
```

---

# 🏷️ Fase 3: Enriquecimiento (MUY IMPORTANTE)

Añadir metadata a cada chunk:

```json
{
  "texto": "...",
  "tipo": "IRPF",
  "subtipo": "capital mobiliario",
  "año": 2025,
  "fuente": "manual X",
  "seccion": "rendimientos"
}
```

💡 Esto permite consultas mucho más precisas.

---

# 🧠 Fase 4: Embeddings

## 4.1 Modelo recomendado

Opciones ligeras:
- `nomic-embed-text`
- `bge-small`

## 4.2 Generación

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en")
embeddings = model.encode(chunks)
```

---

# 🗄️ Fase 5: Base vectorial

## Opción simple (recomendada)
- FAISS

```python
import faiss
import numpy as np

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings))
```

Guardar:
- índice
- chunks
- metadata

---

# 🔎 Fase 6: Recuperación

```python
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=5)

contextos = [chunks[i] for i in I[0]]
```

Filtrado opcional:
- por año
- por tipo de renta

---

# 🤖 Fase 7: Integración con Ollama

## 7.1 Modelos disponibles

- qwen2.5:3b → rápido, razonable
- deepseek-r1:8b → mejor razonamiento

## 7.2 Prompt base

```text
Eres un asesor fiscal experto en España (2025).

Responde solo usando el contexto.
Si no hay información suficiente, dilo.

Contexto:
{contextos}

Pregunta:
{query}
```

## 7.3 Llamada a Ollama

```python
import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "deepseek-r1:8b",
    "prompt": prompt,
    "stream": False
})

print(response.json()["response"])
```

---

# 🧪 Fase 8: Evaluación

Crear preguntas tipo:

- "¿Cómo tributan los dividendos?"
- "¿Qué gastos son deducibles en alquiler?"
- "Calcular rendimiento neto..."

Verificar:
- precisión
- coherencia
- citas implícitas

---

# 🚀 Fase 9: Mejoras

## 9.1 Ranking
- Reordenar chunks por relevancia

## 9.2 Multi-query
- Reformular preguntas automáticamente

## 9.3 Hybrid search
- vectorial + keyword

## 9.4 Caché
- guardar respuestas frecuentes

---

# 🧠 Fase 10: Nivel PRO

- Añadir BOE
- Consultas DGT
- Versionado por año
- Motor de cálculo determinista

---

# 🧰 Stack final sugerido

- Python
- pdfplumber
- sentence-transformers
- FAISS
- Ollama (Docker)

---

# ⚠️ Riesgos

- Texto mal extraído
- Chunking pobre
- Falta de contexto
- Normativa desactualizada

---

# ✅ Siguiente paso

1. Probar extracción del PDF
2. Generar primeros chunks
3. Testear embeddings
4. Hacer primera query simple

---

Si quieres, en el siguiente paso te construyo:

👉 Un repositorio completo listo para ejecutar
👉 O una interfaz tipo chatbot + simulador fiscal

