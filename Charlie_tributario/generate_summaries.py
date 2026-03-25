"""
generate_summaries.py — Resúmenes automáticos y Cheat Sheets de los PDF tributarios.

Pipeline (Map-Reduce):
  STEP 1. Extrae texto de todos los PDFs
  STEP 2. Divide por temas (regex de cabeceras + chunking semántico)
  STEP 3. MAP: Resume cada chunk temático
  STEP 4. REDUCE: Combina en Cheat Sheet estructurada

Genera outputs SEPARADOS por modelo:
  - CheatSheet_LOCAL_[DocName].md
  - CheatSheet_OPENAI_[DocName].md
  - intermediate_LOCAL.json
  - intermediate_OPENAI.json
"""
import os
import re
import glob
import json
import time
import argparse
import pdfplumber
import requests
from datetime import datetime
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Configuración ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "summaries")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LOCAL_MODEL = "qwen2.5:3b"          # 68x más rápido que DeepSeek R1
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

# Para el paso MAP, cada chunk tendrá ~5000 caracteres (~1250 tokens)
CHUNK_SIZE_MAP = 5000
CHUNK_OVERLAP  = 200

# Umbral para sub-dividir una sección temática si es muy grande
MAX_SECTION_SIZE = 12000

# Regex para detectar títulos de sección en manuales tributarios
SECTION_HEADER_RE = re.compile(
    r'(?:^|\n)(?P<header>'
    r'(?:TEMA|CAPÍTULO|SECCIÓN|TÍTULO|TITULO|ARTÍCULO|BLOQUE|PARTE|MÓDULO)\s+\d+[^\n]*'
    r'|(?:IRPF|IVA|IS|IAE|RENTA)\b[^\n]*'
    r')',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 & 2: Extracción de texto y chunking temático
# ═══════════════════════════════════════════════════════════════════════════════

def extract_full_text(pdf_path: str) -> tuple[str, list[int]]:
    """Extrae texto completo con marcadores de página."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            if text.strip():
                pages.append(f"<<<PAGE_{i}>>>\n{text.strip()}\n")
    return "\n".join(pages)


def thematic_split(full_text: str, source_name: str) -> list[dict]:
    """
    Divide el texto en secciones temáticas:
    1. Intenta detectar cabeceras como 'TEMA X', 'CAPÍTULO X', etc.
    2. Si no hay, usa RecursiveCharacterTextSplitter.
    3. Sub-divide secciones muy grandes.
    """
    # Encontrar posiciones de cabeceras
    headers = [(m.start(), m.group('header').strip()) for m in SECTION_HEADER_RE.finditer(full_text)]
    
    chunks = []
    
    if len(headers) < 3:
        # Sin cabeceras detectables → chunking semántico estándar
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_MAP,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n\n", "\n\n", ".\n", ". ", "\n"]
        )
        for chunk in splitter.split_text(full_text):
            pages = re.findall(r'<<<PAGE_(\d+)>>>', chunk)
            clean = re.sub(r'<<<PAGE_\d+>>>\n?', '', chunk).strip()
            if len(clean) > 50:
                chunks.append({
                    "source": source_name,
                    "section": f"Fragmento {len(chunks)+1}",
                    "pages": _page_range(pages),
                    "text": clean
                })
    else:
        # Divide por cabeceras detectadas
        boundaries = [h[0] for h in headers] + [len(full_text)]
        for i, (start, header) in enumerate(headers):
            end = boundaries[i + 1]
            section_text = full_text[start:end]
            
            if len(section_text) > MAX_SECTION_SIZE:
                # Sub-dividir sección grande
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE_MAP,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                sub_chunks = splitter.split_text(section_text)
                for j, sub in enumerate(sub_chunks):
                    pages = re.findall(r'<<<PAGE_(\d+)>>>', sub)
                    clean = re.sub(r'<<<PAGE_\d+>>>\n?', '', sub).strip()
                    if len(clean) > 50:
                        chunks.append({
                            "source": source_name,
                            "section": f"{header} (parte {j+1})",
                            "pages": _page_range(pages),
                            "text": clean
                        })
            else:
                pages = re.findall(r'<<<PAGE_(\d+)>>>', section_text)
                clean = re.sub(r'<<<PAGE_\d+>>>\n?', '', section_text).strip()
                if len(clean) > 50:
                    chunks.append({
                        "source": source_name,
                        "section": header,
                        "pages": _page_range(pages),
                        "text": clean
                    })
    
    return chunks


def _page_range(pages: list[str]) -> str:
    if not pages:
        return ""
    nums = sorted(set(int(p) for p in pages))
    return f"p.{nums[0]}" if len(nums) == 1 else f"pp.{nums[0]}-{nums[-1]}"


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Calls
# ═══════════════════════════════════════════════════════════════════════════════

CHUNK_SUMMARY_PROMPT = """Eres un experto en fiscalidad española. Resume este fragmento de manual tributario de forma concisa y estructurada.

EXTRAE SOLO LO QUE APAREZCA EN EL TEXTO:
- **Tema/Sección**: ¿De qué trata?
- **Conceptos clave**: 2-4 definiciones o reglas importantes.
- **Plazos/Fechas**: Si se mencionan.
- **Límites/Cuantías**: Importes clave (en € o % si aplica).
- **Modelos tributarios**: (ej: Modelo 100, 303...) si se citan.

NOTA: Si el texto es de República Dominicana (ITBIS, RD$), indícalo con ⚠️ JURISDICCIÓN RD.

FRAGMENTO ({source}, {pages}):
{text}
"""

CHEATSHEET_PROMPT = """Eres un experto fiscal. Basado en los siguientes resúmenes, genera una CHEAT SHEET (Chuleta) ultra-detallada y estructurada en Markdown para el documento aportado.

ESTRUCTURA obligatoria:
# 🧾 Chuleta Fiscal — {source} ({model})
*Generada el {date}*

---
## 📅 Calendario / Plazos Clave
[Debe incluir TODOS los plazos, fechas límite y periodos impositivos mencionados en los resúmenes. Usa formato de tabla]

---
## 💰 Límites, Cuantías y Porcentajes
[Debe incluir TODOS los umbrales de obligación de declarar, importes exentos, bases máximas/mínimas y tipos impositivos mencionados en los resúmenes. Usa formato de tabla]

---
## 📋 Modelos Tributarios
[Lista exhaustiva de TODOS los modelos (ej: 100, 303, 111, etc.) y su finalidad, tal como aparecen en los resúmenes]

---
## 🧮 Conceptos y Reglas Clave
[Agrupa por temas principales (ej: IRPF, Deducciones, Sanciones) y describe las reglas fundamentales que un contribuyente o asesor debe conocer]

---
## ⚠️ Excepciones y Notas Especiales
[Alertas, excepciones a la regla general, casos raros o información específica de jurisdicciones ajenas como República Dominicana si aplica]

IMPORTANTE: 
1. NO INVENTES NADA que no esté en los resúmenes.
2. SÉ EXHAUSTIVO. No te dejes reglas ni importes por fuera si están en el origen.
3. Usa el formato de tablas Markdown de forma intensiva para facilitar la lectura visual.

---
RESÚMENES (fuente de información):
{summaries}
"""


def call_local(prompt: str, model: str = LOCAL_MODEL, timeout: int = 120) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[ERROR LOCAL: {e}]"


def call_openai(prompt: str, model: str = OPENAI_MODEL) -> str:
    if not OPENAI_AVAILABLE or not OPENAI_KEY:
        return "[ERROR: No hay OPENAI_API_KEY configurada]"
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR OPENAI: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
# MAP: Resume cada chunk
# ═══════════════════════════════════════════════════════════════════════════════

def map_summaries(chunks: list[dict], llm_fn, label: str) -> list[dict]:
    results = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        progress = f"[{i}/{total}]"
        print(f"  {progress} MAP ({label}): {chunk['source']} | {chunk['section'][:50]} ({chunk['pages']})")
        
        prompt = CHUNK_SUMMARY_PROMPT.format(
            source=chunk["source"],
            pages=chunk["pages"],
            text=chunk["text"][:4500]  # Seguridad de tokens
        )
        
        t0 = time.time()
        summary = llm_fn(prompt)
        elapsed = time.time() - t0
        
        print(f"         → {len(summary)} chars en {elapsed:.1f}s")
        
        results.append({
            "source": chunk["source"],
            "section": chunk["section"],
            "pages": chunk["pages"],
            "summary": summary
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# REDUCE: Combina en Cheat Sheet (con reducción jerárquica si es necesario)
# ═══════════════════════════════════════════════════════════════════════════════

def reduce_cheatsheet(summaries: list[dict], llm_fn, model_label: str, source_name: str) -> str:
    """
    Genera la Cheat Sheet final para un documento específico. 
    Si hay demasiados resúmenes, hace una reducción intermedia.
    """
    if len(summaries) > 40:
        print(f"  ⚠️  {len(summaries)} resúmenes para '{source_name}'. Iniciando reducción jerárquica intermedia...")
        batch_size = 25
        intermediate_summaries = []
        
        num_batches = math.ceil(len(summaries) / batch_size)
        for i in range(num_batches):
            batch = summaries[i*batch_size : (i+1)*batch_size]
            print(f"    🔄 Reduciendo bloque {i+1}/{num_batches} ({len(batch)} resúmenes)...")
            
            combined_batch = "\n".join([f"PAGE {s['pages']} ({s['section']}):\n{s['summary']}" for s in batch])
            prompt = f"Eres un experto fiscal. Resume DE FORMA MUY EXHAUSTIVA Y DETALLADA estos puntos clave para preparar una chuleta final. Conserva TODOS los Modelos tributarios, Fechas, Plazos, Porcentajes y Cuantías económicas exactas. No pierdas el nivel de detalle:\n\n{combined_batch}"
            
            summary_of_batch = llm_fn(prompt)
            intermediate_summaries.append({
                "source": source_name, 
                "section": f"Bloque {i+1}", 
                "pages": "-", 
                "summary": summary_of_batch
            })
        
        summaries_to_reduce = intermediate_summaries
    else:
        summaries_to_reduce = summaries

    combined = ""
    for s in summaries_to_reduce:
        combined += f"\n### {s['section']} ({s['pages']})\n{s['summary']}\n"
    
    prompt = CHEATSHEET_PROMPT.format(
        model=model_label,
        date=datetime.now().strftime("%d/%m/%Y %H:%M"),
        source=source_name,
        summaries=combined[:35000]  # Ampliado para Qwen 32k / GPT-4o-mini 128k
    )
    
    print(f"\n  REDUCE FINAL ({model_label}): Generando Cheat Sheet para '{source_name}'...")
    return llm_fn(prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generador de Cheat Sheets Tributarias")
    parser.add_argument("--fast", action="store_true", help="Usa los resúmenes .json oxistentes para regenerar la Chuleta sin re-evaluar todo.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ── Detectar PDFs ────────────────────────────────────────────────────────
    pdfs = sorted(glob.glob(os.path.join(SCRIPT_DIR, "*.pdf")))
    if not pdfs:
        print("❌ No se encontraron PDFs en el directorio.")
        return
    
    print("=" * 70)
    print("📚 GENERADOR DE RESÚMENES Y CHEAT SHEETS")
    print("=" * 70)
    print(f"  PDFs encontrados: {len(pdfs)}")
    for p in pdfs:
        print(f"    • {os.path.basename(p)}")
    print()
    
    all_chunks = []
    source_names = [os.path.splitext(os.path.basename(p))[0][:40] for p in pdfs]
    
    if not args.fast:
        # ── STEP 1-2: Extracción y chunking ──────────────────────────────────────
        for pdf_path in pdfs:
            source_name = os.path.splitext(os.path.basename(pdf_path))[0][:40]
            print(f"📄 Extrayendo: {source_name}")
            
            full_text = extract_full_text(pdf_path)
            chunks = thematic_split(full_text, source_name)
            
            print(f"   → {len(chunks)} chunks temáticos detectados")
            all_chunks.extend(chunks)
        
        print(f"\n📊 Total chunks para procesar: {len(all_chunks)}\n")
    
    # ── STEP 3-4: Pipeline LOCAL ──────────────────────────────────────────────
    print("━" * 70)
    print(f"🖥️  PIPELINE LOCAL — {LOCAL_MODEL}")
    print("━" * 70)
    
    local_json = os.path.join(OUTPUT_DIR, "intermediate_LOCAL.json")
    if args.fast and os.path.exists(local_json):
        print(f"⚡ [FAST MODE] Cargando resúmenes desde: {local_json}")
        with open(local_json, "r", encoding="utf-8") as f:
            local_summaries = json.load(f)
    else:
        local_summaries = map_summaries(all_chunks, call_local, LOCAL_MODEL)
        # Guardar intermedios
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(local_summaries, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 Intermedios guardados: intermediate_LOCAL.json")
    
    # Reducimos POR DOCUMENTO en lugar de mezclarlo todo
    for doc in source_names:
        doc_summaries = [s for s in local_summaries if s["source"] == doc]
        if not doc_summaries: continue
        
        cheatsheet = reduce_cheatsheet(doc_summaries, call_local, f"Local ({LOCAL_MODEL})", doc)
        safe_name = re.sub(r'[^A-Za-z0-9_-]', '', doc.replace(' ', '_'))
        out_path = os.path.join(OUTPUT_DIR, f"CheatSheet_LOCAL_{safe_name}.md")
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cheatsheet)
        print(f"  ✅ CheatSheet LOCAL para '{doc}' → {out_path}")
    
    # ── STEP 3-4: Pipeline OPENAI ─────────────────────────────────────────────
    if OPENAI_KEY:
        print()
        print("━" * 70)
        print(f"☁️  PIPELINE OPENAI — {OPENAI_MODEL}")
        print("━" * 70)
        
        openai_json = os.path.join(OUTPUT_DIR, "intermediate_OPENAI.json")
        if args.fast and os.path.exists(openai_json):
            print(f"⚡ [FAST MODE] Cargando resúmenes desde: {openai_json}")
            with open(openai_json, "r", encoding="utf-8") as f:
                openai_summaries = json.load(f)
        else:
            if not all_chunks:
                print("⚠️ [FAST MODE] No hay chunks extraídos y no existe cache para OpenAI. Saltando.")
                openai_summaries = []
            else:
                openai_summaries = map_summaries(all_chunks, call_openai, OPENAI_MODEL)
                with open(openai_json, "w", encoding="utf-8") as f:
                    json.dump(openai_summaries, f, ensure_ascii=False, indent=2)
                print(f"\n  💾 Intermedios guardados: intermediate_OPENAI.json")
        
        for doc in source_names:
            doc_summaries = [s for s in openai_summaries if s["source"] == doc]
            if not doc_summaries: continue
            
            cheatsheet = reduce_cheatsheet(doc_summaries, call_openai, f"OpenAI ({OPENAI_MODEL})", doc)
            safe_name = re.sub(r'[^A-Za-z0-9_-]', '', doc.replace(' ', '_'))
            out_path = os.path.join(OUTPUT_DIR, f"CheatSheet_OPENAI_{safe_name}.md")
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cheatsheet)
            print(f"  ✅ CheatSheet OPENAI para '{doc}' → {out_path}")
    else:
        print("\n⚠️  Sin OPENAI_API_KEY — saltando pipeline OpenAI")
    
    # ── Resumen final ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("🏁 COMPLETADO")
    print(f"   Outputs en: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
