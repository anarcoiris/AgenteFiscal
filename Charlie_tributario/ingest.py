"""
ingest.py — Ingesta de documentos PDF tributarios.

Mejoras v3:
  - Modelo de embedding multilingüe (BAAI/bge-m3)
  - Preserva estructura de párrafos en limpieza
  - Metadata con número de página
  - Auto-descubrimiento de PDFs en el directorio
  - Chunks más grandes (1000) con mayor overlap (150)
  - Ingesta de CheatSheets LOCAL + OPENAI
  - Soporte para archivos .txt y .md adicionales
  - Separadores semánticos ampliados
"""
import os
import re
import glob
import pickle
import pdfplumber
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def extract_text_with_pages(pdf_path):
    """Extrae texto de un PDF preservando información de página."""
    print(f"  📄 Extrayendo texto de: {os.path.basename(pdf_path)}")
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pages_data.append({"page": i, "text": page_text})
    print(f"     → {len(pages_data)} páginas con texto")
    return pages_data


def clean_text(text):
    """
    Limpieza conservadora que preserva la estructura del documento.
    - Normaliza espacios múltiples en uno solo
    - Elimina líneas que solo contienen números (paginación)
    - Preserva saltos de párrafo (doble salto de línea)
    """
    # Eliminar líneas que son solo un número (paginación)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Normalizar múltiples espacios en línea a uno solo
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalizar 3+ saltos de línea consecutivos a 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Eliminar espacios al inicio/final de cada línea
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def process_document(pdf_path, source_name):
    """Procesa un PDF en chunks con metadata rica."""
    pages_data = extract_text_with_pages(pdf_path)

    if not pages_data:
        print(f"     ⚠️ Sin texto extraíble en {source_name}")
        return []

    # Construir texto completo con marcadores de página
    # Usamos marcadores para poder trackear la página de cada chunk
    PAGE_MARKER = "<<<PAGE_{}>>"
    full_text = ""
    for pd_item in pages_data:
        cleaned = clean_text(pd_item["text"])
        full_text += f"{PAGE_MARKER.format(pd_item['page'])}\n{cleaned}\n\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n---\n", "\n# ", "\n\n", "\n", ". ", " "]
    )
    raw_chunks = splitter.split_text(full_text)

    chunks = []
    for chunk_text in raw_chunks:
        # Extraer números de página presentes en este chunk
        page_numbers = re.findall(r'<<<PAGE_(\d+)>>', chunk_text)
        # Limpiar los marcadores del texto final
        clean_chunk = re.sub(r'<<<PAGE_\d+>>\n?', '', chunk_text).strip()

        if not clean_chunk or len(clean_chunk) < 20:
            continue

        page_range = ""
        if page_numbers:
            pages = sorted(set(int(p) for p in page_numbers))
            if len(pages) == 1:
                page_range = f"p.{pages[0]}"
            else:
                page_range = f"pp.{pages[0]}-{pages[-1]}"

        chunks.append({
            "texto": clean_chunk,
            "fuente": source_name,
            "paginas": page_range,
        })

    return chunks


def discover_pdfs(directory="."):
    """Auto-descubre todos los PDFs en el directorio."""
    pdfs = glob.glob(os.path.join(directory, "*.pdf"))
    print(f"📂 PDFs encontrados: {len(pdfs)}")
    for p in pdfs:
        print(f"   • {os.path.basename(p)} ({os.path.getsize(p) / 1024:.0f} KB)")
    return pdfs


def process_markdown_cheatsheets():
    """Procesa los MD de Cheat Sheets (OPENAI + LOCAL) para inyectarlos al RAG."""
    chunks = []
    summaries_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summaries")
    # Incluir tanto OPENAI como LOCAL
    md_files = glob.glob(os.path.join(summaries_dir, "CheatSheet_OPENAI*.md"))
    md_files += glob.glob(os.path.join(summaries_dir, "CheatSheet_LOCAL*.md"))
    
    if not md_files:
        return []
        
    print(f"\n📄 Extrayendo contenido de {len(md_files)} Cheat Sheets...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n---\n", "\n# ", "\n\n", "\n"]
    )
    
    for md_path in md_files:
        source_name = os.path.basename(md_path).replace(".md", "")
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        raw_chunks = splitter.split_text(text)
        for chunk in raw_chunks:
            if len(chunk.strip()) > 30:
                chunks.append({
                    "texto": f"[[RESUMEN ESTRUCTURADO / CHEAT SHEET]]\n{chunk.strip()}",
                    "fuente": source_name,
                    "paginas": "Resumen Global"
                })
                
    return chunks


def process_extra_documents():
    """Procesa archivos .txt y .md sueltos en el directorio raíz."""
    chunks = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    extra_files = glob.glob(os.path.join(base_dir, "*.txt"))
    extra_files += glob.glob(os.path.join(base_dir, "*.md"))
    
    # Excluir archivos del propio sistema
    exclude = {"requirements_rag.txt", "plan_rag_tributario_con_ollama.md", "plan_asistente_fiscal_openai.md"}
    extra_files = [f for f in extra_files if os.path.basename(f) not in exclude]
    
    if not extra_files:
        return []
    
    print(f"\n📄 Procesando {len(extra_files)} documentos extra (.txt/.md)...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n---\n", "\n# ", "\n\n", "\n", ". ", " "]
    )
    
    for filepath in extra_files:
        source_name = os.path.splitext(os.path.basename(filepath))[0]
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                text = f.read()
        
        raw_chunks = splitter.split_text(text)
        for chunk in raw_chunks:
            if len(chunk.strip()) > 30:
                chunks.append({
                    "texto": chunk.strip(),
                    "fuente": source_name,
                    "paginas": "Documento extra"
                })
        print(f"   • {source_name}: {len(raw_chunks)} chunks")
    
    return chunks


def main():
    print("=" * 60)
    print("🧾 Ingesta RAG Tributario v2")
    print("=" * 60)

    # Auto-descubrir PDFs
    pdf_files = discover_pdfs()

    if not pdf_files:
        print("❌ No se encontraron PDFs en el directorio.")
        return

    all_chunks = []
    for pdf_path in pdf_files:
        source_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n📂 Procesando: {source_name}")
        doc_chunks = process_document(pdf_path, source_name)
        all_chunks.extend(doc_chunks)
        print(f"     ✅ {len(doc_chunks)} chunks generados.")

    # 2. Procesar Cheat Sheets (Alta densidad de conocimiento)
    md_chunks = process_markdown_cheatsheets()
    if md_chunks:
        all_chunks.extend(md_chunks)
        print(f"     ✅ {len(md_chunks)} chunks generados desde Cheat Sheets.")

    # 3. Procesar documentos extra (.txt, .md)
    extra_chunks = process_extra_documents()
    if extra_chunks:
        all_chunks.extend(extra_chunks)
        print(f"     ✅ {len(extra_chunks)} chunks generados desde docs extra.")

    if not all_chunks:
        print("❌ No se generaron chunks. Verifica los archivos.")
        return

    print(f"\n📊 Total chunks generados: {len(all_chunks)}")

    # Mostrar distribución por fuente
    from collections import Counter
    fuentes = Counter(c["fuente"] for c in all_chunks)
    for fuente, count in fuentes.items():
        print(f"   • {fuente}: {count} chunks")

    print(f"\n🧠 Cargando modelo de embeddings ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("🔢 Generando embeddings...")
    texts = [c["texto"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    print("📦 Construyendo índice FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    print("💾 Guardando índice y chunks...")
    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n✅ Ingesta completada: {index.ntotal} vectores de dimensión {dimension}")


if __name__ == "__main__":
    main()
