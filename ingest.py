"""
ingest.py — Dokumentu indeksēšanas skripts
==========================================
Izpilda VIENREIZ (vai katru reizi, kad dokumentācija mainās).

Nolasa visus failus no ./docs mapes, sadala gabalos,
izveido vektoru indeksu un saglabā ChromaDB datubāzē.

Atbalstītie formāti: PDF, Word (.docx), Excel (.xlsx/.xls),
                     Markdown (.md), HTML (.html)

Lietošana:
    python ingest.py
    python ingest.py --docs_dir ./mana_dokumentacija
"""

import os
import argparse
from pathlib import Path

# Dokumentu parsēšana
import pypdf
from docx import Document as DocxDocument
import pandas as pd
from bs4 import BeautifulSoup
import markdown

# Vektoru datubāze
import chromadb
from chromadb.utils import embedding_functions

# Utilītas
from dotenv import load_dotenv

load_dotenv()

# ── Konfigurācija ────────────────────────────────────────────────────────────

DOCS_DIR = "./docs"          # Mape ar dokumentiem
CHROMA_DIR = "./chroma_db"   # ChromaDB saglabāšanas vieta
COLLECTION_NAME = "dokumentacija"

CHUNK_SIZE = 800             # Gabala lielums (simboli)
CHUNK_OVERLAP = 150          # Pārlappošanās starp gabaliem


# ── Teksta iegūšana no dažādiem formātiem ────────────────────────────────────

def extract_pdf(filepath: Path) -> str:
    """Iegūst tekstu no PDF faila."""
    text = ""
    with open(filepath, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_docx(filepath: Path) -> str:
    """Iegūst tekstu no Word dokumenta."""
    doc = DocxDocument(str(filepath))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_excel(filepath: Path) -> str:
    """Iegūst tekstu no Excel faila (visas lapas)."""
    text_parts = []
    xl = pd.ExcelFile(str(filepath))
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        # Katru rindu pārvērš tekstā
        text_parts.append(f"[Lapa: {sheet_name}]")
        for _, row in df.iterrows():
            row_text = " | ".join(
                str(v) for v in row.values if pd.notna(v) and str(v).strip()
            )
            if row_text:
                text_parts.append(row_text)
    return "\n".join(text_parts)


def extract_markdown(filepath: Path) -> str:
    """Iegūst tekstu no Markdown faila."""
    with open(filepath, "r", encoding="utf-8") as f:
        md_text = f.read()
    # Pārvērš HTML, tad iegūst tīru tekstu
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def extract_html(filepath: Path) -> str:
    """Iegūst tekstu no HTML faila."""
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # Izņem skriptus un stilus
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def extract_text(filepath: Path) -> str | None:
    """Nosaka faila tipu un iegūst tekstu."""
    suffix = filepath.suffix.lower()
    extractors = {
        ".pdf":  extract_pdf,
        ".docx": extract_docx,
        ".doc":  extract_docx,
        ".xlsx": extract_excel,
        ".xls":  extract_excel,
        ".md":   extract_markdown,
        ".html": extract_html,
        ".htm":  extract_html,
    }
    if suffix not in extractors:
        print(f"  ⚠️  Nezināms formāts, izlaižam: {filepath.name}")
        return None
    try:
        return extractors[suffix](filepath)
    except Exception as e:
        print(f"  ❌ Kļūda apstrādājot {filepath.name}: {e}")
        return None


# ── Teksta sadalīšana gabalos ─────────────────────────────────────────────────

def split_into_chunks(text: str, source: str) -> list[dict]:
    """
    Sadala tekstu pārklājošos gabalos.
    Atgriež sarakstu ar {'text': ..., 'source': ..., 'chunk_index': ...}
    """
    # Notīra liekās tukšas rindiņas
    text = "\n".join(
        line for line in text.splitlines() if line.strip()
    )

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]

        # Mēģina nepārgriezt vārda vidū
        if end < len(text):
            last_space = chunk_text.rfind(" ")
            if last_space > CHUNK_SIZE * 0.7:
                chunk_text = chunk_text[:last_space]
                end = start + last_space

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text.strip(),
                "source": source,
                "chunk_index": index,
            })
            index += 1

        start = end - CHUNK_OVERLAP  # Pārlappošanās

    return chunks


# ── Galvenā indeksēšanas funkcija ─────────────────────────────────────────────

def ingest(docs_dir: str = DOCS_DIR):
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"❌ Mape '{docs_dir}' neeksistē!")
        return

    # Inicializē ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Embeddings — sentence-transformers (darbojas lokāli, bez API atslēgas)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Izdzēš veco kolekciju, ja eksistē (pilna pārindeksēšana)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️  Izdzēsta esošā kolekcija, veidojam no jauna...\n")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Apstrādā visus failus
    all_chunks = []
    files = sorted(docs_path.rglob("*"))
    doc_files = [f for f in files if f.is_file()]

    print(f"📂 Atrasti {len(doc_files)} faili mapē '{docs_dir}'\n")

    for filepath in doc_files:
        print(f"📄 Apstrādā: {filepath.name}")
        text = extract_text(filepath)
        if not text or len(text.strip()) < 50:
            print(f"  ⚠️  Teksts nav atrasts vai pārāk īss, izlaižam.\n")
            continue

        chunks = split_into_chunks(text, source=filepath.name)
        print(f"  ✅ {len(chunks)} gabali\n")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("❌ Nav ko indeksēt. Pārliecinies, ka ./docs mapē ir dokumenti.")
        return

    # Saglabā ChromaDB (partijās pa 500)
    print(f"💾 Saglabā {len(all_chunks)} gabalus ChromaDB...")
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[f"{c['source']}__chunk_{c['chunk_index']}" for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in batch],
        )

    print(f"\n✅ Indeksēšana pabeigta! {len(all_chunks)} gabali saglabāti ChromaDB.")
    print(f"   Datubāze: {CHROMA_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dokumentu indeksēšanas skripts")
    parser.add_argument(
        "--docs_dir",
        default=DOCS_DIR,
        help=f"Mape ar dokumentiem (noklusējums: {DOCS_DIR})",
    )
    args = parser.parse_args()
    ingest(docs_dir=args.docs_dir)
