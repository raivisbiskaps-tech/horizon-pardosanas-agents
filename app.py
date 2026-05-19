"""
app.py — Dokumentācijas Q&A aplikācija
=======================================
Streamlit web saskarne, kas ļauj uzdot jautājumus par dokumentāciju.

Priekšnosacījums: vispirms izpildi ingest.py!

Lietošana:
    streamlit run app.py
"""

import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from mistralai import Mistral
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource(show_spinner="Indeksē dokumentāciju, lūdzu uzgaidi...")
def auto_ingest():
    """Automātiski palaiž indeksēšanu, ja chroma_db nav atrasta."""
    if not os.path.exists(CHROMA_DIR):
        # Pievieno projekta mapi Python ceļam
        sys.path.insert(0, os.path.dirname(__file__))
        from ingest import ingest
        ingest()
    return True

# ── Konfigurācija ────────────────────────────────────────────────────────────

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "dokumentacija"

# Cik relevantu fragmentu iegūt no ChromaDB
TOP_K_RESULTS = 5

# Mistral modelis
MISTRAL_MODEL = "mistral-large-latest"

# Sistēmas uzvedne — nosaka Claude uzvedību
SYSTEM_PROMPT = """Tu esi lietotāju atbalsta asistents, kas atbild uz jautājumiem \
balstoties TIKAI uz sniegto dokumentāciju.

Noteikumi:
- Atbildi precīzi un kodolīgi
- Ja atbilde nav atrodama dokumentācijā, atbildi: "Šī informācija nav pieejama dokumentācijā."
- Nekad neizdomā informāciju, ko neatrodi dokumentos
- Ja jautājums ir neskaidrs, lūdz precizējumu
- Atbildi tajā pašā valodā, kurā tiek uzdots jautājums"""


# ── ChromaDB inicializācija ───────────────────────────────────────────────────

@st.cache_resource
def load_collection():
    """Ielādē ChromaDB kolekciju (kešots — ielādē tikai vienreiz)."""
    if not os.path.exists(CHROMA_DIR):
        st.error(
            "❌ ChromaDB datubāze nav atrasta. "
            "Lūdzu, vispirms izpildi: `python ingest.py`"
        )
        st.stop()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
        )
        return collection
    except Exception:
        st.error(
            "❌ Kolekcija nav atrasta. "
            "Lūdzu, izpildi: `python ingest.py`"
        )
        st.stop()


@st.cache_resource
def load_mistral_client():
    """Inicializē Mistral klientu."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("❌ MISTRAL_API_KEY nav iestatīts .env failā.")
        st.stop()
    return Mistral(api_key=api_key)


# ── RAG funkcija ──────────────────────────────────────────────────────────────

def retrieve_context(collection, question: str) -> tuple[str, list[str]]:
    """
    Meklē relevantos fragmentus ChromaDB.
    Atgriež (konteksta_teksts, avotu_saraksts).
    """
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K_RESULTS,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    sources = []

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Filtrē pārāk tālos rezultātus (cosine distance > 0.8 = maz relevants)
        if dist < 0.8:
            source = meta.get("source", "nezināms")
            context_parts.append(f"[Avots: {source}]\n{doc}")
            if source not in sources:
                sources.append(source)

    return "\n\n---\n\n".join(context_parts), sources


def ask_mistral(client, question: str, context: str) -> str:
    """Nosūta jautājumu un kontekstu uz Mistral API."""
    user_message = f"""Dokumentācija:
{context}

---

Jautājums: {question}"""

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="pārdošanas aģents",
        page_icon="📚",
        layout="wide",
    )

    # Virsraksts ar logo un personas attēlu
    col_logo, col_title, col_img = st.columns([1, 3, 1])
    with col_logo:
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
    with col_title:
        st.markdown("<h1 style='color: #003087;'>pārdošanas aģents</h1>", unsafe_allow_html=True)
    with col_img:
        img_path = os.path.join(os.path.dirname(__file__), "assets", "Horizon.jpg")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)

    # Automātiski indeksē, ja nepieciešams
    auto_ingest()

    # Ielādē resursus
    collection = load_collection()
    mistral_client = load_mistral_client()

    # Inicializē čata vēsturi
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Attēlo čata vēsturi
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Avoti"):
                    for src in msg["sources"]:
                        st.text(f"• {src}")

    # Ievades lauks
    if question := st.chat_input("Uzraksti jautājumu..."):

        # Parāda lietotāja jautājumu
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Iegūst atbildi
        with st.chat_message("assistant"):
            with st.spinner("Meklē dokumentācijā..."):
                context, sources = retrieve_context(collection, question)

                if not context:
                    answer = "Šī informācija nav pieejama dokumentācijā."
                    sources = []
                else:
                    answer = ask_mistral(mistral_client, question, context)

            st.markdown(answer)

            if sources:
                with st.expander("📎 Avoti"):
                    for src in sources:
                        st.text(f"• {src}")

        # Saglabā asistenta atbildi vēsturē
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    # Sānjosla ar info
    with st.sidebar:
        st.header("ℹ️ Informācija")
        try:
            count = collection.count()
            st.metric("Indeksēti fragmenti", count)
        except Exception:
            pass

        if st.button("🗑️ Notīrīt čatu"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Darbināts ar Mistral AI")


if __name__ == "__main__":
    main()
