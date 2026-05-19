"""
app.py — Dokumentācijas Q&A aplikācija
=======================================
Streamlit web saskarne, kas ļauj uzdot jautājumus par dokumentāciju.

Lietošana:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Konfigurācija (definē PIRMS funkcijām) ───────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR   = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR     = os.path.join(BASE_DIR, "docs")
COLLECTION_NAME = "dokumentacija"
TOP_K_RESULTS   = 8
MISTRAL_MODEL   = "mistral-large-latest"

SYSTEM_PROMPT = """Tu esi Horizon pārdošanas aģents. Tava vienīgā zināšanu bāze ir \
sniegtā dokumentācija. Atbildi TIKAI uz to, kas ir tieši un skaidri minēts dokumentos.

Stingri noteikumi:
- Atbildi TIKAI ar informāciju, kas ir tieši atrodama sniegtajos dokumentu fragmentos
- NEKAD nepaplašini, neinterpretē vai nemin informāciju, kas nav dokumentos
- Ja atbilde ir dokumentos — citē vai precīzi pārstāsti to, kas tur rakstīts
- Ja atbilde NAV dokumentos — atbildi tikai: "Šī informācija nav pieejama dokumentācijā."
- Neveic pieņēmumus, nesalīdzini ar citiem produktiem, nepievieno savu viedokli
- Atbildi tajā pašā valodā, kurā tiek uzdots jautājums
- Ja dokumentos ir konkrēti soļi vai skaitļi — norādi tos precīzi"""


# ── Indeksēšana ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Indeksē dokumentāciju, lūdzu uzgaidi...")
def auto_ingest():
    """Indeksē dokumentus pie katras jaunas sesijas."""
    sys.path.insert(0, BASE_DIR)
    from ingest import ingest
    ingest(docs_dir=DOCS_DIR)
    return True


# ── ChromaDB ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Ielādē dokumentu datubāzi...")
def load_collection():
    """Ielādē ChromaDB kolekciju."""
    import chromadb
    from chromadb.utils import embedding_functions

    if not os.path.exists(CHROMA_DIR):
        st.error("❌ Dokumentu datubāze nav atrasta. Pārliecinies, ka docs/ mapē ir dokumenti.")
        st.stop()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    except Exception as e:
        st.error(f"❌ Kolekcija nav atrasta: {e}")
        st.stop()


# ── Mistral klients ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Savieno ar AI...")
def load_mistral_client():
    """Inicializē Mistral klientu caur OpenAI-saderīgo API."""
    from openai import OpenAI
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("❌ MISTRAL_API_KEY nav iestatīts. Pievieno to Streamlit Secrets.")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")


# ── RAG ───────────────────────────────────────────────────────────────────────

def retrieve_context(collection, question: str) -> tuple[str, list[str]]:
    """Meklē relevantos fragmentus ChromaDB."""
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K_RESULTS,
        include=["documents", "metadatas", "distances"],
    )
    context_parts, sources = [], []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist < 0.95:
            src = meta.get("source", "nezināms")
            context_parts.append(f"[Avots: {src}]\n{doc}")
            if src not in sources:
                sources.append(src)
    return "\n\n---\n\n".join(context_parts), sources


def ask_mistral(client, question: str, context: str) -> str:
    """Nosūta jautājumu un kontekstu uz Mistral API."""
    response = client.chat.completions.create(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Dokumentācija:\n{context}\n\n---\n\nJautājums: {question}"},
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
        logo_path = os.path.join(BASE_DIR, "assets", "logo.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
    with col_title:
        st.markdown("<h1 style='color: #003087;'>pārdošanas aģents</h1>", unsafe_allow_html=True)
    with col_img:
        img_path = os.path.join(BASE_DIR, "assets", "Horizon.jpg")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)

    # Indeksē un ielādē resursus
    auto_ingest()
    collection     = load_collection()
    mistral_client = load_mistral_client()

    # Čata vēsture
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Avoti"):
                    for src in msg["sources"]:
                        st.text(f"• {src}")

    # Jautājuma ievade
    if question := st.chat_input("Uzraksti jautājumu..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Meklē dokumentācijā..."):
                context, sources = retrieve_context(collection, question)
                if not context:
                    answer  = "Šī informācija nav pieejama dokumentācijā."
                    sources = []
                else:
                    answer = ask_mistral(mistral_client, question, context)

            st.markdown(answer)
            if sources:
                with st.expander("📎 Avoti"):
                    for src in sources:
                        st.text(f"• {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    # Sānjosla
    with st.sidebar:
        st.header("ℹ️ Informācija")
        try:
            st.metric("Indeksēti fragmenti", collection.count())
        except Exception:
            pass
        if st.button("🗑️ Notīrīt čatu"):
            st.session_state.messages = []
            st.rerun()
        if st.button("🔄 Pārindeksēt dokumentus"):
            st.cache_resource.clear()
            st.rerun()
        st.divider()
        st.caption("Darbināts ar Mistral AI")


if __name__ == "__main__":
    main()
