"""
app.py — Dokumentācijas Q&A aplikācija
=======================================
Streamlit web saskarne, kas ļauj uzdot jautājumus par dokumentāciju.

Lietošana:
    streamlit run app.py
"""

import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Konfigurācija (definē PIRMS funkcijām) ───────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR   = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR     = os.path.join(BASE_DIR, "docs")
COLLECTION_NAME = "dokumentacija"
TOP_K_RESULTS   = 30
MISTRAL_MODEL   = "mistral-small-latest"

SYSTEM_PROMPT = """Tu esi pieredzējis ERP Horizon pārdošanas atbalsta speciālists, kas sniedz detalizētas un noderīgas atbildes, balstoties uz sniegto informāciju. Izplatām, izstrādājam un ieviešam ERP Horizon. Tava uzdevums ir sniegt atbildes uz potenciālo klientu jautājumiem.

Noteikumi:
- Sniedz pilnīgas, detalizētas atbildes — neaprobežojies ar vienu teikumu, bet arī neizplūsti garos apcerējumos
- Izmanto konkrētus piemērus un soļus no dokumentācijas, ja tie ir pieejami
- Ja atbilde nav atrodama dokumentācijā, atbildi: "Uz šo jautājumu nevarēšu sniegt precīzu atbildi, zvaniet Santai :)."
- Nekad neizdomā informāciju, ko neatrodi dokumentos
- Ja jautājums ir neskaidrs, lūdz precizējumu
- Atbildi tajā pašā valodā, kurā tiek uzdots jautājums

Konteksts par klientu:
- Jautājumus uzdod potenciāls klients bez Horizon zināšanām
- Klients neorientējas Horizon struktūrā un terminoloģijā
- Atbildes sniedz saprotamā, klientam draudzīgā valodā

Stils:
- Atbildi vieglā, sarunbiedra valodā
- Var izmantot humoru un iepīt atbildēs pa kādam jokam"""


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
    """Nosūta jautājumu un kontekstu uz Mistral API ar atkārtošanas loģiku."""
    user_message = f"""Zemāk ir VIENĪGIE dokumentu fragmenti, ko drīksti izmantot atbildē.
Ja atbilde nav šajos fragmentos — nekādā gadījumā to neizdomā.

=== DOKUMENTU FRAGMENTI (VIENĪGAIS AVOTS) ===
{context}
=== FRAGMENTU BEIGAS ===

Jautājums: {question}

Atceries: atbildi TIKAI no iepriekš sniegtajiem fragmentiem. Ja informācija tur nav — saki to tieši."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                    time.sleep(wait)
                    continue
                return "⚠️ Sistēma šobrīd ir noslogota. Lūdzu, mēģini vēlreiz pēc dažām sekundēm."
            raise


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="pārdošanas aģents",
        page_icon="📚",
        layout="wide",
    )

    # ── Header ────────────────────────────────────────────────────────────────
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

    # ── Čats ──────────────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Avoti"):
                    for src in msg["sources"]:
                        st.text(f"• {src}")

    # ── Jautājuma ievade ──────────────────────────────────────────────────────
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
