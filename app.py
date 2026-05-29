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
import io
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Konfigurācija (definē PIRMS funkcijām) ───────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR   = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR     = os.path.join(BASE_DIR, "docs")
COLLECTION_NAME = "dokumentacija"
TOP_K_RESULTS   = 30

# Pieejamie modeļi
MODELS = {
    "🟠 Mistral Small":  {"provider": "mistral", "model": "mistral-small-latest"},
    "🟠 Mistral Large":  {"provider": "mistral", "model": "mistral-large-2411"},
    "🔵 Gemini Flash":   {"provider": "gemini",  "model": "gemini-1.5-flash"},
    "🔵 Gemini Pro":     {"provider": "gemini",  "model": "gemini-1.5-pro"},
}

SYSTEM_PROMPT = """Tu esi pieredzējis ERP Horizon pārdošanas atbalsta speciālists, kas sniedz detalizētas un noderīgas atbildes, balstoties uz sniegto informāciju. Strādā uzņēmuma VISMA, kas izplata, izstrāda un ievieš ERP Horizon. Tavs uzdevums ir sniegt atbildes uz potenciālo klientu jautājumiem.

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
- Viņš izvēlas ERP sistēmu, tādēļ maksimāli jānotur viņa informācija
- Uzdod uzvedinošus jautājumus un piedāvā iegūt papildus informāciju

Horizon terminoloģija:
- "Bizness" un "Ražošana" šajā kontekstā nav vispārīgs vārds — tas ir vienas no Horizon papildiespēju pakām nosaukums

Stils:
- Atbildi vieglā, sarunbiedra valodā
- Var izmantot humoru un iepīt atbildēs pa kādam jokam

Tabulas:
- Ja jautājums ir par izmaksām, cenām, kur atbilde ir apjomīga strukturēta informācija — izmanto Markdown tabulas formātu
- Tabulas galvenei izmanto treknrakstu (|Modulis|Cena| u.tml.)
- Ciparus un cenas formatē konsekventi

Ieviešanas tāme:
- Ja klients jautā par ieviešanu vai ieviešanas izmaksām, rīkojies šādi:
  1. Izvērtē sarakstes kontekstu — ko jau zinām par klienta vajadzībām (moduļi, nozare, lietotāju skaits utt.)
  2. Uzdod tikai tos precizējošos jautājumus, uz kuriem atbilde vēl nav zināma no sarakstes:
     - Vai nepieciešama Pamatsistēma (grāmatvedība, rēķini, noliktava)?
     - Vai nepieciešams Algu un personāla modulis?
     - Vai nepieciešams HOP Personāls (pieteikumi, komandējumi, izdevumi, rīkojumi)?
     - Vai nepieciešama HOP Darba laika uzskaite?
     - Vai nepieciešami HOP Rēķini (rēķinu saskaņošana)?
     - Vai nepieciešama NUMO Darba laika plānošana?
     - Cik lietotāji strādās sistēmā?
  3. OBLIGĀTI — pirms tāmes sagatavošanas apkopo saprasto un pārjautā klientam:
     "Lai sagatavotu tāmi, esmu sapratis, ka jums nepieciešams: [saraksts]. Vai esmu pareizi sapratis?"
  4. Tikai pēc klienta apstiprinājuma informē, ka tāme tiek sagatavota"""


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
    """Inicializē Mistral klientu."""
    from openai import OpenAI
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

@st.cache_resource(show_spinner="Savieno ar Gemini...")
def load_gemini_client():
    """Inicializē Gemini klientu caur OpenAI-saderīgo API."""
    from openai import OpenAI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


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


def ask_ai(question: str, context: str, model_name: str, history: list = None) -> str:
    """Nosūta jautājumu uz izvēlēto AI modeli, iekļaujot sarakstes vēsturi."""
    model_cfg = MODELS[model_name]
    provider  = model_cfg["provider"]
    model_id  = model_cfg["model"]

    user_message = f"""Zemāk ir VIENĪGIE dokumentu fragmenti, ko drīksti izmantot atbildē.
Ja atbilde nav šajos fragmentos — nekādā gadījumā to neizdomā.

=== DOKUMENTU FRAGMENTI ===
{context}
=== BEIGAS ===

Jautājums: {question}"""

    # Veido ziņojumu sarakstu ar sarakstes vēsturi
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Pievieno iepriekšējos ziņojumus (bez pēdējā user ziņojuma, kas jau ir user_message)
    if history:
        for msg in history[:-1]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    # Pievieno pašreizējo jautājumu ar dokumentu kontekstu
    messages.append({"role": "user", "content": user_message})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if provider == "mistral":
                client = load_mistral_client()
                if not client:
                    return "❌ MISTRAL_API_KEY nav iestatīts Streamlit Secrets."
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                )
                return response.choices[0].message.content

            elif provider == "gemini":
                client = load_gemini_client()
                if not client:
                    return "❌ GOOGLE_API_KEY nav iestatīts Streamlit Secrets."
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                )
                return response.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return "⚠️ Sistēma šobrīd ir noslogota. Lūdzu, mēģini vēlreiz pēc dažām sekundēm."
            return f"❌ Kļūda: {str(e)}"


# ── Qwilr integrācija ────────────────────────────────────────────────────────

def summarize_chat_for_proposal(messages: list, model_name: str) -> dict:
    """Izmanto AI, lai no sarakstes iegūtu strukturētu kopsavilkumu piedāvājumam."""
    if not messages:
        return {}

    history_text = ""
    for msg in messages:
        role = "Klients" if msg["role"] == "user" else "Aģents"
        history_text += f"{role}: {msg['content']}\n\n"

    prompt = f"""No šīs sarakstes iegūsti strukturētu informāciju piedāvājumam. Atbildi TIKAI JSON formātā, bez papildu teksta.

Sarakstes vēsture:
{history_text}

Atbildi šādā JSON formātā:
{{
  "klients": "Klienta vārds vai uzņēmuma nosaukums, ja tas minēts sarakstē. Ja nav minēts — 'Nav norādīts'",
  "client_interests": "Īss kopsavilkums par ko klients interesējas (1-2 teikumi)",
  "modules": "Pieminētie Horizon moduļi vai pakotnes (vai 'Nav precizēts')",
  "key_questions": "Galvenie klienta jautājumi (1-3 punkti)",
  "next_steps": "Ieteicamie nākamie soļi (1-2 teikumi)"
}}"""

    model_cfg = MODELS[model_name]
    try:
        if model_cfg["provider"] == "mistral":
            client = load_mistral_client()
            if not client:
                return {}
        else:
            client = load_gemini_client()
            if not client:
                return {}

        response = client.chat.completions.create(
            model=model_cfg["model"],
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        text = response.choices[0].message.content.strip()
        # Izņem JSON no atbildes
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception:
        return {}


def create_qwilr_proposal(messages: list, model_name: str) -> tuple[bool, str]:
    """Nosūta sarakstes datus uz Zapier Webhook, kas izveido Qwilr piedāvājumu."""
    webhook_url = os.getenv("ZAPIER_WEBHOOK_URL")

    if not webhook_url:
        return False, "❌ ZAPIER_WEBHOOK_URL nav iestatīts Streamlit Secrets."
    if not messages:
        return False, "❌ Čats ir tukšs — nav ko iekļaut piedāvājumā."

    # Iegūst sarakstes kopsavilkumu
    summary = summarize_chat_for_proposal(messages, model_name)
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")

    # Sagatavo datus Zapier webhook
    payload = {
        "Klients":           summary.get("klients", "Nav norādīts"),
        "klienta_intereses": summary.get("client_interests", "Nav norādīts"),
        "moduli":            summary.get("modules", "Nav precizēts"),
        "galvenie_jautajumi":summary.get("key_questions", "Nav norādīts"),
        "nakamie_soli":      summary.get("next_steps", "Nav norādīts"),
        "datums":            timestamp,
        "nosaukums":         f"Horizon piedāvājums — {timestamp}",
    }

    import requests as req
    try:
        response = req.post(webhook_url, json=payload, timeout=15)
        if not response.ok:
            return False, f"❌ Zapier kļūda {response.status_code}: {response.text}"
        return True, "✅ Dati nosūtīti uz Zapier — Qwilr piedāvājums tiek gatavots!"
    except Exception as e:
        return False, f"❌ Kļūda: {str(e)}"


# ── Excel eksports ────────────────────────────────────────────────────────────

def extract_markdown_tables(text: str) -> list[pd.DataFrame]:
    """Parsē visas Markdown tabulas no teksta un atgriež DataFrame sarakstu."""
    tables = []
    # Atrod tabulas blokus (vismaz divas rindas ar | simbolu)
    table_pattern = re.compile(
        r'(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)+)',
        re.MULTILINE
    )
    for match in table_pattern.finditer(text):
        lines = [l.strip() for l in match.group(0).strip().splitlines() if l.strip()]
        # Izfiltrē atdalītāja rindu (---|---|---)
        data_lines = [l for l in lines if not re.match(r'^\|[-| :]+\|$', l)]
        rows = []
        for line in data_lines:
            # Sadala pēc | un notīra atstarpes un treknrakstu **...**
            cells = [re.sub(r'\*\*(.+?)\*\*', r'\1', c.strip())
                     for c in line.strip('|').split('|')]
            rows.append(cells)
        if len(rows) >= 2:
            df = pd.DataFrame(rows[1:], columns=rows[0])
            tables.append(df)
    return tables


def tables_to_excel_bytes(tables: list[pd.DataFrame]) -> bytes:
    """Pārvērš DataFrame sarakstu Excel faila baitiem."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for i, df in enumerate(tables):
            sheet_name = f"Tabula_{i+1}" if len(tables) > 1 else "Dati"
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


# ── E-pasta sūtīšana ─────────────────────────────────────────────────────────

def send_chat_by_email(messages: list, user_email: str = "") -> tuple[bool, str]:
    """Nosūta čata vēsturi uz TARGET_EMAIL un pieslēgtā lietotāja e-pastu."""
    gmail_user     = os.getenv("GMAIL_USER")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")
    target_email   = os.getenv("TARGET_EMAIL")

    if not all([gmail_user, gmail_password, target_email]):
        return False, "❌ Nav iestatīti e-pasta mainīgie (GMAIL_USER, GMAIL_APP_PASSWORD, TARGET_EMAIL)."

    if not messages:
        return False, "❌ Čats ir tukšs — nav ko sūtīt."

    # Saņēmēju saraksts: TARGET_EMAIL + pieslēgtais lietotājs (ja atšķiras)
    recipients = [target_email]
    if user_email and user_email.lower() != target_email.lower():
        recipients.append(user_email)

    # Veido e-pasta saturu
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
    body = f"Horizon pārdošanas aģents — saruna {timestamp}\n"
    if user_email:
        body += f"Lietotājs: {user_email}\n"
    body += "=" * 60 + "\n\n"

    for msg in messages:
        role  = "👤 Klients" if msg["role"] == "user" else "🤖 Aģents"
        body += f"{role}:\n{msg['content']}\n\n"
        body += "-" * 40 + "\n\n"

    # Veido e-pastu
    email_msg = MIMEMultipart()
    email_msg["From"]    = gmail_user
    email_msg["To"]      = ", ".join(recipients)
    email_msg["Subject"] = f"Horizon aģents — saruna {timestamp}"
    email_msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, recipients, email_msg.as_string())
        saņēmēji_teksts = " un ".join(recipients)
        return True, f"✅ Saruna nosūtīta uz {saņēmēji_teksts}"
    except Exception as e:
        return False, f"❌ Sūtīšanas kļūda: {e}"


# ── Firmas.lv rekvizītu iegūšana ─────────────────────────────────────────────

def fetch_firmas_lv(vaicajums: str) -> dict:
    """Iegūst uzņēmuma rekvizītus no firmas.lv pēc reģ. numura, nosaukuma vai PVN numura."""
    import requests as req
    from bs4 import BeautifulSoup
    from urllib.parse import quote

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "lv,en;q=0.9",
    }

    vaicajums = vaicajums.strip()
    # PVN numurus normalizē: "LV40003734170" → "40003734170" (firmas.lv meklē bez "LV")
    meklet = vaicajums.upper().removeprefix("LV") if vaicajums.upper().startswith("LV") and vaicajums[2:].isdigit() else vaicajums

    # 1. Meklēšana pēc jebkura kritērija
    search_url = f"https://www.firmas.lv/lv/uznemumi/meklet?q={quote(meklet)}"
    try:
        r = req.get(search_url, headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        return {"kļūda": f"Nevar piekļūt firmas.lv: {e}"}

    soup = BeautifulSoup(r.text, "html.parser")

    # Atrod pirmo uzņēmuma saiti rezultātu sarakstā
    # (izslēdz pakalpojumu lapas, meklēšanas un nav uznemumi sadaļa)
    IZSLEGT = {"/meklet", "/pakalpojumi", "/par-mums", "/lv/tops",
               "/lv/personas", "/lv/adreses", "/industrijas"}
    company_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/uznemumi/" not in href:
            continue
        if any(x in href for x in IZSLEGT):
            continue
        # Pārliecinās, ka saite beidzas ar skaitli (reģ. numurs)
        segmenti = href.rstrip("/").split("/")
        if segmenti and segmenti[-1].isdigit():
            company_url = "https://www.firmas.lv" + href if href.startswith("/") else href
            break

    if not company_url:
        return {"kļūda": f"Uzņēmums '{vaicajums}' nav atrasts firmas.lv"}

    # 2. Iegūst uzņēmuma lapu
    try:
        r2 = req.get(company_url, headers=headers, timeout=10)
        r2.raise_for_status()
    except Exception as e:
        return {"kļūda": f"Nevar ielādēt uzņēmuma lapu: {e}"}

    soup2 = BeautifulSoup(r2.text, "html.parser")

    # Reģ. numuru izvelk no URL (pēdējais segments)
    url_reg = company_url.rstrip("/").split("/")[-1]

    rekviziti = {
        "nosaukums":        "",
        "reg_numurs":       url_reg,
        "juridiska_adrese": "",
        "pvn_numurs":       "",
        "talrunis":         "",
        "epasts":           "",
        "majas_lapa":       "",
        "url":              company_url,
    }

    # Nosaukums no h1
    h1 = soup2.find("h1")
    if h1:
        rekviziti["nosaukums"] = h1.get_text(strip=True)

    # Parsē "Pamatdati" tabulu — katras rindas pirmā šūna = etiķete, otrā = vērtība
    # Firmas.lv izmanto <th> etiķetēm un <td> vērtībām
    for table in soup2.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True, separator=" ")

            if "pvn numurs" in label or "pvn reģ" in label:
                # Izvelk tikai "LVxxxxxxxxxx" daļu
                import re as _re
                pvn_match = _re.search(r'LV\d+', value)
                if pvn_match:
                    rekviziti["pvn_numurs"] = pvn_match.group(0)

            elif "juridiskā adrese" in label or "juridiska adrese" in label:
                # Adrese ir pirmajā <a> tagā šūnā
                adrese_a = cells[1].find("a")
                if adrese_a:
                    rekviziti["juridiska_adrese"] = adrese_a.get_text(strip=True)
                else:
                    rekviziti["juridiska_adrese"] = cells[1].get_text(strip=True, separator=" ")

            elif "reģistrācijas numurs" in label:
                # "40003734170, 18.03.2005" → paņem tikai numuru
                reg_val = value.split(",")[0].strip()
                if reg_val.isdigit():
                    rekviziti["reg_numurs"] = reg_val

            elif "reģistrēts nosaukums" in label:
                if not rekviziti["nosaukums"]:
                    rekviziti["nosaukums"] = value

    # Kontakti no "Kontakti" sadaļas
    for a in soup2.find_all("a", href=True):
        href = a["href"]
        if href.startswith("tel:") and not rekviziti["talrunis"]:
            rekviziti["talrunis"] = href.replace("tel:", "").strip()
        elif href.startswith("mailto:") and not rekviziti["epasts"]:
            rekviziti["epasts"] = href.replace("mailto:", "").strip()
        elif (href.startswith("http") and
              "firmas.lv" not in href and
              "vid.gov" not in href and
              "ec.europa" not in href and
              not rekviziti["majas_lapa"]):
            teksts_a = a.get_text(strip=True)
            if teksts_a.startswith("http") or "www." in teksts_a:
                rekviziti["majas_lapa"] = teksts_a

    return rekviziti


# ── Līguma sagatavošana ───────────────────────────────────────────────────────

# Šie mainīgo nosaukumi jālieto Word šablonā: {{ mainīgā_nosaukums }}
LIGUMA_MAINĪGIE_NOKLUSĒJUMS = {
    # Klienta rekvizīti (daļēji no firmas.lv)
    "uznemuma_nosaukums":    "",   # {{ uznemuma_nosaukums }}
    "reg_numurs":            "",   # {{ reg_numurs }}
    "pvn_numurs":            "",   # {{ pvn_numurs }}
    "juridiska_adrese":      "",   # {{ juridiska_adrese }}
    "fakt_adrese":           "",   # {{ fakt_adrese }}
    "talrunis":              "",   # {{ talrunis }}
    "epasts":                "",   # {{ epasts }}
    # Parakstītājs (no sarakstes)
    "paraksta":              "",   # {{ paraksta }}
    "paraksta_amats":        "",   # {{ paraksta_amats }}
    "pamat_uz":              "",   # {{ pamat_uz }} — statūti / prokūra / pilnvara
    # Kontaktpersona (no sarakstes)
    "kontaktpersona":        "",   # {{ kontaktpersona }}
    "kontaktpersonas_amats": "",   # {{ kontaktpersonas_amats }}
    "kontaktp_epasts":       "",   # {{ kontaktp_epasts }}
    "kontaktp_talrunis":     "",   # {{ kontaktp_talrunis }}
    # Līguma dati
    "datums":                "",   # {{ datums }} — aizpildās automātiski
    "liguma_numurs":         "",   # {{ liguma_numurs }} — jāaizpilda manuāli
    "lpp":                   "",   # {{ lpp }} — lappušu skaits, jāaizpilda manuāli
}


def extract_liguma_mainīgie(messages: list, model_name: str, rekviziti: dict = None) -> dict:
    """Izvelk līguma aizpildīšanai nepieciešamos mainīgos no sarakstes un rekvizītiem."""
    import json as _json

    mainīgie = dict(LIGUMA_MAINĪGIE_NOKLUSĒJUMS)
    mainīgie["datums"] = datetime.now().strftime("%d.%m.%Y")

    # 1. Aizpilda no firmas.lv rekvizītiem (ja ir)
    if rekviziti:
        mainīgie.update({
            "uznemuma_nosaukums": rekviziti.get("nosaukums", ""),
            "reg_numurs":         rekviziti.get("reg_numurs", ""),
            "pvn_numurs":         rekviziti.get("pvn_numurs", ""),
            "juridiska_adrese":   rekviziti.get("juridiska_adrese", ""),
            "talrunis":           rekviziti.get("talrunis", ""),
            "epasts":             rekviziti.get("epasts", ""),
        })

    if not messages:
        return mainīgie

    # 2. AI izvelk trūkstošo info no sarakstes
    history_text = "\n".join([
        f"{'Klients' if m['role'] == 'user' else 'Aģents'}: {m['content']}"
        for m in messages
    ])

    jau_zinami = {k: v for k, v in mainīgie.items() if v}
    jau_zinami_teksts = "\n".join(f"  {k}: {v}" for k, v in jau_zinami.items()) or "  (nekas nav zināms)"

    prompt = f"""No šīs pārdošanas sarakstes izvelc informāciju līgumam. Atbildi TIKAI ar JSON objektu, bez papildu teksta.

Sarakstes vēsture:
{history_text[-4000:]}

Jau zināmie dati (neaizstāj ar tukšiem!):
{jau_zinami_teksts}

Izvelc šādus laukus (ja nav atrodams — atstāj ""):
{{
  "uznemuma_nosaukums": "uzņēmuma pilnais nosaukums",
  "paraksta": "personas vārds, uzvārds, kas parakstīs līgumu",
  "paraksta_amats": "parakstītāja amats",
  "pamat_uz": "paraksta pamatojoties uz (piemēram: statūtiem, prokūru, pilnvaru Nr. X)",
  "fakt_adrese": "faktiskā adrese, ja atšķiras no juridiskās",
  "kontaktpersona": "kontaktpersonas vārds, uzvārds",
  "kontaktpersonas_amats": "kontaktpersonas amats",
  "kontaktp_epasts": "kontaktpersonas e-pasts",
  "kontaktp_talrunis": "kontaktpersonas tālrunis",
  "reg_numurs": "reģistrācijas numurs, ja minēts"
}}"""

    model_cfg = MODELS[model_name]
    try:
        if model_cfg["provider"] == "mistral":
            client = load_mistral_client()
        else:
            client = load_gemini_client()
        if not client:
            return mainīgie

        response = client.chat.completions.create(
            model=model_cfg["model"],
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        # Izņem JSON no atbildes (arī ja ietīts ```json ... ```)
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            extracted = _json.loads(json_match.group(0))
            for k, v in extracted.items():
                if v and k in mainīgie and not mainīgie.get(k):
                    mainīgie[k] = str(v)
    except Exception:
        pass

    return mainīgie


_TUKSS_PREFIKS = "_TUKSS_"  # + lauka nosaukums, piemēram "_TUKSS_uznemuma_nosaukums"
_LPP           = "_LPP_"   # marķieris lpp → NUMPAGES lauka kods


def _postprocess_ligums(doc_bytes: bytes) -> bytes:
    """Pēc docxtpl renderēšanas:
    - aizstāj _AIZPILDIT_ ar dzeltenā iezīmētām atstarpēm
    - aizstāj _LPP_ ar Word NUMPAGES lauka kodu
    """
    from docx import Document
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from docx.enum.text import WD_COLOR_INDEX

    buf = io.BytesIO(doc_bytes)
    doc = Document(buf)

    def _numpages_xml(run):
        """Iestata NUMPAGES lauku dotajā run elementā."""
        run.text = ""
        fc_begin = OxmlElement("w:fldChar")
        fc_begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = " NUMPAGES "
        fc_sep = OxmlElement("w:fldChar")
        fc_sep.set(qn("w:fldCharType"), "separate")
        fc_end = OxmlElement("w:fldChar")
        fc_end.set(qn("w:fldCharType"), "end")
        run._r.extend([fc_begin, instr, fc_sep, fc_end])

    def _process_run(run):
        if _LPP in run.text:
            # Aizstāj ar NUMPAGES lauku
            prefix, suffix = run.text.split(_LPP, 1)
            run.text = prefix
            _numpages_xml(run)
            # Ja pēc marķiera ir teksts — pievieno jaunu run
            if suffix:
                new_run = OxmlElement("w:r")
                new_t = OxmlElement("w:t")
                new_t.text = suffix
                new_run.append(new_t)
                run._r.addnext(new_run)
        elif _TUKSS_PREFIKS in run.text:
            # Izvelk lauka nosaukumu no marķiera un iezīmē dzeltenā
            # Piemēram "_TUKSS_uznemuma_nosaukums" → "uznemuma_nosaukums" (dzeltenā)
            run.text = run.text.replace(_TUKSS_PREFIKS, "")
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    for para in doc.paragraphs:
        for run in para.runs:
            _process_run(run)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        _process_run(run)

    buf_out = io.BytesIO()
    doc.save(buf_out)
    return buf_out.getvalue()


def generate_ligums_docx(messages: list, model_name: str,
                          rekviziti: dict = None) -> tuple:
    """Ģenerē līguma Word dokumentu no docxtpl šablona.

    Atgriež: (bytes, mainīgie_dict) ja veiksmīgi, (None, kļūdas_teksts) ja neizdevās.
    """
    try:
        from docxtpl import DocxTemplate
    except ImportError:
        return None, "❌ docxtpl nav instalēts. Pievieno 'docxtpl' requirements.txt un pārinstallē."

    template_path = os.path.join(BASE_DIR, "assets", "liguma_sablons.docx")
    if not os.path.exists(template_path):
        return None, (
            "❌ Šablons nav atrasts (assets/liguma_sablons.docx).\n"
            "Izveido Word dokumentu ar {{ mainīgais }} atzīmēm un saglabā kā assets/liguma_sablons.docx."
        )

    # Izvelk mainīgos no sarakstes + rekvizītiem
    mainīgie = extract_liguma_mainīgie(messages, model_name, rekviziti)

    # Sagatavo kontekstu docxtpl: tukšie → marķieris, lpp → LPP marķieris
    konteksts = {}
    for k, v in mainīgie.items():
        if k == "lpp":
            konteksts[k] = _LPP          # aizstāj ar NUMPAGES pēc renderēšanas
        elif not v:
            konteksts[k] = f"{_TUKSS_PREFIKS}{k}"  # "_TUKSS_uznemuma_nosaukums" u.tml.
        else:
            konteksts[k] = v

    try:
        doc = DocxTemplate(template_path)
        doc.render(konteksts)
        buf = io.BytesIO()
        doc.save(buf)
        # Pēcapstrāde: dzeltenā iezīmēšana + NUMPAGES lauks
        doc_bytes = _postprocess_ligums(buf.getvalue())
        return doc_bytes, mainīgie
    except Exception as e:
        return None, f"❌ Kļūda aizpildot šablonu: {e}"


# ── Ieviešanas tāme ───────────────────────────────────────────────────────────

# Visi iespējamie bloki — atslēgvārdi no sarakstes → atbilstošās sadaļas šablonā
TAME_BLOCKS = {
    "instalācija":              ["Instalācija"],
    "pamatsistēma":             ["Instalācija", "Pamatsistēma"],
    "grāmatvedība":             ["Instalācija", "Pamatsistēma"],
    "algas":                    ["Instalācija", "Pamatsistēma"],
    "personāls":                ["Instalācija", "Pamatsistēma", "Personāls"],
    "hop personāls":            ["Instalācija", "Pamatsistēma", "Personāls"],
    "pieteikumi":               ["Personāls"],
    "komandējumi":              ["Personāls"],
    "mani izdevumi":            ["Personāls"],
    "rīkojumi":                 ["Personāls"],
    "darba laika uzskaite":     ["Personāls"],
    "hop rēķini":               ["Grāmatvedība +"],
    "rēķini":                   ["Grāmatvedība +"],
    "numo":                     ["Numo"],
    "darba laika plānošana":    ["Numo"],
}

TAME_CLARIFYING_QUESTIONS = """Ja klients jautā par ieviešanu vai ieviešanas izmaksām, OBLIGĀTI uzdod šos precizējošos jautājumus PIRMS tāmes sagatavošanas:
1. Vai nepieciešama Pamatsistēma (grāmatvedība, rēķini, noliktava)?
2. Vai nepieciešams Algu un personāla modulis?
3. Vai nepieciešams HOP Personāls (darbinieku pieteikumi, komandējumi, izdevumi, rīkojumi)?
4. Vai nepieciešama HOP Darba laika uzskaite?
5. Vai nepieciešami HOP Rēķini (rēķinu saskaņošana)?
6. Vai nepieciešama NUMO Darba laika plānošana?
7. Cik lietotāji strādās sistēmā?
Neģenerē tāmi kamēr nav saņemtas atbildes uz šiem jautājumiem."""


def determine_tame_sections(messages: list, model_name: str) -> list[str]:
    """AI nosaka kuras sadaļas iekļaut tāmē, balstoties uz sarakstes."""
    history_text = ""
    for msg in messages:
        role = "Klients" if msg["role"] == "user" else "Aģents"
        history_text += f"{role}: {msg['content']}\n\n"

    prompt = f"""No šīs sarakstes nosaki, kuras Horizon ieviešanas sadaļas ir nepieciešamas klientam.
Atbildi TIKAI ar JSON sarakstu no šiem variantiem (iekļauj tikai vajadzīgos):
["Instalācija", "Pamatsistēma", "Personāls", "Grāmatvedība +", "Numo", "Projekta vadība"]

Noteikumi:
- "Instalācija" un "Projekta vadība" — vienmēr iekļauj
- "Pamatsistēma" — iekļauj ja minēta grāmatvedība, uzskaite, pamatsistēma, algas
- "Personāls" — iekļauj ja minēts personāls, HOP, pieteikumi, komandējumi, rīkojumi, darba laiks
- "Grāmatvedība +" — iekļauj ja minēti HOP rēķini vai rēķinu saskaņošana
- "Numo" — iekļauj ja minēta darba laika plānošana vai NUMO

Sarakstes vēsture:
{history_text}

Atbildi TIKAI JSON formātā, piemēram: ["Instalācija", "Pamatsistēma", "Projekta vadība"]"""

    model_cfg = MODELS[model_name]
    try:
        if model_cfg["provider"] == "mistral":
            client = load_mistral_client()
        else:
            client = load_gemini_client()
        if not client:
            return ["Instalācija", "Pamatsistēma", "Projekta vadība"]

        response = client.chat.completions.create(
            model=model_cfg["model"],
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        text = response.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        sections = json.loads(text)
        # Vienmēr pievieno Instalācija un Projekta vadība
        for always in ["Instalācija", "Projekta vadība"]:
            if always not in sections:
                sections.append(always)
        return sections
    except Exception:
        return ["Instalācija", "Pamatsistēma", "Projekta vadība"]


def generate_tame_excel(messages: list, model_name: str) -> tuple[bytes, str]:
    """Ģenerē ieviešanas tāmi Excel formātā, filtrējot blokus pēc sarakstes."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from copy import copy

    template_path = os.path.join(BASE_DIR, "assets", "tame_template.xlsx")
    if not os.path.exists(template_path):
        return None, "❌ Tāmes šablons nav atrasts (assets/tame_template.xlsx)"

    # Nosaka nepieciešamās sadaļas
    sections = determine_tame_sections(messages, model_name)

    # Iegūst klienta vārdu
    summary = summarize_chat_for_proposal(messages, model_name)
    klients = summary.get("klients", "Nav norādīts")
    timestamp = datetime.now().strftime("%d.%m.%Y")

    # Ielādē šablonu
    wb = openpyxl.load_workbook(template_path)
    ws = wb.active

    # Atrod rindas ko dzēst (sadaļa nav iekļauta)
    rows_to_delete = []
    for row_idx in range(2, ws.max_row + 1):
        sadala = ws.cell(row=row_idx, column=1).value
        if sadala and sadala not in sections:
            rows_to_delete.append(row_idx)

    # Dzēš rindas no apakšas uz augšu
    for row_idx in reversed(rows_to_delete):
        ws.delete_rows(row_idx)

    # Iestata wrap_text un aprēķina rindu augstumu visām šūnām
    COL_WIDTHS = {1: 18, 2: 22, 3: 60, 4: 16, 5: 12, 6: 14, 7: 18}
    # Efektīvais rakstzīmju skaits uz rindu (mazāks nekā kolonnas platums — fonts un padding)
    EFFECTIVE_CHARS = {1: 15, 2: 18, 3: 48, 4: 13, 5: 10, 6: 11, 7: 15}
    LINE_HEIGHT = 15  # Excel punkti uz teksta rindu
    MIN_HEIGHT  = 32

    for row in ws.iter_rows():
        max_lines = 1
        for cell in row:
            if cell.value and isinstance(cell.value, str):
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                eff_chars = EFFECTIVE_CHARS.get(cell.column, 18)
                # Sadala pēc \n, tad aprēķina wrapping katrai rindai
                text_lines = cell.value.split("\n")
                lines = 0
                for line in text_lines:
                    line = line.strip()
                    if line:
                        lines += max(1, -(-len(line) // eff_chars))  # ceiling division
                    else:
                        lines += 1  # tukša rinda
                max_lines = max(max_lines, lines)
        row_height = max(MIN_HEIGHT, max_lines * LINE_HEIGHT)
        ws.row_dimensions[row[0].row].height = row_height

    # Iestata kolonnu platumu
    for col_idx, width in COL_WIDTHS.items():
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = width

    # Pievieno klienta info virsrakstā (1. rinda pirms tabulas)
    ws.insert_rows(1)
    ws.insert_rows(1)
    ws["A1"] = f"Ieviešanas tāme — {klients}"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A1"].alignment = Alignment(wrap_text=False)
    ws.row_dimensions[1].height = 25
    ws["A2"] = f"Sagatavots: {timestamp}"
    ws["A2"].font = Font(italic=True)
    ws["A2"].alignment = Alignment(wrap_text=False)
    ws.row_dimensions[2].height = 20

    # Saglabā atmiņā
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue(), sections


# ── Autentifikācija ───────────────────────────────────────────────────────────

def load_allowed_emails() -> set:
    """Ielādē atļauto e-pastu sarakstu.

    Avotu prioritāte:
    1. Streamlit Secrets: ALLOWED_EMAILS = "a@b.com,c@d.com"
    2. Fails allowed_emails.txt (viens e-pasts uz rindas, # — komentāri)
    """
    # 1. Streamlit Secrets
    try:
        raw = st.secrets.get("ALLOWED_EMAILS", "")
        if raw:
            return {e.strip().lower() for e in raw.split(",") if e.strip()}
    except Exception:
        pass

    # 2. Fails
    emails_file = os.path.join(BASE_DIR, "allowed_emails.txt")
    if os.path.exists(emails_file):
        emails = set()
        with open(emails_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    emails.add(line.lower())
        return emails

    return set()


def show_login():
    """Rāda pieteikšanās lapu. Atgriež True, ja lietotājs autentificēts."""
    st.set_page_config(
        page_title="Pārdošanas aģents — pieteikšanās",
        page_icon="🔐",
        layout="centered",
    )

    # Centrētā karte
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        logo_path = os.path.join(BASE_DIR, "assets", "logo.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path, width=160)

        st.markdown("## 🔐 Pieteikšanās")
        st.markdown("Ievadi savu e-pasta adresi, lai piekļūtu sistēmai.")

        with st.form("login_form"):
            epasts = st.text_input("E-pasts", placeholder="vards.uzvards@uznemums.lv")
            submit = st.form_submit_button("Pieteikties", use_container_width=True)

        if submit:
            allowed = load_allowed_emails()
            if not allowed:
                st.error("⚠️ Nav konfigurēts atļauto lietotāju saraksts.")
                return False
            if epasts.strip().lower() in allowed:
                st.session_state.authenticated      = True
                st.session_state.authenticated_user = epasts.strip().lower()
                st.rerun()
            else:
                st.error("❌ Šis e-pasts nav reģistrēts. Sazinieties ar sistēmas administratoru.")

    return False


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
    collection = load_collection()

    # ── Čats ──────────────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                tables = extract_markdown_tables(msg["content"])
                if tables:
                    excel_bytes = tables_to_excel_bytes(tables)
                    ts = msg.get("timestamp", str(idx))
                    st.download_button(
                        label="📥 Lejupielādēt kā Excel",
                        data=excel_bytes,
                        file_name=f"horizon_aprekins_{ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_{idx}",
                    )
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
                    answer = ask_ai(question, context, st.session_state.selected_model, st.session_state.messages)

            st.markdown(answer)
            # Ja atbildē ir tabula — piedāvā Excel lejupielādi
            tables = extract_markdown_tables(answer)
            if tables:
                excel_bytes = tables_to_excel_bytes(tables)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Lejupielādēt kā Excel",
                    data=excel_bytes,
                    file_name=f"horizon_aprekins_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            if sources:
                with st.expander("📎 Avoti"):
                    for src in sources:
                        st.text(f"• {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        })

    # Sānjosla
    with st.sidebar:
        # Modeļa izvēle paslēpta — noklusēti Mistral Small
        # st.header("🤖 AI modelis")
        # st.session_state.selected_model = st.selectbox(...)
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "🟠 Mistral Small"
        st.divider()
        st.header("🏢 Klienta rekvizīti")
        reg_input = st.text_input(
            "Meklēt uzņēmumu:",
            placeholder="reģ. nr., nosaukums vai PVN nr.",
            key="firmas_meklet_input",
        )
        if st.button("🔍 Iegūt rekvizītus"):
            if not reg_input.strip():
                st.warning("⚠️ Ievadi reģistrācijas numuru, nosaukumu vai PVN numuru.")
            else:
                with st.spinner("Meklē firmas.lv..."):
                    rek = fetch_firmas_lv(reg_input.strip())
                if "kļūda" in rek:
                    st.error(rek["kļūda"])
                else:
                    st.session_state.klienta_rekviziti = rek
                    st.success(f"✅ {rek.get('nosaukums', 'Atrasts!')}")
        if "klienta_rekviziti" in st.session_state and st.session_state.klienta_rekviziti:
            rek = st.session_state.klienta_rekviziti
            with st.expander("📋 Rekvizīti", expanded=True):
                for atslega, nosaukums in [
                    ("nosaukums",        "Nosaukums"),
                    ("reg_numurs",       "Reģ. nr."),
                    ("juridiska_adrese", "Juridiskā adrese"),
                    ("pvn_numurs",       "PVN nr."),
                    ("talrunis",         "Tālrunis"),
                    ("epasts",           "E-pasts"),
                    ("majas_lapa",       "Mājas lapa"),
                ]:
                    val = rek.get(atslega, "")
                    if val:
                        st.caption(f"**{nosaukums}:** {val}")
                st.markdown(f"[🔗 firmas.lv]({rek.get('url', '')})")
        st.divider()
        st.header("ℹ️ Informācija")
        try:
            st.metric("Indeksēti fragmenti", collection.count())
        except Exception:
            pass
        if st.button("📊 Sagatavot ieviešanas tāmi"):
            if not st.session_state.messages:
                st.warning("⚠️ Uzsāc sarakstes pirms tāmes sagatavošanas.")
            else:
                with st.spinner("Analizē sarakstes un sagatavo tāmi..."):
                    excel_bytes, result = generate_tame_excel(
                        st.session_state.messages,
                        st.session_state.selected_model,
                    )
                if excel_bytes:
                    sadaļas = ", ".join(result) if isinstance(result, list) else ""
                    st.success(f"✅ Iekļautās sadaļas: {sadaļas}")
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📥 Lejupielādēt tāmi",
                        data=excel_bytes,
                        file_name=f"horizon_tame_{ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="tame_download",
                    )
                else:
                    st.error(result)
        st.divider()
        if st.button("📝 Sagatavot līgumu"):
            if not st.session_state.messages:
                st.warning("⚠️ Uzsāc sarakstes pirms līguma sagatavošanas.")
            else:
                rekviziti = st.session_state.get("klienta_rekviziti")
                with st.spinner("Analizē sarakstes un sagatavo līgumu..."):
                    doc_bytes, rezultats = generate_ligums_docx(
                        st.session_state.messages,
                        st.session_state.selected_model,
                        rekviziti,
                    )
                if doc_bytes:
                    st.session_state.ligums_bytes    = doc_bytes
                    st.session_state.ligums_mainīgie = rezultats
                else:
                    st.error(rezultats)
        if st.session_state.get("ligums_bytes"):
            mainīgie_dict = st.session_state.get("ligums_mainīgie", {})
            klienta_nos   = mainīgie_dict.get("klienta_nosaukums", "ligums") if isinstance(mainīgie_dict, dict) else "ligums"
            ts = datetime.now().strftime("%Y%m%d")
            with st.expander("📋 Aizpildītie dati", expanded=False):
                if isinstance(mainīgie_dict, dict):
                    LAUKU_NOSAUKUMI = {
                        "uznemuma_nosaukums":    "Nosaukums",
                        "reg_numurs":            "Reģ. nr.",
                        "pvn_numurs":            "PVN nr.",
                        "juridiska_adrese":      "Juridiskā adrese",
                        "fakt_adrese":           "Faktiskā adrese",
                        "talrunis":              "Tālrunis",
                        "epasts":                "E-pasts",
                        "paraksta":              "Paraksta",
                        "paraksta_amats":        "Parakstītāja amats",
                        "pamat_uz":              "Pamatojums",
                        "kontaktpersona":        "Kontaktpersona",
                        "kontaktpersonas_amats": "Kontaktpersonas amats",
                        "kontaktp_epasts":       "Kontaktpersonas e-pasts",
                        "kontaktp_talrunis":     "Kontaktpersonas tālrunis",
                        "datums":                "Datums",
                        "liguma_numurs":         "Līguma nr.",
                        "lpp":                   "Lappušu skaits",
                    }
                    for k, label in LAUKU_NOSAUKUMI.items():
                        val = mainīgie_dict.get(k, "")
                        st.caption(f"**{label}:** {val or '—'}")
            st.download_button(
                label="📥 Lejupielādēt līgumu",
                data=st.session_state.ligums_bytes,
                file_name=f"horizon_ligums_{klienta_nos}_{ts}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="ligums_download",
            )
        st.divider()
        # Qwilr poga pagaidām paslēpta (API pieejams tikai maksas plānā)
        # if st.button("📄 Sagatavot Qwilr piedāvājumu"):
        #     ...
        st.divider()
        if st.button("📧 Nosūtīt sarunu uz e-pastu"):
            ok, msg = send_chat_by_email(
                st.session_state.messages,
                st.session_state.get("authenticated_user", ""),
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        if st.button("🗑️ Notīrīt čatu"):
            st.session_state.messages = []
            st.rerun()
        if st.button("🔄 Pārindeksēt dokumentus"):
            st.cache_resource.clear()
            st.rerun()
        st.divider()
        user = st.session_state.get("authenticated_user", "")
        st.caption(f"👤 {user}")
        if st.button("🚪 Izrakstīties"):
            st.session_state.authenticated      = False
            st.session_state.authenticated_user = ""
            st.rerun()


if __name__ == "__main__":
    if not st.session_state.get("authenticated"):
        show_login()
    else:
        main()
