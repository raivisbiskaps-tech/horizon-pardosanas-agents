# 📚 Dokumentācijas Q&A Asistents

RAG (Retrieval-Augmented Generation) aplikācija, kas ļauj lietotājiem uzdot jautājumus par dokumentāciju dabiskā valodā. Atbildes tiek sniegtas, balstoties tikai uz indeksētajiem dokumentiem.

## Projekta struktūra

```
docs-qa/
├── docs/           ← Šeit liec savus dokumentus
├── chroma_db/      ← Automātiski izveidojas pēc ingest.py
├── ingest.py       ← Indeksēšanas skripts (izpildi vienreiz)
├── app.py          ← Streamlit web aplikācija
├── requirements.txt
└── .env.example
```

## Uzstādīšana

### 1. Klonē/kopē projektu un uzstādi atkarības

```bash
pip install -r requirements.txt
```

### 2. Izveido .env failu

```bash
cp .env.example .env
```

Atver `.env` un ieraksti savu Anthropic API atslēgu:
```
ANTHROPIC_API_KEY=sk-ant-...
```

API atslēgu iegūst: https://console.anthropic.com/

### 3. Pievieno dokumentus

Nokopē visus dokumentācijas failus mapē `./docs/`.

Atbalstītie formāti:
- PDF (`.pdf`)
- Word (`.docx`, `.doc`)
- Excel (`.xlsx`, `.xls`)
- Markdown (`.md`)
- HTML (`.html`, `.htm`)

### 4. Indeksē dokumentus

```bash
python ingest.py
```

Šis solis jāatkārto tikai tad, ja dokumentācija mainās.

Ja dokumenti atrodas citā mapē:
```bash
python ingest.py --docs_dir /cels/uz/dokumentiem
```

### 5. Palaid aplikāciju

```bash
streamlit run app.py
```

Pārlūkā atveras: http://localhost:8501

## Kā tas darbojas

```
Lietotājs → Jautājums
              ↓
         ChromaDB (vektoru meklēšana)
              ↓
         Relevantu fragmentu atlase
              ↓
         Claude API (atbild balstoties uz fragmentiem)
              ↓
         Atbilde + avotu norādes
```

## Konfigurācija

Galvenie parametri `ingest.py` failā:
| Parametrs | Noklusējums | Apraksts |
|-----------|-------------|----------|
| `CHUNK_SIZE` | 800 | Fragmenta lielums simbolos |
| `CHUNK_OVERLAP` | 150 | Pārlappošanās starp fragmentiem |

Galvenie parametri `app.py` failā:
| Parametrs | Noklusējums | Apraksts |
|-----------|-------------|----------|
| `TOP_K_RESULTS` | 5 | Cik fragmentus iegūt meklēšanā |
| `CLAUDE_MODEL` | claude-sonnet-4-6 | Izmantojamais Claude modelis |

## Dokumentācijas atjaunināšana

Kad dokumentācija mainās, vienkārši izpildi atkārtoti:
```bash
python ingest.py
```

Aplikācija turpina darboties — nākamais jautājums izmantos jauno indeksu.
"# horizon-pardosanas-agents" 
