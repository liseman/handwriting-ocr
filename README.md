# Handwriting OCR

A web application that converts handwritten documents to text using OCR, then learns your handwriting over time through corrections. Upload photos of handwritten pages, get instant transcriptions, correct mistakes through a gamified "Play" mode, and watch accuracy improve as the model fine-tunes to your writing style.

## Features

- **Dual OCR Engine** -- Gemini Flash API (primary, high-quality) with TrOCR local fallback
- **Auto-rotation** -- Automatically detects and corrects image orientation
- **Auto-crop** -- Detects content bounds to focus OCR on the writing area
- **Line Detection** -- Paper-aware ink detection with horizontal projection profiles finds individual text lines
- **Custom Bounding Boxes** -- Draw boxes on the page to OCR specific regions
- **Play Mode** -- Gamified correction interface that prioritizes low-confidence results
- **Personalized Fine-tuning** -- Per-user LoRA adapters trained on your corrections
- **Calibration** -- Bootstrap training with a single handwriting sample
- **Google Photos Import** -- Import photos directly via the Google Photos Picker API
- **Full-text Search** -- Whoosh-indexed search across all your transcribed documents
- **Model Export** -- Download your personalized LoRA weights

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Google Cloud project with OAuth 2.0 credentials
- (Optional) A [Gemini API key](https://aistudio.google.com/apikey) for high-quality OCR

### Setup

```bash
# Clone
git clone https://github.com/lukeiseman/handwriting-ocr.git
cd handwriting-ocr

# Configure
cp .env.example .env
# Edit .env with your Google OAuth credentials and (optional) Gemini API key

# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend (in a separate terminal)
cd frontend
npm install
```

### Run

```bash
# Backend (from backend/)
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Frontend (from frontend/)
npm run dev
```

Open http://localhost:5173 in your browser. The frontend dev server proxies API requests to the backend on port 8000.

### Docker

```bash
docker compose up --build
# App available at http://localhost:8000
```

## Architecture

```
handwriting-ocr/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, CORS, lifespan
│   │   ├── auth.py          # JWT token verification
│   │   ├── config.py        # Pydantic settings
│   │   ├── database.py      # SQLAlchemy async engine
│   │   ├── models.py        # ORM models (User, Document, Page, OcrResult, Correction, UserModel)
│   │   ├── schemas.py       # Pydantic request/response schemas
│   │   ├── ocr.py           # OCR engines (Gemini + TrOCR), line detection, ink analysis
│   │   ├── finetune.py      # LoRA fine-tuning pipeline
│   │   └── routes/
│   │       ├── auth.py          # Google OAuth login/callback, JWT
│   │       ├── documents.py     # Upload, camera, CRUD, rotate, crop
│   │       ├── ocr.py           # Trigger OCR, get results, process bbox
│   │       ├── corrections.py   # Submit corrections, Play mode batches
│   │       ├── search.py        # Full-text search (Whoosh)
│   │       ├── photos.py        # Google Photos Picker import
│   │       └── model.py         # Training, calibration, export
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js               # Axios client + all API functions
│   │   ├── hooks/useAuth.jsx    # Auth context + token management
│   │   ├── components/
│   │   │   ├── PageViewer.jsx           # Image viewer with bbox overlays, draw/crop modes
│   │   │   └── BboxHighlightViewer.jsx  # Bbox drawing component (Play, Calibrate)
│   │   └── pages/
│   │       ├── Login.jsx         # Google OAuth sign-in
│   │       ├── Dashboard.jsx     # Document list, upload, navigation
│   │       ├── Upload.jsx        # Multi-source upload (file, camera, Google Photos)
│   │       ├── DocumentView.jsx  # Page viewer + OCR results panel
│   │       ├── Play.jsx          # Correction game with speech-to-text
│   │       ├── Search.jsx        # Full-text search
│   │       ├── Model.jsx         # Training controls + export
│   │       └── Calibrate.jsx     # Bootstrap training with sample text
│   └── package.json
├── Dockerfile              # Multi-stage (Node build + Python runtime)
├── docker-compose.yml
├── .env.example
└── CLAUDE.md               # AI assistant instructions
```

## How It Works

### Data Flow

1. **Upload** -- Images saved to `data/uploads/user_{id}/`, Document + Page rows created
2. **OCR** -- Auto-rotation detects orientation, line detection finds text regions, OCR engine transcribes each line, results stored with confidence scores and bounding boxes
3. **Correct** -- Play mode surfaces lowest-confidence results first, user approves or corrects
4. **Train** -- Corrections crop original images to bounding boxes, LoRA fine-tunes TrOCR decoder attention layers (q_proj, v_proj), new adapter version saved
5. **Improve** -- Subsequent OCR loads user's LoRA adapter for better accuracy

### OCR Pipeline

The OCR pipeline uses a dual-engine approach:

**Gemini Flash** (when `GEMINI_API_KEY` is set):
- Sends full page image to Gemini for transcription
- Auto-rotation via orientation detection prompt
- Line positions mapped using paper-aware ink detection + horizontal projection peaks

**TrOCR** (local fallback):
- `microsoft/trocr-large-handwritten` via HuggingFace Transformers
- Line segmentation via horizontal projection profiles
- Auto-rotation by trying all 4 orientations, keeping highest confidence
- Confidence from average token log-probabilities

**Line Detection** (`_find_line_positions`):
- Paper-aware ink detection separates ink from paper background
- Spine removal for notebook photos (only narrow vertical features)
- Gaussian-smoothed horizontal projection with adaptive sigma
- Peak detection with midpoint boundaries between lines
- Per-line horizontal extent with gap splitting for two-page spreads

### Fine-tuning

Per-user LoRA adapters via the PEFT library:
- **Rank:** 8, **Alpha:** 16
- **Target:** Decoder attention (q_proj, v_proj)
- **Storage:** `data/models/user_{id}/v{N}/`
- **Training:** CPU-compatible, small batches from correction pairs

## Configuration

Copy `.env.example` to `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_CLIENT_ID` | Yes | Google Cloud OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | Yes | Google Cloud OAuth 2.0 client secret |
| `SECRET_KEY` | Yes | Random string for JWT signing |
| `GEMINI_API_KEY` | No | Gemini API key for high-quality OCR |
| `DATABASE_URL` | No | SQLite path (default: `sqlite:///./data/app.db`) |
| `UPLOAD_DIR` | No | Upload directory (default: `./data/uploads`) |
| `MODEL_DIR` | No | Model directory (default: `./data/models`) |

### Google Cloud Setup

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the **Google Photos Picker API**
3. Create OAuth 2.0 credentials (Web application)
4. Set authorized redirect URI to `http://localhost:8000/auth/callback`
5. Copy client ID and secret to `.env`

## API Endpoints

### Authentication
| Method | Path | Description |
|--------|------|-------------|
| GET | `/auth/login` | Redirect to Google OAuth |
| GET | `/auth/callback` | OAuth callback, mints JWT |
| GET | `/auth/me` | Current user profile |
| POST | `/auth/logout` | Clear session |

### Documents
| Method | Path | Description |
|--------|------|-------------|
| GET | `/documents` | List all documents |
| POST | `/documents/upload` | Upload image files |
| POST | `/documents/camera` | Camera capture (base64) |
| GET | `/documents/{id}` | Document with pages + results |
| DELETE | `/documents/{id}` | Delete document and files |
| POST | `/documents/pages/{id}/rotate` | Rotate page image |
| POST | `/documents/pages/{id}/crop` | Set crop region |
| POST | `/documents/pages/{id}/crop/clear` | Clear crop |
| POST | `/documents/pages/{id}/crop/auto` | Auto-detect crop |

### OCR
| Method | Path | Description |
|--------|------|-------------|
| POST | `/ocr/process/{page_id}` | Trigger OCR (background) |
| POST | `/ocr/process-document/{doc_id}` | OCR all pages |
| POST | `/ocr/process-bbox/{page_id}` | OCR drawn region (sync) |
| GET | `/ocr/results/{page_id}` | Get OCR results |
| GET | `/ocr/processing-status` | Poll processing state |

### Corrections
| Method | Path | Description |
|--------|------|-------------|
| POST | `/corrections` | Submit correction |
| GET | `/corrections` | List corrections |
| GET | `/corrections/play` | Get Play mode batch |
| POST | `/corrections/play/submit` | Submit Play correction |

### Search, Photos, Model
| Method | Path | Description |
|--------|------|-------------|
| GET | `/search` | Full-text search |
| POST | `/photos/picker/session` | Create Photos Picker session |
| GET | `/photos/picker/session/{id}` | Poll picker status |
| POST | `/photos/picker/import` | Import selected photos |
| GET | `/model/status` | Model version + stats |
| POST | `/model/train` | Start fine-tuning |
| POST | `/model/calibrate` | Calibrate with sample |
| GET | `/model/export` | Download LoRA weights |

## Tech Stack

**Backend:** FastAPI, SQLAlchemy 2.0 (async), aiosqlite, python-jose (JWT), authlib (OAuth)

**Frontend:** React 19, Vite 6, Tailwind CSS 4, React Router 6, TanStack Query 5, Axios

**ML:** Transformers (TrOCR), PEFT (LoRA), PyTorch, Google GenAI (Gemini Flash)

**Search:** Whoosh

## Deployment

Target: **Google Cloud Run** with persistent volume for `data/`.

The Dockerfile is multi-stage: Node builds the frontend, Python serves both the API and static assets. Mount `data/` as a persistent volume or use GCS for production storage.

## License

MIT
