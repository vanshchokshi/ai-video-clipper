# 🎬 AI Video Clipper Studio

Transform long-form videos, podcasts, interviews, and streams into high-retention, viral **9:16 vertical Reels, TikToks, and YouTube Shorts** powered by state-of-the-art AI.

---

## ✨ Features

- **Multi-Model AI Intelligence**:
  - **Google Gemini 2.5 Flash**: Lightning-fast multimodal analysis and transcript comprehension.
  - **DeepSeek-R1 (8B)** & **Qwen 2.5 (7B)**: Local Ollama-powered reasoning for offline processing.
  - **OpenAI GPT-4o / Whisper Base**: High-precision audio transcription with word-level timestamps.
- **Precision 9:16 Frame Optimization**:
  - Automatically crops and centers footage to fill modern smartphone screens without distortion or stretching.
  - Maintains full native audio and synchronized multi-clip exports.
- **Rule-Based & Semantic Extraction**:
  - Filter clips by dynamic durations (5s to 60s) or generate up to 10 non-overlapping viral moments based on custom prompts and guidelines.
- **Real-Time Streaming Progress**:
  - Live Server-Sent Events (SSE) provide step-by-step processing feedback (Audio Extraction → Transcription → AI Scene Selection → Video Rendering).
- **Interactive Web UI**:
  - Clean, dark-mode responsive dashboard with instant in-browser reel previews, custom text overlays, and one-click batch downloads.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **FFmpeg** installed and accessible via your system PATH.
- (Optional) [Ollama](https://ollama.ai) installed with `deepseek-r1:8b` or `qwen2.5:7b` for local AI generation.
- (Optional) Google Gemini or OpenAI API keys for cloud processing.

### 2. Clone the Repository
```bash
git clone https://github.com/vanshchokshi/ai-video-clipper.git
cd ai-video-clipper
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Configuration (Optional)
Create a `.env` file in the root directory if you wish to use cloud models:
```env
GEMINI_API_KEY="your-gemini-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

### 5. Launch the Application
Run the startup script or start via Python:
```bash
python run.py
```
Open your browser and navigate to:
```
http://localhost:8000
```

---

## 🛠️ Project Structure

```text
├── app.py             # FastAPI backend with SSE streaming endpoints
├── clipper_core.py    # Video processing pipeline, Whisper transcription & AI logic
├── run.py             # Application runner & server config
├── requirements.txt   # Python package dependencies
├── static/
│   └── index.html     # Single-page web studio & real-time preview dashboard
├── media/             # Local upload & exported reel storage (git-ignored)
└── .gitignore         # Excludes media, cache, and virtual environments
```

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
