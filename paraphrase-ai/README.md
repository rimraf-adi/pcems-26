# ❇️ Paraphrase AI — Granular Semantic Similarity Studio

A local-first, Apple Silicon (M1/M2/M3) optimized Streamlit application for multi-granularity semantic similarity analysis between original and paraphrased texts.

Powered by [`uv`](https://github.com/astral-sh/uv), [`sentence-transformers`](https://www.sbert.net/), and PyTorch with native Apple Silicon Metal (MPS) acceleration.

---

## 🌟 Key Features

- **Multi-Granularity Analysis**:
  - 📊 **Document Level**: Global cosine similarity & length ratio metrics.
  - 📑 **Paragraph Level**: Automatic paragraph segmentation, pairwise mapping, and inter-paragraph heatmap.
  - 🔍 **Sentence Level**: Full sentence alignment matrix ($M \times N$), best-match pairing, and risk level tagging.
- **Color-Coded Risk Inspector**:
  - 🔴 **High Risk (>0.85 Similarity)**: Sentence is too close to original (rephrase recommended).
  - 🟡 **Moderate (0.65 – 0.85 Similarity)**: Good semantic balance.
  - 🟢 **Substantial Rewrite (<0.65 Similarity)**: Distinct wording while retaining meaning.
- **Apple Silicon Hardware Acceleration**:
  - Automatically detects and utilizes PyTorch `mps` (Metal Performance Shaders) on M1 Macs.
- **Data Export & Presets**:
  - Pre-loaded sample presets (Academic, Technical, Narrative).
  - One-click CSV export of sentence alignments and similarity scores.

---

## 🚀 Quick Start (Powered by `uv`)

### Prerequisites
- Python $\ge$ 3.10
- [`uv`](https://github.com/astral-sh/uv) package manager installed

### Running the App

```bash
cd paraphrase-ai
chmod +x run.sh
./run.sh
```

Or execute directly with `uv`:

```bash
uv sync
uv run streamlit run app.py
```

The Streamlit dashboard will automatically open at `http://localhost:8501`.

---

## ⚙️ Supported Embedding Models

| Model Name | Description | Size | Best Use Case |
| :--- | :--- | :---: | :--- |
| `all-MiniLM-L6-v2` | Ultra-fast & lightweight (Default) | ~80 MB | Real-time interactive analysis |
| `all-mpnet-base-v2` | High precision transformer | ~400 MB | Rigorous semantic comparison |
| `paraphrase-MiniLM-L6-v2` | Paraphrase tuned model | ~80 MB | Paraphrase evaluation |

---

## 📂 Project Structure

```
paraphrase-ai/
├── analyzer.py        # Core NLP segmentation & similarity engine
├── app.py             # Streamlit visual dashboard
├── pyproject.toml     # uv project configuration & dependencies
├── run.sh             # Launch script
└── README.md          # Documentation
```
