# Titan - Approach Document

## Overview

Titan is a **Literary Consistency Checker** that uses a RAG (Retrieval-Augmented Generation) pipeline to verify if character backstories are consistent with evidence from novels.

---

## Architecture

```
┌─────────────────┐    ┌────────────────┐    ┌─────────────────┐    ┌────────────┐
│   Novel Texts   │ →  │   Retriever    │ →  │   Classifier    │ →  │   Output   │
│   (data/novels) │    │  (retriever.py)│    │ (classifier.py) │    │ results.csv│
└─────────────────┘    └────────────────┘    └─────────────────┘    └────────────┘
                               ↓
                       ┌────────────────┐
                       │ Causal Checker │
                       │(causal_checker)│
                       └────────────────┘
```

---

## 1. Retriever Strategy (`retriever.py`)

### Character-Centric Retrieval
- Extracts the main character name from the backstory
- Prioritizes chunks that mention the character by name
- Uses regex-based name extraction for reliability

### Temporal/Section-Based Chunking
- Chunks novels into 1000-character segments with position tracking
- Tags chunks as `early` (0-33%), `middle` (33-66%), or `late` (66-100%)
- Ensures retrieval covers the full narrative arc

### Multi-Pass Retrieval
- Retrieves proportional samples from each section (early/middle/late)
- Guarantees context diversity across the novel
- Uses sentence-transformer embeddings (`all-MiniLM-L6-v2`)

---

## 2. Classification Model (`classifier.py`)

### Model
- **Primary**: Local inference via **Ollama**
- **Models Verified**: `llama3.2` (high performance) or `gemma2:2b` (efficient memory usage)
- **Fallback**: Claude 3.5 Sonnet via OpenRouter API

### Scoring Logic
1. **EXTRACT**: Identify specific claims (names, dates, events, locations)
2. **MATCH**: Find relevant evidence for each claim
3. **DETECT CONTRADICTIONS**: Check for direct factual conflicts
4. **CAUSAL CHECK**: Verify logical/temporal consistency
5. **SCORE**: Assign confidence (0.0-1.0) and binary prediction


### Output
- `prediction`: 0 (inconsistent) or 1 (consistent)
- `rationale`: Explanation of the decision
- Silence in evidence does NOT mean contradiction

---

## 3. Causal Consistency Logic (`causal_checker.py`)

### Contradiction Detection
- **Temporal Consistency**: Detects impossible year gaps (>50 years)
- **Name Consistency**: Identifies near-match name variations (e.g., "Brittany" vs "Britannia")
- **Claim Extraction**: Parses dates, ages, names, and locations using regex

### Confidence Adjustment
| Contradictions Found | Modifier | Result |
|---------------------|----------|--------|
| 3+ | -0.30 | Inconsistent |
| 1-2 | -0.15 | Inconsistent |
| 0 | +0.10 | Consistent |

---

## 4. Pathway Integration

Pathway is used for real-time vector indexing and retrieval in `main.py`:
- `pw.io.fs.read` for ingesting novel files
- `KNNIndex` for building the vector search index
- `SentenceTransformerEmbedder` for embedding generation
- Streaming pipeline with `pw.run()`

---

## 5. Execution

### Prerequisites
1. Install [Ollama](https://ollama.com/)
2. Pull a model: `ollama pull llama3.2` or `ollama pull gemma2:2b`
3. Start server: `ollama serve`

### Running the Pipeline
```bash
# Option 1: Run with local Ollama (Recommended)
$env:USE_OLLAMA="true"
$env:OLLAMA_MODEL="llama3.2"  # or "gemma2:2b"
python run.py --input data/ --output results.csv

# Option 2: Run with OpenRouter (Cloud)
$env:USE_OLLAMA="false"
$env:OPENROUTER_API_KEY="your-key"
python run.py --input data/ --output results.csv
```

---

## Dependencies
- `pathway` - Real-time data processing
- `openai` - OpenRouter API client
- `sentence-transformers` - Text embeddings
- `pandas` - Data manipulation
- `python-dotenv` - Environment variables

---

## Team
**Team Name**: Titan
