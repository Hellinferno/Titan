# 🚀 Titan - Literary Consistency Checker
# Task A Submission

A strict, robust AI pipeline ensuring character consistency using **local LLMs (Ollama)** or **Claude 3.5**.
Compliant with Track A requirements: Binary Classification, Causal Reasoning, and Pathway integration.

---

## 📄 Submission Contents
- **`run.py`**: Canonical local runner. **Automatically enforces strict CSV formatting** (removes newlines, ensures binary output).
- **`REPORT.md`**: Technical report covering methodology, causal logic, and error analysis.
- **`classifier.py`**: Hardened scoring logic with retry mechanisms (N=3) for deterministic results.
- **`tests.py`**: Unit tests for schema and logic verification.
- **`evaluate.py`**: Metrics calculation script.

---

## 🛠️ Quick Start (Local)

**Note**: This local runner uses `numpy` and `sentence-transformers` for ease of judging. For the full Pathway-enabled pipeline (Cloud), see `main.py` in the root repository.

### Prerequisites
- Python 3.9+

- [Ollama](https://ollama.com/) (Required for local mode)
- `pip install -r requirements.txt`

### 1. Setup
```bash
# Pull the model (approx 2GB)
ollama pull gemma2:2b
# OR for better performance:
ollama pull llama3.2
```

### 2. Run the Pipeline
```bash
# Set environment to use Ollama
$env:USE_OLLAMA="true"
$env:OLLAMA_MODEL="gemma2:2b" 

# Execute
python run.py --input data/ --output results.csv
```

### 3. Validation
```bash
# Run unit tests
python tests.py

# (Optional) Evaluate against ground truth
python evaluate.py --pred results.csv --gold data/train.csv
```

---

## 🧠 Approach Highlights

1. **Deterministic Logic**: `classifier.py` enforces binary output (0/1) via strict parsing and retry loops.
2. **Causal Checker**: `causal_checker.py` adds a layer of logic (Time/Name checks) beyond simple R/A.
3. **Pathway**: Used in `main.py` for scalable vector indexing (cloud implementation).
4. **Validation**: Output is programmatically validated for schema compliance (`StoryID,Prediction,Rationale`).

(See `REPORT.md` for full details)

---

## ⚙️ Configuration
- **OLLAMA_MODEL**: defaults to `llama3.2`
- **OPENROUTER_API_KEY**: fallback if `USE_OLLAMA=false`

---

**Team Titan**
