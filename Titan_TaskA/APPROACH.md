# Titan - Approach Document

## 1. The Core Objective
Task A is a Binary Classification problem.
- **Input**: A pair of events (Context/Hypothesis) linked to a long novel.
- **Goal**: Determine if the proposed "Past" causally and logically explains the observed "Future."
- **Output**: A label (1 for Valid/Consistent, 0 for Invalid/Inconsistent) in `results.csv`.

---

## 2. The Technical Pipeline
We treated Task A as a Long-Context Reasoning problem using a RAG (Retrieval-Augmented Generation) pipeline.

### Step A: Data Ingestion (`main.py` & `data/`)
The system loads the challenge questions from `data/test.csv`. It loads the full text of the novels (e.g., *The Count of Monte Cristo*) from `data/novels/`. Since novels are too large to feed into an LLM entirely, they are processed for retrieval.

### Step B: Context Retrieval (`retriever.py`)
For each question, this module searches the novel for relevant text chunks using:
- **Character-Centric Filtering**: Prioritizes chunks mentioning the main character.
- **Semantic Search**: Uses `all-MiniLM-L6-v2` embeddings to find conceptually similar segments.
- **Multi-Pass Retrieval**: Ensures coverage across early, middle, and late sections of the book.

**Output**: A concentrated set of evidence text snippets.

### Step C: Causal Reasoning (`classifier.py`)
This is the brain of the system. It constructs a dynamic prompt for the LLM (Large Language Model).
- **Prompt Structure**: "Given this context from the book [Evidence], does [Event A] causally lead to [Event B]?"
- **Model**: Local inference via **Ollama** (`gemma2:2b` or `llama3.2`) or **Claude 3.5 Sonnet** (via OpenRouter).
- **Chain of Thought**: The model extracts claims, matches them to evidence, and detects contradictions.

### Step D: Logical Verification (`causal_checker.py`)
A post-processing validation step that ensures consistency:
- **Temporal Check**: Verifies that dates/ages in the backstory don't chronologically contradict the evidence.
- **Name Check**: Flags potential entity mismatches (e.g., "Brittany" vs "Britannia").
- **Consistency Verification**: Adjusts confidence scores based on detected logical flaws.

### Step E: Submission Generation (`run.py` -> `results.csv`)
The `run.py` script orchestrates the loop over all test cases, collects final decisions, and generates the `results.csv` submission file.

---

## 3. Summary for Report
"We treated Task A as a Long-Context Reasoning problem. We implemented a pipeline that first retrieves relevant narrative segments associated with the hypothesis using semantic similarity. These segments are then fed into a Large Language Model (via `classifier.py`) which is prompted to evaluate logical consistency and causal flow. Finally, a `causal_checker` module validates the outputs to ensure chronological integrity before generating the final binary predictions."

---

## 4. Execution

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
```

---

## Team
**Team Name**: Titan
