# Titan - Technical Report (Task A)

## 1. Approach Summary
We treated Task A as a Long-Context Reasoning problem. We implemented a pipeline that first retrieves relevant narrative segments associated with the hypothesis using semantic similarity. These segments are then fed into a Large Language Model (via `classifier.py`) which is prompted to evaluate logical consistency and causal flow. Finally, a `causal_checker` module validates the outputs to ensure chronological integrity before generating the final binary predictions.

## 2. Pipeline Diagram
```
┌─────────────────┐    ┌────────────────┐    ┌─────────────────┐    ┌────────────┐
│   Novel Texts   │ →  │   Retriever    │ →  │   Classifier    │ →  │   Output   │
│   (data/novels) │    │(Semantic + KW) │    │ (LLM Evaluator) │    │ results.csv│
└─────────────────┘    └────────────────┘    └─────────────────┘    └────────────┘
                                ↓
                        ┌────────────────┐
                        │ Causal Checker │
                        │ (Logic/Regex)  │
                        └────────────────┘
```

## 3. Pathway Integration
Pathway is used to orchestrate the vector indexing for scalable retrieval (server-side):
- **Ingestion**: `pw.io.fs.read` monitors the library of novels.
- **Indexing**: `KNNIndex` builds a real-time vector index using `SentenceTransformerEmbedder`.
- **Querying**: The `main.py` pipeline queries this index to retrieve context for each new backstory hypothesis.

## 4. Causal Reasoning Justification
We go beyond simple similarity matching by:
- **Chain-of-Thought Prompting**: The LLM is explicitly instructed to "Extract Claims" -> "Match Evidence" -> "Detect Contradictions".
- **Deterministic Causal Checks**: `causal_checker.py` performs rigorous logic checks on names (Levenstein distance) and timelines (date parsing).
- **Rule-Based Overrides**: If the LLM finds consistency but the Causal Checker finds a temporal impossibility (>50 year gap), the system overrides the decision to "Inconsistent".

## 5. Error Analysis
- **Ambiguity**: Silence in the text is not a contradiction. Our prompt explicitly handles this ("Silence != Contradiction").
- **Name Variations**: "Brittany" vs "Britannia" can confuse basic retrieval; our regex-based causal checker catches these subtle inconsistencies.
- **Memory**: Local LLMs (gemma2/llama3) can struggle with long context; our chunking strategy (early/middle/late) mitigates this by providing diverse context windows.

## 6. Ethical Considerations
- **Bias**: We rely on open-weights models (Llama 3, Gemma 2) or Claude to minimize vendor-specific bias.
- **Reproducibility**: The pipeline uses pinned checks and deterministic seeding where possible.

---

## Detailed Architecture
(See APPROACH.md for full component breakdown)

## Execution
See `run.py` or `APPROACH.md` for running instructions.
