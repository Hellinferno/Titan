# 🚀 Titan - Literary Consistency Checker

A powerful AI-powered pipeline that verifies backstory consistency against novel evidence using **Claude 3.5 Sonnet** and **Pathway** for real-time data processing.

## 📋 Overview

This project is a hackathon submission for the **Track A - Pathway Challenge**. It uses a RAG (Retrieval-Augmented Generation) approach to determine if character backstories are consistent with evidence from novels.

### How It Works

1. **Ingestion & Chunking**: Reads novel text files and chunks them into 1000-character segments
2. **Vector Indexing**: Creates embeddings using `all-MiniLM-L6-v2` sentence transformer
3. **Semantic Search**: Retrieves the most relevant novel chunks for each backstory query
4. **AI Judgment**: Uses Claude 3.5 Sonnet via OpenRouter to determine consistency
5. **Output**: Generates a CSV with predictions (0 = contradiction, 1 = consistent) and rationales

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Pathway** | Real-time data processing & vector indexing |
| **Modal** | Serverless cloud compute with GPU support |
| **Claude 3.5 Sonnet** | AI reasoning via OpenRouter API |
| **Sentence Transformers** | Text embeddings (`all-MiniLM-L6-v2`) |
| **Pandas** | Data manipulation |

## 📁 Project Structure

```
hackathon_project/
├── main.py                      # Main pipeline script
├── fix_results.py               # Post-processing utility
├── submission_final.csv         # Final submission output
├── submission_final_fixed.csv   # Cleaned submission
└── data/
    ├── novels/                  # Novel text files
    ├── test.csv                 # Test data with backstories
    └── train.csv                # Training data
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Modal](https://modal.com/) account
- OpenRouter API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hellinferno/Titan-.git
   cd Titan-
   ```

2. **Install Modal CLI**
   ```bash
   pip install modal
   modal setup
   ```

3. **Set your API key** (environment variable):
   
   **PowerShell:**
   ```powershell
   $env:OPENROUTER_API_KEY = "your-api-key"
   ```
   
   **Linux/Mac:**
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   ```

### Running the Pipeline

```bash
modal run main.py
```

This will:
- Deploy the pipeline to Modal's cloud infrastructure
- Process all novels and test backstories
- Generate `submission_final.csv` locally

## 📊 Output Format

The output CSV contains:

| Column | Description |
|--------|-------------|
| `Story ID` | Unique identifier for the backstory |
| `Prediction` | `0` (contradiction) or `1` (consistent) |
| `Rationale` | Brief explanation from Claude |

## ⚙️ Configuration

Key parameters in `main.py`:

```python
APP_NAME = "track-a-pathway-claude"  # Modal app name
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")  # Environment variable
DATA_DIR = "/root/data/"             # Data directory in cloud
```

## 🧠 AI Prompt Design

The AI judge uses a **Chain of Thought** prompt for comprehensive analysis:
1. **IDENTIFY**: Specific details in evidence relating to the backstory
2. **COMPARE**: Check for logical contradictions
3. **DECIDE**: Determine consistency

- Returns `0` if backstory **contradicts** novel evidence
- Returns `1` if backstory is **supported by** or **silent** in the evidence
- Provides comprehensive evidence rationale for each decision

## 📈 Performance

- **Model**: Claude 3.5 Sonnet (via OpenRouter)
- **Embedding**: all-MiniLM-L6-v2 (384 dimensions)
- **Chunk Size**: 1000 characters
- **K-Nearest Neighbors**: k=5 for wider context retrieval

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

MIT License - feel free to use this project for your own hackathons and learning.

---

**Built with ❤️ for the Pathway Hackathon**
