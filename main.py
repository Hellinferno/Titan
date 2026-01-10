import modal
import os

# --- CONFIGURATION ---
APP_NAME = "track-a-pathway-claude"
OPENROUTER_API_KEY = "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"

# Define the Cloud Environment
image = (
    modal.Image.debian_slim()
    .apt_install("poppler-utils")
    .pip_install(
        "pathway",
        "openai",  # OpenRouter uses OpenAI-compatible API
        "sentence-transformers",
        "pandas",
        "pdf2image",
        "unstructured",
        "docling",
    )
    .add_local_dir("./data", remote_path="/root/data")
)

app = modal.App(APP_NAME)

@app.function(
    image=image,
    timeout=1200,
    gpu="any"
)
def run_pipeline():
    import pathway as pw
    import pandas as pd
    from openai import OpenAI
    import json
    import time
    import csv
    from pathway.stdlib.ml.index import KNNIndex
    from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

    # --- CONFIGURATION ---
    DATA_DIR = "/root/data/"

    print("🚀 Starting Track A Pipeline with Claude 3.5 Sonnet...")
    
    # 1. SETUP OPENROUTER CLIENT (Claude 3.5 Sonnet)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # Define the Judge Function
    @pw.udf
    def ai_judge(backstory: str, evidence_text: str) -> str:
        prompt = f"""You are a literary consistency checker.
Task: Determine if the 'Backstory' is consistent with the 'Evidence' from the novel.

Rules:
- If the Backstory contradicts the Evidence, return 0.
- If the Backstory is supported by OR fits within the Evidence (silent), return 1.

Return ONLY JSON: {{"prediction": 0 or 1, "rationale": "One short sentence explaining why."}}

Backstory: {backstory}
Evidence: {evidence_text}"""
        
        try:
            time.sleep(0.5)  # Rate limit pause
            response = client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            clean_text = response.choices[0].message.content.strip()
            clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            pred = data.get("prediction", 1)
            rat = data.get("rationale", "No rationale provided.")
            return f"{pred}|||{rat}"
        except Exception as e:
            return f"1|||Error: {str(e)[:50]}"

    # --- Step A: Ingest & Chunk Novels ---
    novels = pw.io.fs.read(
        f"{DATA_DIR}novels/",
        format="binary",
        mode="static",
        with_metadata=True
    )

    def chunk_text(data):
        text = data.decode("utf-8", errors="ignore")
        return [text[i:i+1000] for i in range(0, len(text), 1000)]

    chunks = novels.select(
        chunk_text=pw.apply(chunk_text, pw.this.data)
    ).flatten(pw.this.chunk_text)

    # --- Step B: Build Vector Index ---
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    
    enriched_chunks = chunks.select(
        text=pw.this.chunk_text,
        vector=embedder(pw.this.chunk_text)
    )

    index = KNNIndex(enriched_chunks.vector, enriched_chunks, n_dimensions=384)

    # --- Step C: Process the Test File ---
    df_test = pd.read_csv(f"{DATA_DIR}test.csv")
    if 'content' in df_test.columns:
        df_test = df_test.rename(columns={'id': 'story_id', 'content': 'backstory'})
    df_test.to_csv(f"{DATA_DIR}temp_input.csv", index=False)

    questions = pw.io.csv.read(
        f"{DATA_DIR}temp_input.csv",
        mode="static",
        schema=pw.schema_from_csv(f"{DATA_DIR}temp_input.csv")
    )

    questions = questions.select(
        pw.this.story_id,
        pw.this.backstory,
        query_vector=embedder(pw.this.backstory)
    )

    # --- Step D: Retrieve & Decide ---
    knn_results = index.get_nearest_items(questions.query_vector, k=1)
    
    matches = questions.join(knn_results, pw.this.id == knn_results.id).select(
        story_id=pw.this.story_id,
        backstory=pw.this.backstory,
        evidence=knn_results.text
    )

    results = matches.select(
        story_id=pw.this.story_id,
        combined_result=ai_judge(pw.this.backstory, pw.this.evidence)
    )

    # --- Step E: Output ---
    pw.io.csv.write(results, f"{DATA_DIR}results.csv")
    pw.run()
    
    print("🔧 Formatting output for submission...")
    
    final_df = pd.read_csv(f"{DATA_DIR}results.csv")
    
    if 'combined_result' in final_df.columns:
        final_df[['prediction', 'rationale']] = final_df['combined_result'].str.split('|||', n=1, expand=True)
        final_df['prediction'] = final_df['prediction'].fillna('1')
        final_df['rationale'] = final_df['rationale'].fillna('')
    
    final_df = final_df.rename(columns={'story_id': 'Story ID', 'prediction': 'Prediction', 'rationale': 'Rationale'})
    final_df = final_df[['Story ID', 'Prediction', 'Rationale']]
    
    output_path = f"{DATA_DIR}submission_final.csv"
    final_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    
    print("✅ Success! File 'submission_final.csv' is ready.")
    
    with open(output_path, "r") as f:
        return f.read()

@app.local_entrypoint()
def main():
    csv_content = run_pipeline.remote()
    with open("submission_final.csv", "w") as f:
        f.write(csv_content)
    print("✅ Downloaded submission_final.csv")
