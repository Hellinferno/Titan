import modal
import os

# --- CONFIGURATION ---
APP_NAME = "track-a-pathway-claude"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")

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

    # Define the Judge Function with Chain of Thought reasoning
    @pw.udf
    def ai_judge(backstory: str, evidence_text: str) -> str:
        prompt = f"""You are a literary consistency checker analyzing character backstories against novel evidence.

Task: Determine if the 'Backstory' is consistent with the 'Evidence' from the novel.

Analyze the retrieved evidence carefully using these steps:
1. IDENTIFY: First, identify any specific details in the evidence that relate to the backstory (names, events, relationships, timelines, locations).
2. COMPARE: Then, check for logical contradictions between the backstory claims and the evidence details.
3. DECIDE: Finally, determine if the backstory is consistent or contradictory.

Rules:
- Return 0 if the Backstory DIRECTLY CONTRADICTS specific facts in the Evidence.
- Return 1 if the Backstory is SUPPORTED BY or FITS WITHIN the Evidence (including cases where evidence is silent on the backstory claims).

Return ONLY JSON: {{"prediction": 0 or 1, "rationale": "Comprehensive evidence rationale explaining your step-by-step analysis."}}

Backstory: {backstory}

Evidence from Novel:
{evidence_text}"""
        
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

    # --- Step D: Retrieve & Decide (k=5 for wider context) ---
    knn_results = index.get_nearest_items(questions.query_vector, k=5)
    
    # Join questions with KNN results - this may result in multiple rows per query
    matches = questions.join(knn_results, pw.this.id == knn_results.id).select(
        story_id=pw.this.story_id,
        backstory=pw.this.backstory,
        evidence=knn_results.text
    )

    # Aggregate multiple evidence chunks per story_id by concatenating them
    @pw.udf
    def concat_with_separator(texts: list) -> str:
        return "\n\n---EVIDENCE CHUNK---\n\n".join(texts)

    aggregated_matches = matches.groupby(pw.this.story_id, pw.this.backstory).reduce(
        story_id=pw.reducers.any(pw.this.story_id),
        backstory=pw.reducers.any(pw.this.backstory),
        evidence_list=pw.reducers.tuple(pw.this.evidence)
    )

    # Convert tuple to concatenated string
    aggregated_matches = aggregated_matches.select(
        story_id=pw.this.story_id,
        backstory=pw.this.backstory,
        evidence=pw.apply(lambda x: "\n\n---EVIDENCE CHUNK---\n\n".join(x), pw.this.evidence_list)
    )

    results = aggregated_matches.select(
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
