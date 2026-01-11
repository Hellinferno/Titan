#!/usr/bin/env python3
"""
Titan Track A - Reproducible Runner
Literary Consistency Checker using Pathway

Usage:
    python run.py --input data/ --output results.csv

Note: This local runner uses numpy/sentence-transformers directly.
      The full Pathway pipeline runs on Modal cloud (main.py).
"""
import argparse
import os
import sys
import csv

# Set default API key if not in environment
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = "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"

import pandas as pd
import numpy as np

from classifier import score_backstory, get_client
from causal_checker import analyze_causal_consistency
from retriever import extract_character_name, filter_chunks_by_character, aggregate_evidence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Titan Track A - Literary Consistency Checker"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/",
        help="Input data directory containing novels/ folder and test.csv"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results.csv",
        help="Output CSV path (default: results.csv)"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5,
        help="Number of evidence chunks to retrieve per query (default: 5)"
    )
    return parser.parse_args()


def run_pipeline(input_dir: str, output_path: str, k: int = 5):
    """
    Main pipeline execution.
    
    Args:
        input_dir: Path to data directory with novels/ and test.csv
        output_dir: Path for output results.csv
        k: Number of chunks to retrieve
    """
    print("🚀 Starting Titan Track A Pipeline...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_path}")
    
    # Normalize paths
    input_dir = os.path.abspath(input_dir)
    novels_dir = os.path.join(input_dir, "novels")
    test_file = os.path.join(input_dir, "test.csv")
    
    if not os.path.exists(novels_dir):
        print(f"❌ Error: novels directory not found at {novels_dir}")
        sys.exit(1)
    
    if not os.path.exists(test_file):
        print(f"❌ Error: test.csv not found at {test_file}")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = get_client()
    
    # --- Step 1: Load and chunk novels ---
    print("\n📚 Step 1: Loading and chunking novels...")
    
    all_chunks = []
    for filename in os.listdir(novels_dir):
        filepath = os.path.join(novels_dir, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                # Chunk into 1000-char segments
                for i in range(0, len(text), 1000):
                    chunk_text = text[i:i+1000]
                    position = i / len(text) if len(text) > 0 else 0
                    section = "early" if position < 0.33 else ("middle" if position < 0.66 else "late")
                    all_chunks.append({
                        "text": chunk_text,
                        "source": filename,
                        "position": position,
                        "section": section
                    })
            except Exception as e:
                print(f"   Warning: Could not read {filename}: {e}")
    
    print(f"   Loaded {len(all_chunks)} chunks from novels")
    
    # --- Step 2: Build embeddings ---
    print("\n🔢 Step 2: Building embeddings...")
    
    from sentence_transformers import SentenceTransformer
    embedder_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Embed all chunks
    chunk_texts = [c["text"] for c in all_chunks]
    chunk_embeddings = embedder_model.encode(chunk_texts, show_progress_bar=True)
    
    for i, emb in enumerate(chunk_embeddings):
        all_chunks[i]["embedding"] = emb
    
    print(f"   Built {len(chunk_embeddings)} embeddings")
    
    # --- Step 3: Load test data ---
    print("\n📋 Step 3: Loading test data...")
    
    df_test = pd.read_csv(test_file)
    
    # Handle different column naming
    if 'id' in df_test.columns:
        df_test = df_test.rename(columns={'id': 'story_id'})
    if 'content' in df_test.columns:
        df_test = df_test.rename(columns={'content': 'backstory'})
    
    print(f"   Loaded {len(df_test)} test cases")
    
    # --- Step 4: Process each backstory ---
    print("\n🔍 Step 4: Processing backstories...")
    
    results = []
    for idx, row in df_test.iterrows():
        story_id = row.get('story_id', idx)
        backstory = row.get('backstory', '')
        char_name = row.get('char', extract_character_name(backstory))
        
        print(f"   Processing {idx+1}/{len(df_test)}: Story {story_id}...", end=" ")
        
        # Embed the backstory
        query_embedding = embedder_model.encode(backstory)
        
        # Find nearest chunks
        import numpy as np
        similarities = []
        for chunk in all_chunks:
            sim = np.dot(query_embedding, chunk["embedding"])
            similarities.append((sim, chunk))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_chunks = [c for _, c in similarities[:k*2]]  # Get extra for filtering
        
        # Character-centric filtering
        if char_name:
            top_chunks = filter_chunks_by_character(
                [c["text"] for c in top_chunks], 
                char_name
            )
            top_chunks = top_chunks[:k]
        else:
            top_chunks = [c["text"] for c in top_chunks[:k]]
        
        # Aggregate evidence
        evidence = aggregate_evidence(top_chunks)
        
        # Run causal consistency check
        causal_result = analyze_causal_consistency(backstory, evidence)
        
        # Get AI classification
        result = score_backstory(backstory, evidence, client)
        
        # Adjust based on causal check
        if causal_result["contradictions"]:
            result["rationale"] = (
                f"Causal issues: {'; '.join(causal_result['contradictions'][:2])}. "
                + result["rationale"]
            )
        
        prediction = result["prediction"]
        rationale = result["rationale"]
        
        results.append({
            "StoryID": story_id,
            "Prediction": prediction,
            "Rationale": rationale
        })
        
        print(f"{'✓' if prediction == 1 else '✗'}")
    
    # --- Step 5: Write output ---
    print(f"\n💾 Step 5: Writing results to {output_path}...")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["StoryID", "Prediction", "Rationale"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\n✅ Success! Results written to {output_path}")
    print(f"   Total: {len(results)} predictions")
    print(f"   Consistent: {sum(1 for r in results if r['Prediction'] == 1)}")
    print(f"   Inconsistent: {sum(1 for r in results if r['Prediction'] == 0)}")
    
    return results


def main():
    args = parse_args()
    run_pipeline(args.input, args.output, args.k)


if __name__ == "__main__":
    main()
