"""
Binary Classifier Module for Titan Track A
Converts LLM reasoning into probability scores and binary predictions.
Uses Ollama for local LLM inference.
"""
import os
import json
import time
from openai import OpenAI

def _is_ollama_enabled():
    return os.environ.get("USE_OLLAMA", "true").lower() == "true"

def get_client():
    """Get LLM client - Ollama (local) or OpenRouter (cloud)."""
    import httpx
    http_client = httpx.Client(timeout=120.0)  # Longer timeout for local inference
    
    if _is_ollama_enabled():
        # Use local Ollama server
        return OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Ollama doesn't need a real key
            http_client=http_client,
        )
    else:
        # Use OpenRouter (cloud)
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            http_client=http_client,
        )


def get_model_name():
    """Get the model name based on configuration."""
    if _is_ollama_enabled():
        return os.environ.get("OLLAMA_MODEL", "llama3.2")
    else:
        return "anthropic/claude-3.5-sonnet"



def score_backstory(backstory: str, evidence: str, client=None) -> dict:
    """
    Score a backstory against evidence for consistency.
    
    Args:
        backstory: The character backstory to verify
        evidence: Retrieved evidence chunks from the novel
        client: Optional OpenAI client (creates one if not provided)
    
    Returns:
        dict with keys:
        - probability: float 0.0-1.0 (confidence in consistency)
        - prediction: int 0 or 1 (0=inconsistent, 1=consistent)
        - rationale: str explanation
    """
    if client is None:
        client = get_client()
    
    model = get_model_name()
    
    prompt = f"""You are a literary consistency classifier. Analyze if a character backstory is consistent with novel evidence.

## Task
Determine if the Backstory is CONSISTENT (1) or INCONSISTENT (0) with the Evidence.

## Analysis Steps (Chain of Thought)
1. **EXTRACT**: List specific claims in the backstory (names, dates, events, relationships, locations)
2. **MATCH**: For each claim, find relevant evidence (if any)
3. **DETECT CONTRADICTIONS**: Check for direct factual conflicts
4. **CAUSAL CHECK**: Verify logical/temporal consistency
5. **SCORE**: Assign confidence 0.0-1.0

## Rules
- Return 0 (INCONSISTENT) if backstory DIRECTLY CONTRADICTS evidence
- Return 1 (CONSISTENT) if backstory is SUPPORTED BY or NOT CONTRADICTED BY evidence
- Silence in evidence does NOT mean contradiction

## Output Format
Return ONLY valid JSON:
{{"confidence": 0.0-1.0, "prediction": 0 or 1, "contradictions": ["list of contradictions found"], "rationale": "Brief explanation"}}

---

**Backstory:**
{backstory}

**Evidence from Novel:**
{evidence}
"""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            
            text = response.choices[0].message.content.strip()
            # Clean potential markdown
            clean_text = text.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON substring
            json_start = clean_text.find("{")
            json_end = clean_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                clean_text = clean_text[json_start:json_end]
            
            data = json.loads(clean_text)
            
            # Validate essential fields
            confidence = float(data.get("confidence", 0.5))
            prediction = int(data.get("prediction", 1 if confidence >= 0.5 else 0))
            if prediction not in [0, 1]:
                 raise ValueError("Prediction must be 0 or 1")
                 
            contradictions = data.get("contradictions", [])
            rationale = data.get("rationale", "No rationale provided.")
            
            if contradictions:
                rationale = f"Contradictions: {'; '.join(contradictions)}. {rationale}"
            
            return {
                "probability": confidence,
                "prediction": prediction,
                "rationale": rationale
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                time.sleep(1) # Wait before retry
                continue
            
            # Fallback only on final failure
            raw_text = locals().get('text', '')
            if "prediction" in raw_text.lower() and ": 0" in raw_text:
                 return {"probability": 0.2, "prediction": 0, "rationale": "Fallback: detected 0 after retries"}
            
            return {"probability": 0.5, "prediction": 1, "rationale": f"Model Error (Retried {max_retries}x): {str(e)[:50]}"}
        except Exception as e:
            # Non-parsing errors (e.g. API connection) - standard handling
            return {"probability": 0.5, "prediction": 1, "rationale": f"System Error: {str(e)[:50]}"}




def batch_score(backstories: list, evidences: list, client=None) -> list:
    """
    Score multiple backstory-evidence pairs.
    
    Args:
        backstories: List of backstory strings
        evidences: List of evidence strings (same length as backstories)
        client: Optional OpenAI client
    
    Returns:
        List of result dicts from score_backstory
    """
    if client is None:
        client = get_client()
    
    results = []
    for backstory, evidence in zip(backstories, evidences):
        result = score_backstory(backstory, evidence, client)
        results.append(result)
    
    return results
