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
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,  # More tokens for local model
        )
        
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Try to extract JSON from the response
        # Sometimes models add extra text before/after JSON
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            text = text[json_start:json_end]
        
        data = json.loads(text)
        confidence = float(data.get("confidence", 0.5))
        prediction = int(data.get("prediction", 1 if confidence >= 0.5 else 0))
        contradictions = data.get("contradictions", [])
        rationale = data.get("rationale", "No rationale provided.")
        
        # Add contradiction info to rationale if found
        if contradictions:
            rationale = f"Contradictions: {'; '.join(contradictions)}. {rationale}"
        
        return {
            "probability": confidence,
            "prediction": prediction,
            "rationale": rationale
        }
        
    except json.JSONDecodeError as e:
        # Try to extract prediction from raw text
        text_lower = text.lower() if 'text' in dir() else ""
        if "inconsistent" in text_lower or '"prediction": 0' in text_lower:
            return {"probability": 0.3, "prediction": 0, "rationale": f"Parse error but detected inconsistency: {str(e)[:50]}"}
        return {"probability": 0.5, "prediction": 1, "rationale": f"JSON parse error: {str(e)[:50]}"}
        
    except Exception as e:
        # FALLBACK: Deterministic failure mode
        # If we can't parse, check for strong "inconsistent" keywords again in raw text
        # If still unsure, default to 0 (Conservative approach for consistency checking)
        
        # Try to recover text if available
        raw_text = locals().get('text', '') or str(e)
        
        if "prediction" in raw_text.lower() and ": 0" in raw_text:
             return {"probability": 0.2, "prediction": 0, "rationale": "Fallback parse: detected 0"}
             
        # Default to consistent (1) if silent, or 0 if we want to be strict?
        # Task A often prefers 1 as default if no contradiction found.
        return {"probability": 0.5, "prediction": 1, "rationale": f"Model Error: {str(e)[:50]}"}



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
