"""
Binary Classifier Module for Titan Track A
Converts LLM reasoning into probability scores and binary predictions.
"""
import os
import json
import time
from openai import OpenAI

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def get_client():
    """Get OpenRouter client."""
    import httpx
    # Use custom http_client to avoid proxy issues
    http_client = httpx.Client()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        http_client=http_client,
    )


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
        time.sleep(0.3)  # Rate limit
        response = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        
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
        return {"probability": 0.5, "prediction": 1, "rationale": f"API error: {str(e)[:50]}"}


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
