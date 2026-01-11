"""
Causal Consistency Checker for Titan Track A
Provides structured contradiction detection and causal reasoning.
"""
import re


def extract_claims(backstory: str) -> list:
    """
    Extract verifiable claims from a backstory.
    
    Returns list of dicts with claim type and content:
    - names: character names mentioned
    - dates: years, ages, time periods
    - events: actions, incidents
    - locations: places, settings
    - relationships: family, social connections
    """
    claims = []
    
    # Extract potential years (4 digits)
    years = re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', backstory)
    for year in years:
        claims.append({"type": "date", "value": year})
    
    # Extract ages
    ages = re.findall(r'\b(?:at |aged? |when he was |when she was )(\d+)\b', backstory.lower())
    for age in ages:
        claims.append({"type": "age", "value": age})
    
    # Extract quoted names or capitalized proper nouns
    names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', backstory)
    for name in set(names):
        if len(name) > 2 and name not in ['The', 'He', 'She', 'His', 'Her', 'At', 'In', 'On']:
            claims.append({"type": "name", "value": name})
    
    # Extract locations (common patterns)
    locations = re.findall(r'\b(?:in|at|from|to|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', backstory)
    for loc in set(locations):
        claims.append({"type": "location", "value": loc})
    
    return claims


def check_temporal_consistency(backstory_claims: list, evidence_claims: list) -> list:
    """
    Check for temporal inconsistencies between backstory and evidence.
    
    Returns list of contradiction descriptions.
    """
    contradictions = []
    
    backstory_dates = [c for c in backstory_claims if c["type"] in ["date", "age"]]
    evidence_dates = [c for c in evidence_claims if c["type"] in ["date", "age"]]
    
    # Simple check: if both have dates, flag potential conflicts
    if backstory_dates and evidence_dates:
        bs_years = [int(c["value"]) for c in backstory_dates if c["type"] == "date"]
        ev_years = [int(c["value"]) for c in evidence_dates if c["type"] == "date"]
        
        # Check for impossible overlaps (e.g., backstory date after evidence event)
        for bs_year in bs_years:
            for ev_year in ev_years:
                if abs(bs_year - ev_year) > 50:
                    contradictions.append(f"Temporal gap: backstory ({bs_year}) vs evidence ({ev_year})")
    
    return contradictions


def check_name_consistency(backstory: str, evidence: str) -> list:
    """
    Check for name/entity inconsistencies.
    """
    contradictions = []
    
    # Extract names from both
    bs_names = set(re.findall(r'\b([A-Z][a-z]{2,})\b', backstory))
    ev_names = set(re.findall(r'\b([A-Z][a-z]{2,})\b', evidence))
    
    # Filter common words
    stop_words = {'The', 'He', 'She', 'His', 'Her', 'At', 'In', 'On', 'By', 'To', 'For', 'With', 'This', 'That'}
    bs_names -= stop_words
    ev_names -= stop_words
    
    # Check for near-matches that might indicate errors (e.g., "Brittany" vs "Britannia")
    for bs_name in bs_names:
        for ev_name in ev_names:
            # Check for similar but different names (potential contradiction)
            if bs_name != ev_name and len(bs_name) > 4 and len(ev_name) > 4:
                # Simple similarity: same first 3 letters but different ending
                if bs_name[:3].lower() == ev_name[:3].lower() and bs_name[-2:] != ev_name[-2:]:
                    contradictions.append(f"Name mismatch: '{bs_name}' vs '{ev_name}'")
    
    return contradictions


def analyze_causal_consistency(backstory: str, evidence: str) -> dict:
    """
    Full causal consistency analysis.
    
    Returns:
        dict with:
        - is_consistent: bool
        - contradictions: list of found contradictions
        - confidence_modifier: float adjustment to base confidence (-0.3 to +0.2)
    """
    all_contradictions = []
    
    # Extract claims
    bs_claims = extract_claims(backstory)
    ev_claims = extract_claims(evidence)
    
    # Check temporal consistency
    temporal_issues = check_temporal_consistency(bs_claims, ev_claims)
    all_contradictions.extend(temporal_issues)
    
    # Check name consistency
    name_issues = check_name_consistency(backstory, evidence)
    all_contradictions.extend(name_issues)
    
    # Calculate confidence modifier
    if len(all_contradictions) >= 3:
        confidence_modifier = -0.3
        is_consistent = False
    elif len(all_contradictions) >= 1:
        confidence_modifier = -0.15
        is_consistent = False
    else:
        confidence_modifier = 0.1
        is_consistent = True
    
    return {
        "is_consistent": is_consistent,
        "contradictions": all_contradictions,
        "confidence_modifier": confidence_modifier
    }
