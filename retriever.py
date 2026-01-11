"""
Enhanced Retriever for Titan Track A
Provides character-centric and multi-pass retrieval for long-context handling.
"""
import re


def extract_character_name(backstory: str) -> str:
    """
    Extract the likely main character name from a backstory.
    Looks for patterns like "he", "she", proper nouns at start, etc.
    """
    # Try to find a name early in the text
    # Pattern: Capitalized word that's likely a name
    words = backstory.split()
    for word in words[:10]:
        clean = re.sub(r'[^A-Za-z]', '', word)
        if clean and clean[0].isupper() and len(clean) > 2:
            if clean not in ['The', 'He', 'She', 'His', 'Her', 'At', 'In', 'On', 'By']:
                return clean
    return ""


def filter_chunks_by_character(chunks: list, character_name: str) -> list:
    """
    Filter chunk list to prioritize those mentioning the character.
    
    Args:
        chunks: List of text chunks (strings)
        character_name: Name to search for
    
    Returns:
        Reordered list with character-relevant chunks first
    """
    if not character_name:
        return chunks
    
    relevant = []
    others = []
    
    name_lower = character_name.lower()
    for chunk in chunks:
        if name_lower in chunk.lower():
            relevant.append(chunk)
        else:
            others.append(chunk)
    
    return relevant + others


def chunk_by_sections(text: str, chunk_size: int = 1000, overlap: int = 100) -> list:
    """
    Smart chunking that respects paragraph boundaries.
    
    Args:
        text: Full text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
    
    Returns:
        List of chunk dicts with text and position info
    """
    # Split by paragraph
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    current_position = 0  # Track position as percentage through text
    total_len = len(text)
    char_position = 0
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                position_pct = char_position / total_len if total_len > 0 else 0
                chunks.append({
                    "text": current_chunk.strip(),
                    "position": position_pct,
                    "section": "early" if position_pct < 0.33 else ("middle" if position_pct < 0.66 else "late")
                })
            char_position += len(current_chunk)
            # Start new chunk with overlap from previous
            if overlap > 0 and current_chunk:
                current_chunk = current_chunk[-overlap:] + para + "\n\n"
            else:
                current_chunk = para + "\n\n"
    
    # Don't forget the last chunk
    if current_chunk.strip():
        position_pct = char_position / total_len if total_len > 0 else 1.0
        chunks.append({
            "text": current_chunk.strip(),
            "position": position_pct,
            "section": "early" if position_pct < 0.33 else ("middle" if position_pct < 0.66 else "late")
        })
    
    return chunks


def multi_pass_retrieval(query: str, all_chunks: list, embedder, k: int = 5) -> list:
    """
    Multi-pass retrieval that ensures coverage across early/middle/late sections.
    
    Args:
        query: The backstory to search for
        all_chunks: List of chunk dicts with 'text' and 'section' keys
        embedder: Function to compute embeddings
        k: Total number of chunks to retrieve
    
    Returns:
        List of most relevant chunks with section diversity
    """
    # Separate by section
    early = [c for c in all_chunks if c.get("section") == "early"]
    middle = [c for c in all_chunks if c.get("section") == "middle"]
    late = [c for c in all_chunks if c.get("section") == "late"]
    
    # Compute query embedding
    query_emb = embedder(query)
    
    def get_top_from_section(section_chunks, n):
        if not section_chunks:
            return []
        
        # Simple similarity search
        scored = []
        for chunk in section_chunks:
            chunk_emb = embedder(chunk["text"])
            # Cosine similarity
            similarity = sum(a * b for a, b in zip(query_emb, chunk_emb))
            scored.append((similarity, chunk))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored[:n]]
    
    # Get proportional samples from each section
    results = []
    per_section = max(1, k // 3)
    
    results.extend(get_top_from_section(early, per_section))
    results.extend(get_top_from_section(middle, per_section))
    results.extend(get_top_from_section(late, per_section))
    
    # Fill remaining slots from any section
    remaining = k - len(results)
    if remaining > 0:
        all_remaining = early + middle + late
        for chunk in all_remaining:
            if chunk not in results:
                results.append(chunk)
                remaining -= 1
                if remaining <= 0:
                    break
    
    return results[:k]


def aggregate_evidence(chunks: list, separator: str = "\n\n---EVIDENCE---\n\n") -> str:
    """
    Combine multiple chunks into a single evidence string.
    """
    texts = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            section = chunk.get("section", "")
            if section:
                texts.append(f"[{section.upper()} SECTION]\n{text}")
            else:
                texts.append(text)
        else:
            texts.append(str(chunk))
    
    return separator.join(texts)
