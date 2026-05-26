import os
import json
import time
from typing import List
from google import genai
from google.genai.errors import APIError


def eval(
    contents: List[str],
    cuts: List[int],
    unit_coherence: float = 0.6,
    alpha: float = 1.0,
) -> float:
    """Evaluates segmentation coherence."""
    n = len(contents)
    if len(cuts) != n-1:
        raise ValueError("cuts must have length n-1")

    seg_ranges = []
    start = 0
    for i, cut in enumerate(cuts):
        if cut == 1:
            seg_ranges.append((start, i+1))
            start = i+1
    seg_ranges.append((start, n))

    total = 0.0
    for l, r in seg_ranges:
        total += get_coherence(contents, l, r, unit_coherence=unit_coherence)
    return total - alpha * len(seg_ranges)



# ---------- Load API keys from keys.txt ----------
KEYS_FILE = "keys.txt"
MODEL_NAME = "gemini-3.1-flash-lite"
CACHE_FILE_LLM = "data/llm_cache.json"


def load_api_keys() -> List[str]:
    """Reads keys.txt and returns a list of non-empty keys."""
    if not os.path.exists(KEYS_FILE):
        raise FileNotFoundError(f"Key file {KEYS_FILE} not found")
    with open(KEYS_FILE, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError(f"No valid keys found in {KEYS_FILE}")
    return keys

API_KEYS = load_api_keys()
CREDENTIALS = [(key, MODEL_NAME) for key in API_KEYS]   # (api_key, model_name)

_current_cred_index = 0

def get_current_credential():
    """Returns the current (api_key, model_name) tuple."""
    return CREDENTIALS[_current_cred_index % len(CREDENTIALS)]

def rotate_credential():
    """Rotates to the next API key in the list."""
    global _current_cred_index
    _current_cred_index += 1
    print(f"🔄 Rotating credential to index: {_current_cred_index % len(CREDENTIALS)}")



def get_coherence(
    contents: List[str],
    l: int,
    r: int,
    unit_coherence: float = 0.6,
    base_wait: float = 2.0,      
) -> float:
    """
    Evaluates semantic coherence for a segment.
    
    Behavior:
    - Only rotates API key on 429 (Resource Exhausted) errors.
    - For any error (including 429 and others), uses exponential backoff.
    - Raises exception only after 5 consecutive errors.
    """
    if r - l == 1:
        return unit_coherence

    segment_text = " ".join(contents[l:r])
    os.makedirs(os.path.dirname(CACHE_FILE_LLM), exist_ok=True)

    # Load cache
    cache = {}
    if os.path.exists(CACHE_FILE_LLM):
        with open(CACHE_FILE_LLM, "r", encoding="utf-8") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}

    if segment_text in cache:
        return cache[segment_text]

    # Prompt remains in Spanish as requested
    prompt = f"""Evalúa la coherencia semántica del siguiente fragmento de texto en una escala de 0 a 1, donde 0 es completamente incoherente y 1 es perfectamente coherente. Responde ÚNICAMENTE con un número decimal entre 0 y 1.

            Texto: "{segment_text}"

            Coherencia:"""

    max_consecutive_errors = 5
    consecutive_errors = 0
    attempt = 0   # for exponential backoff

    while True:
        api_key, model_name = get_current_credential()
        client = genai.Client(api_key=api_key)

        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            coherence = float(response.text.strip())

            # Success: store in cache and return
            cache[segment_text] = coherence
            with open(CACHE_FILE_LLM, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            return coherence

        except Exception as e:
            consecutive_errors += 1
            wait_time = base_wait * (2 ** attempt)   # exponential backoff
            is_429 = isinstance(e, APIError) and hasattr(e, 'code') and e.code == 429

            if is_429:
                print(f"🛑 [Error 429] Resource exhausted with key {api_key[:8]}... "
                      f"(failure {consecutive_errors}/{max_consecutive_errors}). "
                      f"Rotating key and waiting {wait_time:.1f}s")
                rotate_credential()   # rotate only on 429
            else:
                print(f"⚠️ Error ({type(e).__name__}): {str(e)} with key {api_key[:8]}... "
                      f"(failure {consecutive_errors}/{max_consecutive_errors}). "
                      f"Retrying with same key in {wait_time:.1f}s")

            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    f"🚨 {max_consecutive_errors} consecutive failures. "
                    f"Last error: {e}"
                )

            # Exponential backoff before next attempt
            time.sleep(wait_time)
            attempt += 1   # increase for next wait