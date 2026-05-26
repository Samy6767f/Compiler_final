"""
LLM layer — Hybrid: Groq (primary) + DeepSeek (fallback) for generation, NVIDIA for review
Optimised for low latency, high accuracy, and free tier reliability.
"""

import time, logging, os, json, re
from typing import Tuple
from openai import OpenAI

# Groq SDK (pip install groq)
try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False
    logging.getLogger("ai-compiler").warning("groq not installed. Generation fallback disabled.")

logger = logging.getLogger("ai-compiler")

# Environment variables
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY")

MODELS = {
    "generation": "llama-3.1-8b-instant",           
    "generation_primary": "llama-3.1-8b-instant",   
    "generation_fallback": "deepseek-ai/deepseek-v4-flash",
    "generation_groq_quality": "llama-3.3-70b-versatile",
    "review": "meta/llama-3.2-3b-instruct",
}

MAX_RETRIES = 2
RETRY_DELAY = 0.5

_nvidia_client = None
_groq_client = None

def _get_nvidia_client() -> OpenAI:
    global _nvidia_client
    if _nvidia_client is None:
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY not set.")
        _nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    return _nvidia_client

def _get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY and GROQ_SDK_AVAILABLE:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

# ── PROMPT COMPRESSION (same) ──────────────────────────────────────────────
_FILLER = re.compile(
    r'\b(please|kindly|basically|essentially|just|simply|really|very|quite|'
    r'actually|honestly|of course|i want to|i need|i would like|could you|'
    r'can you|make sure|ensure that|it should|it must|the system|the app|'
    r'the application|the platform|the website|the tool|the software)\b',
    re.IGNORECASE
)
_EXPAND = {
    r'\bauth\b': 'authentication', r'\badmin\b': 'administrator',
    r'\bdb\b': 'database', r'\brbac\b': 'role-based access control',
    r'\bsso\b': 'single sign-on', r'\bcrud\b': 'create/read/update/delete',
    r'\bnotifs?\b': 'notifications', r'\bmvp\b': 'minimum viable product',
}

def compress_prompt(raw: str) -> str:
    text = raw.strip()
    text = _FILLER.sub('', text)
    for pat, rep in _EXPAND.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip().rstrip('.!?')
    compressed = f"[COMPILER_INPUT] APP_SPEC: {text} [CONSTRAINTS: strict_json_only, no_markdown, no_explanation, all_fields_required, snake_case_keys]"
    logger.info(f"Prompt compressed: {len(raw)}→{len(compressed)} chars")
    return compressed

# ── JSON HELPERS (unchanged) ────────────────────────────────────────────────
def minify_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), separators=(',', ':'))
    except Exception:
        return text

def expand_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2)
    except Exception:
        return text

def repair_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    if not text.startswith('{'):
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            text = m.group()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        diff = text.count('{') - text.count('}')
        if diff > 0:
            text += '}' * diff
        try:
            json.loads(text)
            return text
        except Exception:
            try:
                from json_repair import repair_json as _jr
                return _jr(text)
            except Exception:
                return text

def _json_structurally_different(a: str, b: str) -> bool:
    try:
        return json.loads(a) != json.loads(b)
    except Exception:
        return a.strip() != b.strip()

# ── GENERATION (Groq primary, then DeepSeek, then Groq quality) ──────────────
def _call_groq(messages, max_tokens, model_name, timeout=30):
    client = _get_groq_client()
    if client is None:
        raise RuntimeError("Groq client unavailable")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response.choices[0].message.content or ""

def _call_nvidia(messages, max_tokens, timeout=45):
    client = _get_nvidia_client()
    response = client.chat.completions.create(
        model=MODELS["generation_fallback"],
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stream=False,
        timeout=timeout,
    )
    return response.choices[0].message.content or ""

def generate_with_llama(prompt: str, system_message: str,
                        max_tokens: int = 1024) -> str:  # Reduced from 2048
    """
    Generate using Groq (primary) for speed, fallback to DeepSeek, then Groq quality.
    """
    compressed = compress_prompt(prompt)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": compressed},
    ]

    # Try Groq fast model first (fastest)
    if GROQ_SDK_AVAILABLE and GROQ_API_KEY:
        try:
            t0 = time.time()
            raw = _call_groq(messages, max_tokens, MODELS["generation_primary"], timeout=30)
            logger.info(f"[Groq Fast] {len(raw)} chars in {time.time()-t0:.1f}s")
            if raw.strip():
                return repair_json(raw)
        except Exception as e:
            logger.warning(f"Groq fast generation failed: {e}. Falling back to DeepSeek.")

    # Fallback to DeepSeek (NVIDIA)
    try:
        t0 = time.time()
        raw = _call_nvidia(messages, max_tokens, timeout=45)
        logger.info(f"[DeepSeek] {len(raw)} chars in {time.time()-t0:.1f}s")
        if raw.strip():
            return repair_json(raw)
    except Exception as e:
        logger.warning(f"DeepSeek generation failed: {e}. Trying Groq quality...")

    # Final fallback: Groq quality model
    if GROQ_SDK_AVAILABLE and GROQ_API_KEY:
        try:
            t0 = time.time()
            raw = _call_groq(messages, max_tokens, MODELS["generation_groq_quality"], timeout=40)
            logger.info(f"[Groq Quality] {len(raw)} chars in {time.time()-t0:.1f}s")
            if raw.strip():
                return repair_json(raw)
        except Exception as e:
            logger.warning(f"Groq quality model failed: {e}")

    raise RuntimeError("All generation providers failed (Groq fast, DeepSeek, Groq quality)")

# ── REVIEW (NVIDIA only, with input type safety) ───────────────────────────
def review_with_model(draft: str, review_task: str,
                      max_tokens: int = 1024) -> Tuple[str, bool]:
    """
    Review using NVIDIA Llama 3.2 3B (fast, reliable).
    Returns (corrected_json_str, was_fixed).
    Handles both string and dict input.
    """
    if isinstance(draft, dict):
        draft = json.dumps(draft)
        logger.debug("Converted dict draft to JSON string for review.")

    mini = minify_json(draft)
    system = (
        "JSON_CORRECTOR. Fix ONLY structural errors and missing required fields. "
        "Keep all correct parts unchanged. Output ONLY raw minified JSON, no markdown, no explanation. "
        f"TASK: {review_task}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": mini},
    ]
    try:
        t0 = time.time()
        client = _get_nvidia_client()
        response = client.chat.completions.create(
            model=MODELS["review"],
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            stream=False,
            timeout=30,
        )
        raw = response.choices[0].message.content or ""
        logger.info(f"[NVIDIA Review] {len(mini)}B → {len(raw)}B in {time.time()-t0:.1f}s")
        if not raw.strip():
            return draft, False
        corrected = repair_json(raw)
        was_fixed = _json_structurally_different(draft, corrected)
        pretty = expand_json(corrected)
        logger.info(f"NVIDIA review: fixed={was_fixed}")
        return pretty, was_fixed
    except Exception as e:
        logger.warning(f"Review failed: {e}")
        return draft, False

# Keep old alias for compatibility
review_with_minimax = review_with_model
