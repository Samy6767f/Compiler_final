"""
LLM layer — Hybrid: DeepSeek for generation, Gemini for ultra‑fast review
Optimised for low latency, high accuracy, and free tier reliability.
"""

import time, logging, os, json, re
from typing import Tuple
from openai import OpenAI

# Google Gemini SDK (pip install google-generativeai)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.getLogger("ai-compiler").warning("google-generativeai not installed. Review will fall back to NVIDIA.")

logger = logging.getLogger("ai-compiler")

NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY")

MODELS = {
    # Fast generation model (DeepSeek-V4-Flash)
    "generation": "deepseek-ai/deepseek-v4-flash",
    # Fallback review model if Gemini fails
    "review_fallback": "meta/llama-3.2-3b-instruct",
}

MAX_RETRIES = 2
RETRY_DELAY = 0.5

_nvidia_client = None
_gemini_model = None

def _get_nvidia_client() -> OpenAI:
    global _nvidia_client
    if _nvidia_client is None:
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY not set.")
        _nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    return _nvidia_client

def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None and GEMINI_API_KEY and GEMINI_AVAILABLE:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel('models/gemini-2.0-flash-lite')
    return _gemini_model

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

# ── GENERATION (unchanged, still using NVIDIA DeepSeek) ──────────────────────
def _non_streaming_call(model: str, messages: list, temperature: float,
                        max_tokens: int, timeout: int) -> str:
    t0 = time.time()
    client = _get_nvidia_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        stream=False,
        timeout=timeout,
    )
    content = response.choices[0].message.content or ""
    logger.info(f"[{model.split('/')[-1]}] {len(content)} chars in {time.time()-t0:.1f}s")
    return content

def generate_with_llama(prompt: str, system_message: str,
                        max_tokens: int = 2048) -> str:
    compressed = compress_prompt(prompt)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": compressed},
    ]
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _non_streaming_call(
                MODELS["generation"], messages,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=60
            )
            if raw.strip():
                return repair_json(raw)
        except Exception as e:
            last_err = e
            logger.warning(f"Gen attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"Generation failed after {MAX_RETRIES} attempts: {last_err}")

# ── REVIEW: Gemini first, fallback to NVIDIA ─────────────────────────────────
def _fallback_review_with_nvidia(draft: str, review_task: str,
                                 max_tokens: int = 1024) -> Tuple[str, bool]:
    """Fallback review using NVIDIA Llama 3.2 3B."""
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
        raw = _non_streaming_call(
            MODELS["review_fallback"], messages,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=30
        )
        if not raw.strip():
            return draft, False
        corrected = repair_json(raw)
        was_fixed = _json_structurally_different(draft, corrected)
        pretty = expand_json(corrected)
        logger.info(f"NVIDIA fallback review: {len(mini)}B→{len(corrected)}B, fixed={was_fixed}")
        return pretty, was_fixed
    except Exception as e:
        logger.warning(f"Fallback review failed: {e}")
        return draft, False

def review_with_model(draft: str, review_task: str,
                      max_tokens: int = 1024) -> Tuple[str, bool]:
    """
    Ultra‑fast review using Gemini 2.0 Flash‑Lite.
    Falls back to NVIDIA Llama if Gemini unavailable or fails.
    """
    # Try Gemini first
    gemini_model = _get_gemini_model()
    if gemini_model is not None:
        mini = minify_json(draft)
        prompt = f"""{review_task}
Correct the following JSON. Output ONLY raw minified JSON, no markdown, no explanation.

JSON to correct:
{mini}
"""
        try:
            t0 = time.time()
            response = gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": max_tokens
                }
            )
            corrected = response.text
            logger.info(f"[Gemini Review] {len(mini)}B → {len(corrected)}B in {time.time()-t0:.1f}s")
            if corrected.strip():
                corrected = repair_json(corrected)
                was_fixed = _json_structurally_different(draft, corrected)
                pretty = expand_json(corrected)
                logger.info(f"Gemini review: fixed={was_fixed}")
                return pretty, was_fixed
        except Exception as e:
            logger.warning(f"Gemini review failed: {e}. Falling back to NVIDIA.")
    else:
        logger.warning("Gemini not available (missing key or SDK). Using fallback.")

    # Fallback to NVIDIA
    return _fallback_review_with_nvidia(draft, review_task, max_tokens)

# Keep old alias for compatibility
review_with_minimax = review_with_model
