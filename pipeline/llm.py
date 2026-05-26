"""
LLM layer — NVIDIA NIM free tier
Optimised for low latency:
  - Reduced max_tokens
  - Faster review model
  - Non‑streaming for short outputs
  - Aggressive timeouts with retry backoff
"""

import time, logging, os, json, re
from typing import Tuple
from openai import OpenAI

logger = logging.getLogger("ai-compiler")

NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

MODELS = {
    # Fast generation model (DeepSeek-V4-Flash)
    "generation": "deepseek-ai/deepseek-v4-flash",
    # Fast review model – Llama 3.2 3B is much quicker than MiniMax M2.7
    "review":     "meta/llama-3.2-3b-instruct",
}

MAX_RETRIES = 2
RETRY_DELAY = 0.5  # seconds

_nvidia_client = None


def _get_nvidia_client() -> OpenAI:
    global _nvidia_client
    if _nvidia_client is None:
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY not set.")
        _nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    return _nvidia_client


# ── PROMPT COMPRESSION (keep as is) ──────────────────────────────────────────

_FILLER = re.compile(
    r'\b(please|kindly|basically|essentially|just|simply|really|very|quite|'
    r'actually|honestly|of course|i want to|i need|i would like|could you|'
    r'can you|make sure|ensure that|it should|it must|the system|the app|'
    r'the application|the platform|the website|the tool|the software)\b',
    re.IGNORECASE
)
_EXPAND = {
    r'\bauth\b':    'authentication',
    r'\badmin\b':   'administrator',
    r'\bdb\b':      'database',
    r'\brbac\b':    'role-based access control',
    r'\bsso\b':     'single sign-on',
    r'\bcrud\b':    'create/read/update/delete',
    r'\bnotifs?\b': 'notifications',
    r'\bmvp\b':     'minimum viable product',
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


# ── JSON HELPERS (keep as is) ────────────────────────────────────────────────

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


# ── NON‑STREAMING CALL (faster for short outputs) ────────────────────────────

def _non_streaming_call(model: str, messages: list, temperature: float,
                        max_tokens: int, timeout: int) -> str:
    """Use non‑streaming for JSON responses – much lower latency."""
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


# ── PUBLIC API (with reduced token budgets) ──────────────────────────────────

def generate_with_llama(prompt: str, system_message: str,
                        max_tokens: int = 2048) -> str:   # reduced from 8192
    """Generate schema using DeepSeek-V4-Flash."""
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
                temperature=0.0,          # deterministic
                max_tokens=max_tokens,
                timeout=60                # 60 seconds max
            )
            if raw.strip():
                return repair_json(raw)
        except Exception as e:
            last_err = e
            logger.warning(f"Gen attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"Generation failed after {MAX_RETRIES} attempts: {last_err}")


def review_with_model(draft: str, review_task: str,
                      max_tokens: int = 1024) -> Tuple[str, bool]:   # reduced
    """
    Fast review using Llama 3.2 3B (non‑streaming, low max_tokens).
    Returns (corrected_json_str, was_fixed).
    """
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
            MODELS["review"], messages,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=30                     # 30 seconds max
        )
        if not raw.strip():
            return draft, False
        corrected = repair_json(raw)
        was_fixed = _json_structurally_different(draft, corrected)
        pretty = expand_json(corrected)
        logger.info(f"Review: {len(mini)}B→{len(corrected)}B, fixed={was_fixed}")
        return pretty, was_fixed
    except Exception as e:
        logger.warning(f"Review failed: {e}")
        return draft, False


review_with_minimax = review_with_model   # alias
