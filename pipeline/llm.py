"""
LLM layer — NVIDIA NIM free tier only  
Flow per pipeline stage:
  1. compress_prompt()      — strip filler, ~35% token reduction
  2. generate_with_llama()  — DeepSeek-R1 on NVIDIA NIM (best free reasoning)
  3. minify_json()          — shrink draft ~60% before review
  4. review_with_model()    — MiniMax M2.7 reviews minified JSON only
  5. expand_json()          — pretty-print final output
"""

import time, logging, os, json, re
from typing import Tuple
from openai import OpenAI

logger = logging.getLogger("ai-compiler")

NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

MODELS = {
    # DeepSeek-R1: best free reasoning model on NVIDIA NIM for structured generation
    "generation": "deepseek-ai/deepseek-r1",
    # MiniMax: fast reviewer — only receives minified JSON so it's quick
    "review":     "minimaxai/minimax-m2.7",
}

MAX_RETRIES = 2
RETRY_DELAY = 1

_nvidia_client = None


def _get_nvidia_client() -> OpenAI:
    global _nvidia_client
    if _nvidia_client is None:
        if not NVIDIA_API_KEY:
            raise RuntimeError(
                "NVIDIA_API_KEY not set. Add it to Railway → Variables."
            )
        _nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    return _nvidia_client


# ── PROMPT COMPRESSION ────────────────────────────────────────────────────────

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
    """Strip filler words and normalise abbreviations — ~35% token reduction."""
    text = raw.strip()
    text = _FILLER.sub('', text)
    for pat, rep in _EXPAND.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip().rstrip('.!?')
    compressed = (
        f"[COMPILER_INPUT] APP_SPEC: {text} "
        f"[CONSTRAINTS: strict_json_only, no_markdown, no_explanation, "
        f"all_fields_required, snake_case_keys]"
    )
    logger.info(
        f"Prompt compressed: {len(raw)}→{len(compressed)} chars "
        f"({100 - int(len(compressed)/max(len(raw),1)*100)}% reduction)"
    )
    return compressed


# ── JSON HELPERS ──────────────────────────────────────────────────────────────

def minify_json(text: str) -> str:
    """Remove whitespace — cuts payload ~60% before sending to MiniMax."""
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
    """Best-effort JSON extraction and repair."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()
    # Strip DeepSeek-R1 think tags
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    # Find JSON object
    if not text.startswith('{'):
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            text = m.group()
    # Quick parse
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    # Fix unbalanced braces
    diff = text.count('{') - text.count('}')
    if diff > 0:
        text += '}' * diff
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    # json_repair fallback
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


def _safe_get_content(completion) -> str:
    if not completion.choices:
        return ""
    choice = completion.choices[0]
    # Non-streaming response
    if hasattr(choice, 'message'):
        return choice.message.content or ""
    return ""


# ── STREAMING CALL ────────────────────────────────────────────────────────────

def _stream_call(model: str, messages: list, temperature: float,
                 max_tokens: int, timeout: int = 90) -> str:
    t0 = time.time()
    client = _get_nvidia_client()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=max(temperature, 0.01),
        top_p=0.9,
        max_tokens=max_tokens,
        stream=True,
        timeout=timeout,
    )
    chunks = []
    for chunk in completion:
        if getattr(chunk, 'choices', None) and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    text = ''.join(chunks)
    logger.info(f"[{model.split('/')[-1]}] {len(text)} chars in {time.time()-t0:.1f}s")
    return text


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def generate_with_llama(prompt: str, system_message: str,
                        max_tokens: int = 8192) -> str:
    """
    Generate using DeepSeek-R1 on NVIDIA NIM.
    Prompt is compressed before sending.
    Falls back gracefully.
    """
    compressed = compress_prompt(prompt)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": compressed},
    ]
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _stream_call(MODELS["generation"], messages, 0.05, max_tokens)
            if raw.strip():
                return repair_json(raw)
        except Exception as e:
            last_err = e
            logger.warning(f"DeepSeek attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"DeepSeek-R1 failed after {MAX_RETRIES} attempts: {last_err}")


def review_with_model(draft: str, review_task: str,
                      max_tokens: int = 4096) -> Tuple[str, bool]:
    """
    MiniMax reviews MINIFIED JSON — faster, cheaper, same quality.
    Returns (corrected_json_str, was_fixed).
    """
    mini = minify_json(draft)

    system = (
        "JSON_CORRECTOR. Fix ONLY structural errors and missing required fields. "
        "Keep all correct parts unchanged. "
        f"Output ONLY raw minified JSON, no markdown, no explanation. "
        f"TASK: {review_task}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": mini},
    ]

    try:
        raw = _stream_call(MODELS["review"], messages, 0.02, max_tokens, timeout=60)
        if not raw.strip():
            return draft, False
        corrected = repair_json(raw)
        was_fixed  = _json_structurally_different(draft, corrected)
        pretty     = expand_json(corrected)
        logger.info(
            f"MiniMax review: {len(mini)}B→{len(corrected)}B, was_fixed={was_fixed}"
        )
        return pretty, was_fixed
    except Exception as e:
        logger.warning(f"MiniMax review failed: {e}")
        return draft, False


# Keep old alias for compatibility
review_with_minimax = review_with_model
