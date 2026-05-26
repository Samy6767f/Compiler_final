import json
import re
import jsonschema
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any, Optional, Callable
from functools import lru_cache
from datetime import datetime

logger = logging.getLogger("ai-compiler")

# Regex precompiled
REPAIR_MARKDOWN_RE = re.compile(r"```(?:json)?\s*|\s*```")
REPAIR_BRACE_RE = re.compile(r"\{[\s\S]*\}")
REPAIR_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

# Pluralization helpers
PLURAL_IRREGULAR = {
    "person": "people", "man": "men", "woman": "women", "child": "children",
    "tooth": "teeth", "foot": "feet", "mouse": "mice", "ox": "oxen",
    "cactus": "cacti", "focus": "foci", "fungus": "fungi", "nucleus": "nuclei",
    "radius": "radii", "stimulus": "stimuli", "syllabus": "syllabi",
    "analysis": "analyses", "crisis": "crises", "diagnosis": "diagnoses",
    "hypothesis": "hypotheses", "thesis": "theses", "phenomenon": "phenomena",
    "criterion": "criteria", "datum": "data"
}
PLURAL_RULES = [
    (re.compile(r"(s|x|z|ch|sh)$"), r"\1es"),
    (re.compile(r"([^aeiou])y$"), r"\1ies"),
    (re.compile(r"(?:f|fe)$"), r"ves"),
    (re.compile(r"ss$"), r"sses"),
]

# Synonym mapping for key normalization
KEY_SYNONYMS = {
    "identifier": "id",
    "userid": "user_id",
    "useridentifier": "user_id",
    "created": "created_at",
    "updated": "updated_at",
    "timestamp": "created_at",
}


@dataclass
class ValidationResult:
    valid: bool
    data: Any = None
    errors: List[str] = field(default_factory=list)
    repaired: bool = False
    repairs_log: List[str] = field(default_factory=list)

    def __repr__(self):
        status = "VALID" if self.valid else "INVALID"
        if self.repaired:
            status += " (repaired)"
        return f"<ValidationResult {status} errors={self.errors}>"


class Validator:
    _validator_cache: Dict[str, jsonschema.Draft7Validator] = {}

    def __init__(self):
        self.schemas = {}
        self.repair_count = 0   # Track total repair attempts
        # Pluggable repair level registry
        self.repair_strategies: Dict[int, Callable] = {
            1: self._level1_repair,
            2: self._level2_repair,
            3: self._level3_repair,
        }

    @classmethod
    def _get_validator(cls, schema: Dict) -> jsonschema.Draft7Validator:
        """Return cached validator for schema using SHA256 key."""
        schema_key = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        if schema_key not in cls._validator_cache:
            cls._validator_cache[schema_key] = jsonschema.Draft7Validator(schema)
        return cls._validator_cache[schema_key]

    @staticmethod
    @lru_cache(maxsize=128)
    def _repair_json_cached(raw: str) -> Tuple[str, Tuple[str, ...]]:
        """Cached JSON repair. Returns (repaired_json_string, repairs_log_tuple)."""
        repairs = []
        raw = REPAIR_MARKDOWN_RE.sub("", raw.strip())

        texts = []
        if raw.startswith("["):
            texts.extend(REPAIR_ARRAY_RE.findall(raw))
        elif raw.startswith("{"):
            texts.extend(REPAIR_BRACE_RE.findall(raw))
        else:
            texts = REPAIR_BRACE_RE.findall(raw) + REPAIR_ARRAY_RE.findall(raw)

        if not texts:
            texts = [raw]

        repaired_texts = []
        for text in texts:
            try:
                json.loads(text)
                repaired_texts.append(text)
            except json.JSONDecodeError:
                try:
                    from json_repair import repair_json as repair
                    fixed = repair(text)
                    repairs.append(f"json_repair fixed: {text[:50]}...")
                    repaired_texts.append(fixed)
                except ImportError:
                    repaired_texts.append(text)

        if len(repaired_texts) > 1:
            repairs.append(f"Multiple JSON fragments found: {len(repaired_texts)} pieces")
        result = repaired_texts[0] if repaired_texts else raw
        return result, tuple(repairs)

    def repair_json(self, raw: str) -> Tuple[List[str], List[str]]:
        """Public wrapper – returns (list_of_json_strings, list_of_repair_logs)."""
        repaired_str, repairs_tuple = self._repair_json_cached(raw)
        return [repaired_str], list(repairs_tuple)

    def safe_json_parse(self, text: str) -> Tuple[bool, Any, str, List[str]]:
        """Safely parse JSON with repair. Returns (success, data, error_msg, repair_logs)."""
        repaired_str, repairs = self._repair_json_cached(text)
        try:
            return True, json.loads(repaired_str), "", list(repairs)
        except json.JSONDecodeError as e:
            return False, None, str(e), list(repairs)

    def validate(self, data: Any, schema: Dict, level: int = 1, schema_name: str = "unnamed") -> ValidationResult:
        """Validate data against schema, with repair attempts."""
        try:
            validator = self._get_validator(schema)
            validator.validate(data)
            logger.debug(f"[{schema_name}] Validation passed without repairs.")
            return ValidationResult(valid=True, data=data)
        except jsonschema.ValidationError as e:
            logger.info(f"[{schema_name}] Validation failed, attempting repair level {level}: {e}")
            return self._repair_and_validate(data, schema, e, level, schema_name)

    def _repair_and_validate(self, data: Any, schema: Dict, original_error: Exception,
                             level: int, schema_name: str) -> ValidationResult:
        all_errors = [str(original_error)]
        repairs_log = []

        # Use registry to get repair function
        repair_fn = self.repair_strategies.get(level, self._level3_repair)
        repaired, log = repair_fn(data, schema)
        repairs_log.extend(log)

        if repaired is None:
            logger.error(f"[{schema_name}] Repair returned None, validation failed.")
            return ValidationResult(valid=False, data=data, errors=all_errors,
                                    repairs_log=repairs_log)

        try:
            validator = self._get_validator(schema)
            validator.validate(repaired)
            logger.info(f"[{schema_name}] Validation succeeded after repairs: {repairs_log}")
            # Increment repair count if any repairs were applied
            if repairs_log:
                self.repair_count += 1
            return ValidationResult(valid=True, data=repaired, repaired=True,
                                    repairs_log=repairs_log)
        except jsonschema.ValidationError as e:
            all_errors.append(str(e))
            logger.warning(f"[{schema_name}] Repair failed at level {level}: {all_errors}")
            return ValidationResult(valid=False, data=repaired, errors=all_errors,
                                    repaired=True, repairs_log=repairs_log)

    # ---------- Repair Levels ----------
    def _level1_repair(self, data: Any, schema: Dict) -> Tuple[Any, List[str]]:
        """Lightweight type coercion for primitives, arrays, objects."""
        repairs = []
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in schema.get("properties", {}):
                    expected = schema["properties"][key].get("type", "string")
                    format_ = schema["properties"][key].get("format")
                    coerced, log = self._coerce_type_verbose(value, expected, format_)
                    if log:
                        repairs.append(log)
                    result[key] = coerced
                else:
                    result[key] = value
            return result, repairs
        return data, repairs

    def _level2_repair(self, data: Any, schema: Dict) -> Tuple[Any, List[str]]:
        """Add missing required fields using schema defaults or enum fallback."""
        repairs = []
        if isinstance(data, dict):
            result = {}
            required = set(schema.get("required", []))
            properties = schema.get("properties", {})

            for req in required:
                if req not in data:
                    prop = properties.get(req, {})
                    default = prop.get("default")
                    if default is None:
                        # If enum exists, pick first value as fallback
                        enum_vals = prop.get("enum")
                        if enum_vals and len(enum_vals) > 0:
                            default = enum_vals[0]
                            repairs.append(f"Added missing required field '{req}' with first enum value {default}")
                        else:
                            default = self._default_for_type(prop.get("type", "string"))
                            repairs.append(f"Added missing required field '{req}' with default {default}")
                    else:
                        repairs.append(f"Added missing required field '{req}' with schema default {default}")
                    result[req] = default

            for key, value in data.items():
                if key in properties:
                    prop_schema = properties[key]
                    expected = prop_schema.get("type", "string")
                    format_ = prop_schema.get("format")
                    coerced, log = self._coerce_type_verbose(value, expected, format_)
                    if log:
                        repairs.append(log)
                    result[key] = coerced
                else:
                    result[key] = value
            return result, repairs
        return data, repairs

    def _level3_repair(self, data: Any, schema: Dict) -> Tuple[Any, List[str]]:
        """Aggressive repair: normalize keys, apply synonyms, pluralization, drop unknown."""
        repaired, repairs = self._level2_repair(data, schema)
        if isinstance(repaired, dict):
            allowed_keys = set(schema.get("properties", {}).keys())
            normalized_mapping = {}
            for key in list(repaired.keys()):
                # Normalization steps
                norm = key.strip().lower()
                # Apply synonyms
                if norm in KEY_SYNONYMS:
                    norm = KEY_SYNONYMS[norm]
                    repairs.append(f"Applied synonym: '{key}' → '{norm}'")
                # Check pluralization
                if norm not in allowed_keys:
                    plural = self._check_pluralization(norm)
                    if plural in allowed_keys:
                        norm = plural
                        repairs.append(f"Pluralized: '{key}' → '{norm}'")
                # If still not allowed, drop
                if norm in allowed_keys:
                    if norm != key:
                        repairs.append(f"Normalized key '{key}' → '{norm}'")
                    normalized_mapping[norm] = repaired.pop(key)
                else:
                    repairs.append(f"Dropped unknown field '{key}'")
            # Update with normalized keys
            repaired.update(normalized_mapping)
        return repaired, repairs

    # ---------- Helpers ----------
    def _default_for_type(self, ftype: str) -> Any:
        defaults = {
            "string": "", "array": [], "object": {},
            "boolean": False, "integer": 0, "number": 0.0,
            "null": None
        }
        return defaults.get(ftype, "")

    def _coerce_type_verbose(self, value: Any, ftype: str, format_: Optional[str] = None) -> Tuple[Any, Optional[str]]:
        """Coerce value to target type with format handling (e.g., date-time)."""
        try:
            # Special handling for date-time strings
            if ftype == "string" and format_ == "date-time" and isinstance(value, str):
                try:
                    # Validate ISO format
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return value, None
                except ValueError:
                    return "", f"Invalid date-time format '{value}', replaced with empty string"
            if ftype == "integer":
                if isinstance(value, (int, float)):
                    return int(value), None
                return int(value), f"Coerced '{value}' to integer"
            if ftype == "number":
                if isinstance(value, (int, float)):
                    return float(value), None
                return float(value), f"Coerced '{value}' to number"
            if ftype == "boolean":
                if isinstance(value, bool):
                    return value, None
                if isinstance(value, str):
                    result = value.lower() in ("true", "1", "yes")
                    return result, f"Coerced string '{value}' to boolean {result}"
                return bool(value), f"Coerced {type(value).__name__} '{value}' to boolean"
            if ftype == "array":
                if isinstance(value, list):
                    return value, None
                return [value] if value is not None else [], f"Wrapped '{value}' in array"
            if ftype == "object":
                if isinstance(value, dict):
                    return value, None
                return {"value": value} if value is not None else {}, f"Wrapped '{value}' in object"
        except Exception as e:
            default = self._default_for_type(ftype)
            return default, f"Failed coercion, applied default ({ftype}): {e}"
        return value, None

    def _check_pluralization(self, word: str) -> str:
        """Return plural form of a word."""
        lower = word.lower()
        if lower in PLURAL_IRREGULAR:
            return PLURAL_IRREGULAR[lower]
        for pattern, replacement in PLURAL_RULES:
            if pattern.search(word):
                return pattern.sub(replacement, word)
        return word + "s"

    # ---------- Cross-Layer Validation (Enhanced) ----------
    def validate_cross_layer(self, design: Dict, schemas: Dict) -> Tuple[List[str], List[str]]:
        """
        Compiler‑semantic checks across DB, Auth, UI, and API layers.
        Returns (errors, warnings).
        """
        errors = []
        warnings = []

        db_schema = schemas.get("db", {})
        db_tables = list(db_schema.get("tables", {}).keys())
        relationships = db_schema.get("relationships", [])
        auth_schema = schemas.get("auth", {})
        auth_roles = set(auth_schema.get("roles", {}).keys())
        ui_routing = schemas.get("ui", {}).get("routing", {})
        api_endpoints = schemas.get("api", {}).get("endpoints", [])
        design_entities = {e["name"].lower() for e in design.get("entities", [])}

        # 1. Multi‑table relational integrity
        if len(db_tables) > 1 and len(relationships) == 0:
            errors.append("Compiler Semantic Fault: Multi-table database schema with zero relational foreign keys.")

        # 2. RBAC: UI roles must exist in Auth
        used_roles = set()
        for route, route_config in ui_routing.items():
            allowed_roles = route_config.get("allowed_roles", [])
            for role in allowed_roles:
                used_roles.add(role)
                if role not in auth_roles and role not in ["guest", "user"]:
                    errors.append(f"Security Mismatch: UI route '{route}' allows role '{role}' missing from Auth.")

        # 3. Unused roles warning
        unused_roles = auth_roles - used_roles - {"guest", "user"}
        if unused_roles:
            warnings.append(f"Unused roles in Auth schema: {unused_roles}")

        # 4. DB table ↔ design entity consistency
        for table in db_tables:
            if table.lower() not in design_entities:
                warnings.append(f"DB table '{table}' has no matching entity in design layer.")

        # 5. API endpoints must have at least one allowed role
        for endpoint in api_endpoints:
            roles = endpoint.get("roles", [])
            if not roles:
                warnings.append(f"API endpoint '{endpoint.get('path')}' has no allowed roles.")

        # 6. API endpoint resources should exist in DB tables (basic)
        for endpoint in api_endpoints:
            path = endpoint.get("path", "")
            segments = [s for s in path.split("/") if s and not s.startswith("{")]
            if segments:
                last = segments[-1]
                singular = last.rstrip("s")
                if (singular.lower() not in [t.lower() for t in db_tables] and
                        last.lower() not in [t.lower() for t in db_tables]):
                    warnings.append(f"API endpoint '{path}' references resource not in DB tables.")

        # 7. UI routes should have corresponding API resources (semantic check, not exact path match)
        # Build a set of API resources (e.g., "products") from endpoints
        api_resources = set()
        for ep in api_endpoints:
            path = ep.get("path", "")
            # Extract resource name from path (first segment after leading slash)
            segments = [s for s in path.split("/") if s and not s.startswith("{")]
            if segments:
                api_resources.add(segments[0])
        # Also add table names as fallback
        api_resources.update([t.lower() for t in db_tables])

        for route, route_config in ui_routing.items():
            # Skip common non‑resource pages
            if route in ["/login", "/dashboard","/"]:
                continue
            # Extract resource name from route (first segment after slash)
            parts = route.strip('/').split('/')
            resource = parts[0].lower()
            if resource not in api_resources:
                warnings.append(f"UI route '{route}' references resource '{resource}' with no matching API endpoint or DB table.")
            # Also check if there is a creation page but no POST endpoint for that resource
            if len(parts) > 1 and parts[1] == "new":
                has_post = any(ep.get('method') == 'POST' and ep.get('path', '').strip('/').split('/')[0] == resource
                               for ep in api_endpoints)
                if not has_post:
                    warnings.append(f"UI creation route '{route}' has no matching POST /{resource} API endpoint.")

        return errors, warnings
