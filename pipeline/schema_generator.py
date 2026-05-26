import json, os, logging, re
from typing import Dict

logger = logging.getLogger("ai-compiler")

SCHEMA_GENERATION_PROMPT = """You are Stage 3 of an AI compiler — Schema Generator.
Generate complete JSON schemas for all entities in the system design.
Output ONLY raw JSON. No markdown. No explanation.

Output this exact shape:
{
  "db": {
    "tables": {
      "User": {
        "fields": {
          "id":         {"type": "uuid",      "primary_key": true},
          "email":      {"type": "string",    "unique": true},
          "role":       {"type": "enum"},
          "created_at": {"type": "timestamp"}
        }
      }
    },
    "relationships": []
  },
  "api": {
    "endpoints": [
      {"path": "/users",     "method": "GET",    "roles": ["admin"], "table": "User"},
      {"path": "/users",     "method": "POST",   "roles": ["admin"], "table": "User"},
      {"path": "/users/{id}","method": "PUT",    "roles": ["admin"], "table": "User"},
      {"path": "/users/{id}","method": "DELETE", "roles": ["admin"], "table": "User"}
    ]
  },
  "ui": {
    "pages": {"Dashboard": {"route": "/dashboard", "components": ["StatsCard","Table"]}},
    "routing": {"/dashboard": {"page": "Dashboard", "allowed_roles": ["admin","user"]}}
  },
  "auth": {
    "roles": {"admin": ["read","write","delete"], "user": ["read"]},
    "permissions": {}
  }
}"""


class SchemaGenerator:
    def __init__(self, schema_dir: str = None):
        # Use relative path — no hardcoded local paths
        self.schema_dir = schema_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas"
        )
        os.makedirs(self.schema_dir, exist_ok=True)

    # ── LLM-enhanced generation ───────────────────────────────────────────────

    def generate_llm(self, system_design: Dict) -> Dict:
        """Rule-based draft → MiniMax review and fix."""
        draft = self.generate_rule_based(system_design)

        try:
            from pipeline.llm import review_with_model, minify_json
            draft_json = json.dumps(draft)
            corrected, was_fixed = review_with_model(
                draft_json,
                "Ensure db.tables has fields with types and primary_key=true on id, "
                "api.endpoints has path/method/roles/table, "
                "ui.pages and ui.routing are populated, "
                "auth.roles has permission arrays"
            )
            if was_fixed:
                try:
                    draft = json.loads(corrected)
                    logger.info("Schema: MiniMax applied fixes")
                except json.JSONDecodeError:
                    logger.warning("Schema: MiniMax output unparseable, keeping rule-based draft")
        except Exception as e:
            logger.warning(f"Schema LLM review failed (using rule-based): {e}")

        self._save_schemas(draft)
        return draft

    def generate(self, system_design: Dict) -> Dict:
        return self.generate_llm(system_design)

    # ── Rule-based generation ─────────────────────────────────────────────────

    def generate_rule_based(self, system_design: Dict) -> Dict:
        entities = system_design.get("entities", [])
        roles    = system_design.get("roles", [])
        pages    = system_design.get("pages", [])

        schemas = {
            "db":   {"tables": {}, "relationships": []},
            "api":  {"endpoints": []},
            "ui":   {"pages": {}, "routing": {}},
            "auth": {"roles": {}, "permissions": {}}
        }

        role_names = [r.get("name", "") for r in roles]

        for entity in entities:
            name   = entity.get("name", "Unknown")
            fields = entity.get("fields", [])
            rels   = entity.get("relations", [])

            # DB table
            table_fields = {}
            for field in fields:
                parts = field.split(":")
                fname = parts[0]
                ftype = parts[1] if len(parts) > 1 else "string"
                table_fields[fname] = {
                    "type":        ftype,
                    "primary_key": fname == "id",
                    "nullable":    fname not in ("id", "created_at"),
                }
            # Guarantee id + timestamps
            if "id" not in table_fields:
                table_fields = {"id": {"type": "uuid", "primary_key": True, "nullable": False}, **table_fields}
            for ts in ("created_at", "updated_at"):
                if ts not in table_fields:
                    table_fields[ts] = {"type": "timestamp", "primary_key": False, "nullable": False}

            schemas["db"]["tables"][name] = {"fields": table_fields}

            # Relationships
            for rel in rels:
                if isinstance(rel, dict):
                    schemas["db"]["relationships"].append({
                        "from": name, "to": rel.get("target", ""),
                        "type": rel.get("type", "many-to-one"),
                        "foreign_key": rel.get("foreign_key", ""),
                    })

            # Determine plural route segment
            lower = name.lower()
            if lower in ("person",): plural = "people"
            elif lower.endswith(("s","x","z","ch","sh")): plural = lower + "es"
            elif lower.endswith("y") and lower[-2] not in "aeiou": plural = lower[:-1] + "ies"
            else: plural = lower + "s"

            write_roles = ["admin"] if "admin" in role_names else role_names[:1] or ["admin"]
            read_roles  = role_names if role_names else ["admin", "user"]

            schemas["api"]["endpoints"].extend([
                {"path": f"/{plural}",      "method": "GET",    "roles": read_roles,  "table": name},
                {"path": f"/{plural}",      "method": "POST",   "roles": write_roles, "table": name},
                {"path": f"/{plural}/{{id}}","method": "GET",    "roles": read_roles,  "table": name},
                {"path": f"/{plural}/{{id}}","method": "PUT",    "roles": write_roles, "table": name},
                {"path": f"/{plural}/{{id}}","method": "DELETE", "roles": ["admin"],   "table": name},
            ])

        # Auth roles
        for role in roles:
            rn   = role.get("name", "user")
            perms = role.get("permissions", ["read"])
            schemas["auth"]["roles"][rn] = perms

        # UI pages
        for page in pages:
            pname  = page.get("name", "Page")
            proute = page.get("route", "/" + pname.lower().replace(" ", "_"))
            schemas["ui"]["pages"][pname] = {
                "route":      proute,
                "components": page.get("components", []),
            }
            schemas["ui"]["routing"][proute] = {
                "page":          pname,
                "allowed_roles": page.get("allowed_roles", role_names or ["user"]),
            }

        return schemas

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_schemas(self, schemas: Dict) -> None:
        try:
            for name, schema in schemas.get("db", {}).get("tables", {}).items():
                path = os.path.join(self.schema_dir, f"{name.lower()}.json")
                with open(path, "w") as f:
                    json.dump(schema, f, indent=2)
        except Exception as e:
            logger.warning(f"Schema save failed: {e}")

    def _parse_and_save(self, raw: str) -> Dict:
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().replace("```", "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', text)
            data = json.loads(m.group()) if m else {}
        self._save_schemas(data)
        return data
