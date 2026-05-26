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
        self.schema_dir = schema_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas"
        )
        os.makedirs(self.schema_dir, exist_ok=True)

    def _dedupe_endpoints(self, endpoints: list) -> list:
        """Remove duplicate endpoints based on (path, method)."""
        seen = set()
        unique = []
        for ep in endpoints:
            key = (ep.get("path"), ep.get("method"))
            if key not in seen:
                seen.add(key)
                unique.append(ep)
        return unique

    def generate_llm(self, system_design: Dict) -> Dict:
        """Rule‑based draft → MiniMax review and fix (without breaking data)."""
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
                    fixed = json.loads(corrected)
                    draft = self._merge_schemas(draft, fixed)
                    # Deduplicate endpoints after merging
                    draft["api"]["endpoints"] = self._dedupe_endpoints(draft["api"]["endpoints"])
                    logger.info("Schema: MiniMax applied fixes (merged)")
                except json.JSONDecodeError:
                    logger.warning("Schema: MiniMax output unparseable, keeping rule‑based draft")
        except Exception as e:
            logger.warning(f"Schema LLM review failed (using rule‑based): {e}")

        self._save_schemas(draft)
        return draft

    def generate(self, system_design: Dict) -> Dict:
        return self.generate_llm(system_design)

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

        role_names = [r.get("name", "") for r in roles if isinstance(r.get("name"), str)]
        if not role_names:
            role_names = ["user", "admin"]
            schemas["auth"]["roles"] = {"user": ["read"], "admin": ["create", "read", "update", "delete"]}
        else:
            for role in roles:
                rn = role.get("name", "user")
                perms = role.get("permissions", ["read"])
                schemas["auth"]["roles"][rn] = perms

        # Helper to pluralise a singular noun (or detect already plural)
        def to_plural(word: str) -> str:
            lower = word.lower()
            if lower.endswith('s'):
                return lower
            if lower in ("person",):
                return "people"
            if lower.endswith(("x", "z", "ch", "sh")):
                return lower + "es"
            if lower.endswith("y") and lower[-2] not in "aeiou":
                return lower[:-1] + "ies"
            return lower + "s"

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
                    "nullable":    fname not in ("id", "created_at", "updated_at"),
                }
            # Guarantee id + timestamps
            if "id" not in table_fields:
                table_fields["id"] = {"type": "uuid", "primary_key": True, "nullable": False}
            for ts in ("created_at", "updated_at"):
                if ts not in table_fields:
                    table_fields[ts] = {"type": "timestamp", "primary_key": False, "nullable": False}

            schemas["db"]["tables"][name] = {"fields": table_fields}

            # Relationships (from system design)
            for rel in rels:
                if isinstance(rel, dict) and rel.get("target") and rel.get("foreign_key"):
                    schemas["db"]["relationships"].append({
                        "from": name,
                        "to": rel.get("target"),
                        "type": rel.get("type", "many-to-one"),
                        "foreign_key": rel.get("foreign_key"),
                    })

            # API endpoints with proper pluralisation
            plural = to_plural(name)
            write_roles = ["admin"] if "admin" in role_names else role_names[:1] or ["admin"]
            read_roles = role_names if role_names else ["admin", "user"]

            schemas["api"]["endpoints"].extend([
                {"path": f"/{plural}",      "method": "GET",    "roles": read_roles,  "table": name},
                {"path": f"/{plural}",      "method": "POST",   "roles": write_roles, "table": name},
                {"path": f"/{plural}/{{id}}","method": "GET",    "roles": read_roles,  "table": name},
                {"path": f"/{plural}/{{id}}","method": "PUT",    "roles": write_roles, "table": name},
                {"path": f"/{plural}/{{id}}","method": "DELETE", "roles": ["admin"],   "table": name},
            ])

        # UI pages (from design pages)
        if pages:
            for page in pages:
                pname = page.get("name", "Page")
                proute = page.get("route", "/" + pname.lower().replace(" ", "_"))
                schemas["ui"]["pages"][pname] = {
                    "route": proute,
                    "components": page.get("components", []),
                }
                schemas["ui"]["routing"][proute] = {
                    "page": pname,
                    "allowed_roles": page.get("allowed_roles", role_names or ["user"]),
                }
        else:
            schemas["ui"]["pages"]["Dashboard"] = {"route": "/dashboard", "components": ["StatsCard", "Table"]}
            schemas["ui"]["routing"]["/dashboard"] = {"page": "Dashboard", "allowed_roles": role_names or ["user"]}

        # Default API endpoint if nothing exists
        if not schemas["api"]["endpoints"]:
            schemas["api"]["endpoints"] = [
                {"path": "/health", "method": "GET", "roles": ["guest"], "table": "none"}
            ]
            logger.warning("No entities found, added default health endpoint")

        # Deduplicate endpoints before returning
        schemas["api"]["endpoints"] = self._dedupe_endpoints(schemas["api"]["endpoints"])
        return schemas

    def _merge_schemas(self, base: Dict, incoming: Dict) -> Dict:
        """Merge incoming fixes into base, preserving existing fields but NOT adding new tables or endpoints."""
        if not isinstance(incoming, dict):
            return base

        # DB tables: only merge fields into existing tables; ignore new tables
        base_tables = base["db"]["tables"]
        for table_name, table_data in incoming.get("db", {}).get("tables", {}).items():
            if table_name in base_tables:
                base_fields = base_tables[table_name].get("fields", {})
                for field_name, field_props in table_data.get("fields", {}).items():
                    if field_name not in base_fields:
                        base_fields[field_name] = field_props
            else:
                logger.debug(f"Ignoring new table '{table_name}' added by review (not in original design)")

        # Relationships: only append new ones (no harm)
        for rel in incoming.get("db", {}).get("relationships", []):
            if rel not in base["db"]["relationships"]:
                base["db"]["relationships"].append(rel)

        # API endpoints: only keep those where the table exists in base DB
        base_tables_set = set(base["db"]["tables"].keys())
        for ep in incoming.get("api", {}).get("endpoints", []):
            table = ep.get("table")
            if table in base_tables_set:
                if ep not in base["api"]["endpoints"]:
                    base["api"]["endpoints"].append(ep)
            else:
                logger.debug(f"Ignoring endpoint for non‑existent table '{table}' added by review")

        # UI pages and routing: add only if not already present
        for page_name, page_data in incoming.get("ui", {}).get("pages", {}).items():
            if page_name not in base["ui"]["pages"]:
                base["ui"]["pages"][page_name] = page_data
        for route, route_data in incoming.get("ui", {}).get("routing", {}).items():
            if route not in base["ui"]["routing"]:
                base["ui"]["routing"][route] = route_data

        # Auth roles: add new roles only if not present
        for role_name, perms in incoming.get("auth", {}).get("roles", {}).items():
            if role_name not in base["auth"]["roles"]:
                base["auth"]["roles"][role_name] = perms

        return base

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
