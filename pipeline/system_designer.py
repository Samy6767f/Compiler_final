import json
import logging
import re
from typing import Dict, List, Any
import jsonschema

logger = logging.getLogger("ai-compiler")

SYSTEM_Design_SCHEMA = {  # (same as yours, unchanged)
    "type": "object",
    "required": ["entities", "flows", "roles", "permissions", "pages"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "fields", "relations"],
                "properties": {
                    "name": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "string"}},
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["target", "type", "foreign_key"],
                            "properties": {
                                "target": {"type": "string"},
                                "type": {"type": "string"},
                                "foreign_key": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        "flows": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "steps", "actors"],
                "properties": {
                    "name": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "actors": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "permissions"],
                "properties": {
                    "name": {"type": "string"},
                    "permissions": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "permissions": {"type": "object"},
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "route", "allowed_roles", "components"],
                "properties": {
                    "name": {"type": "string"},
                    "route": {"type": "string"},
                    "allowed_roles": {"type": "array", "items": {"type": "string"}},
                    "components": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    }
}

SYSTEM_DESIGN_PROMPT = """You are Stage 2 of an AI Compiler — System Designer.
Your task is to convert an application intent structure into a detailed architectural design framework.

Generate complete application architecture components as valid JSON matching the system schema guidelines.
Output ONLY raw JSON. Do not include markdown code block backticks (```), thought processes, or narrative text.

Ensure:
1. Every entity contains basic system schema auditing fields: 'id:uuid', 'created_at:timestamp', 'updated_at:timestamp'.
2. Relationship joins are accurately mapped between logical dependent keys.
3. Every page contains explicit frontend route paths and associated authorization arrays.

Required Format Shape:
{"entities":[{"name":"EntityName","fields":["id:uuid","name:string","created_at:timestamp"],"relations":[]}],"flows":[{"name":"FlowName","steps":["step1","step2"],"actors":["role"]}],"roles":[{"name":"rolename","permissions":["read","write"]}],"permissions":{"permName":["role"]},"pages":[{"name":"PageName","route":"/route","allowed_roles":["role"],"components":["Component"]}]}"""


class SystemDesigner:
    def design(self, intent: Dict) -> Dict:
        """Main compilation entry point."""
        return self.design_llm(intent)

    def design_llm(self, intent: Dict) -> Dict:
        """LLM‑driven synthesis."""
        try:
            from pipeline.llm import _get_nvidia_client, MODELS
            client = _get_nvidia_client()

            messages = [
                {"role": "system", "content": SYSTEM_DESIGN_PROMPT},
                {"role": "user", "content": f"Application Intent Map to Expand: {json.dumps(intent)}"}
            ]

            response = client.chat.completions.create(
                model=MODELS["generation"],
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            raw_content = response.choices[0].message.content

            if "</thought>" in raw_content:
                raw_content = raw_content.split("</thought>")[-1].strip()

            draft = self._parse_and_validate(raw_content)

            # Downstream review flow
            try:
                from pipeline.llm import review_with_model
                draft_json = json.dumps(draft)
                corrected, was_fixed = review_with_model(
                    draft_json,
                    "Ensure all entities have fields (id, name, created_at), roles have permissions, pages have routes and allowed_roles, permissions mapping is correct"
                )
                if was_fixed:
                    draft = self._parse_and_validate(corrected)
                    logger.info("Design Phase: Architecture optimized via validation verification flow.")
            except Exception as e:
                logger.warning(f"Downstream design check skipped: {e}")

            return draft

        except Exception as e:
            logger.error(f"LLM design generation failed, falling back to rule‑based: {e}")
            return self.design_rule_based(intent)

    def _normalize_relation(self, rel: Any) -> Dict:
        """Convert string relation to object, and ensure dict has required keys."""
        if isinstance(rel, dict):
            return {
                "target": rel.get("target", ""),
                "type": rel.get("type", "many-to-one"),
                "foreign_key": rel.get("foreign_key", "")
            }
        if isinstance(rel, str):
            parts = rel.split("->")
            if len(parts) == 2:
                left = parts[0].strip().split(".")
                right = parts[1].strip().split(".")
                foreign_key = left[-1] if len(left) > 1 else left[0]
                target_table = right[0].strip() if right else ""
                return {
                    "target": target_table.title(),
                    "type": "many-to-one",
                    "foreign_key": foreign_key
                }
            return {
                "target": "Unknown",
                "type": "many-to-one",
                "foreign_key": rel
            }
        return {"target": "", "type": "many-to-one", "foreign_key": ""}

    def _parse_and_validate(self, raw: str) -> Dict:
        """Parse, repair, inject missing keys, normalise relations, validate."""
        # Clean markdown
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().replace("```", "").strip()

        # Attempt to parse JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                data = json.loads(match.group())
            else:
                # Fallback: return minimal valid structure
                logger.error("No JSON object found, using empty structure")
                data = {}

        # Safety: ensure data is a dict (not list, not None)
        if not isinstance(data, dict):
            logger.warning(f"Parsed data is {type(data).__name__}, wrapping in dict")
            data = {"entities": []}

        # --- Ensure all required top‑level keys exist ---
        required_keys = ["entities", "flows", "roles", "permissions", "pages"]
        for key in required_keys:
            if key not in data:
                data[key] = [] if key != "permissions" else {}
                logger.warning(f"Added missing required top‑level key: '{key}'")

        # Ensure permissions is always a dict
        if not isinstance(data.get("permissions"), dict):
            data["permissions"] = {}
            logger.warning("Reset 'permissions' to empty dict")

        # --- Inject standard fields into each entity ---
        for entity in data.get("entities", []):
            if not isinstance(entity, dict):
                continue
            if "fields" not in entity or not isinstance(entity["fields"], list):
                entity["fields"] = []
            fields = entity["fields"]
            field_names = [f.split(":")[0].strip() for f in fields]
            if "id" not in field_names:
                entity["fields"].insert(0, "id:uuid")
            if "created_at" not in field_names:
                entity["fields"].append("created_at:timestamp")
            if "updated_at" not in field_names:
                entity["fields"].append("updated_at:timestamp")

        # --- Normalise relations (convert string to object) ---
        for entity in data.get("entities", []):
            if not isinstance(entity, dict):
                continue
            if "relations" not in entity or not isinstance(entity["relations"], list):
                entity["relations"] = []
            entity["relations"] = [self._normalize_relation(r) for r in entity["relations"]]

        # --- Validate against schema (non‑fatal) ---
        try:
            jsonschema.validate(instance=data, schema=SYSTEM_DESIGN_SCHEMA)
        except jsonschema.ValidationError as e:
            logger.warning(f"Schema validation warning (non‑fatal): {e}")

        return data

    # ---------- Rule‑based fallback (unchanged, works fine) ----------
    def design_rule_based(self, intent: Dict) -> Dict:
        entities = self._design_entities(intent)
        roles = self._design_roles(intent)
        return {
            "entities": entities,
            "flows": self._design_flows(entities, roles),
            "roles": roles,
            "permissions": self._generate_permissions(roles),
            "pages": self._design_pages(entities, roles)
        }

    def _design_entities(self, intent: Dict) -> List[Dict]:
        # ... (your existing implementation, unchanged) ...
        entities = []
        entity_names = intent.get("entities", [])
        base_fields = {
            'users': ['id:uuid', 'email:string', 'password_hash:string', 'role:enum', 'created_at:timestamp', 'updated_at:timestamp'],
            'contacts': ['id:uuid', 'name:string', 'email:string', 'phone:string', 'company:string', 'created_at:timestamp', 'updated_at:timestamp'],
            'customers': ['id:uuid', 'name:string', 'email:string', 'phone:string', 'address:text', 'created_at:timestamp', 'updated_at:timestamp'],
            'products': ['id:uuid', 'name:string', 'description:text', 'price:float', 'stock:integer', 'created_at:timestamp', 'updated_at:timestamp'],
            'orders': ['id:uuid', 'customer_id:uuid', 'total:float', 'status:enum', 'created_at:timestamp', 'updated_at:timestamp'],
            'payments': ['id:uuid', 'order_id:uuid', 'amount:float', 'method:string', 'status:enum', 'created_at:timestamp', 'updated_at:timestamp'],
            'invoices': ['id:uuid', 'order_id:uuid', 'amount:float', 'status:enum', 'due_date:timestamp', 'created_at:timestamp', 'updated_at:timestamp'],
            'clinics': ['id:uuid', 'name:string', 'address:string', 'created_at:timestamp', 'updated_at:timestamp'],
            'doctors': ['id:uuid', 'name:string', 'specialty:string', 'clinic_id:uuid', 'created_at:timestamp', 'updated_at:timestamp'],
            'patients': ['id:uuid', 'name:string', 'email:string', 'phone:string', 'clinic_id:uuid', 'created_at:timestamp', 'updated_at:timestamp'],
            'medical_records': ['id:uuid', 'patient_id:uuid', 'doctor_id:uuid', 'diagnosis:text', 'created_at:timestamp', 'updated_at:timestamp'],
        }
        has_multi_tenant = any('clinic' in e.lower() or 'tenant' in e.lower() for e in entity_names)
        for name in entity_names:
            name_lower = name.lower()
            fields = base_fields.get(name_lower, ['id:uuid', 'name:string', 'created_at:timestamp', 'updated_at:timestamp'])
            relations = []
            if has_multi_tenant and name_lower not in ['clinics', 'tenants']:
                relations.append({"target": "Clinics", "type": "many-to-one", "foreign_key": "clinic_id"})
            if name_lower in ['doctors', 'patients', 'medical_records']:
                if 'Doctors' not in entity_names and 'doctors' not in entity_names:
                    relations.append({"target": "Doctors", "type": "many-to-one", "foreign_key": "doctor_id"})
                if 'Patients' not in entity_names and 'patients' not in entity_names:
                    relations.append({"target": "Patients", "type": "many-to-one", "foreign_key": "patient_id"})
            entities.append({"name": name.title(), "fields": fields, "relations": relations})
        if not entities:
            entities.append({"name": "Item", "fields": ['id:uuid', 'name:string', 'created_at:timestamp', 'updated_at:timestamp'], "relations": []})
        return entities

    def _design_roles(self, intent: Dict) -> List[Dict]:
        roles_data = intent.get("roles", [])
        roles = []
        for role in roles_data:
            name = role.get("name", "user") if isinstance(role, dict) else role
            if name == "admin":
                perms = ["create", "read", "update", "delete", "admin"]
            elif name == "guest":
                perms = ["read"]
            else:
                perms = ["create", "read", "update"]
            roles.append({"name": name, "permissions": perms})
        return roles

    def _generate_permissions(self, roles: List[Dict]) -> Dict:
        permissions = {}
        for role in roles:
            role_name = role["name"]
            perms = role.get("permissions", [])
            for perm in perms:
                if perm not in permissions:
                    permissions[perm] = []
                if role_name not in permissions[perm]:
                    permissions[perm].append(role_name)
        return permissions

    def _design_flows(self, entities: List[Dict], roles: List[Dict]) -> List[Dict]:
        flows = [
            {"name": "User Authentication", "steps": ["POST /auth/login", "Validate credentials", "Return JWT token"], "actors": ["guest"]},
            {"name": "Manage Resources", "steps": ["GET /{resource}", "POST /{resource}", "PUT /{resource}/:id", "DELETE /{resource}/:id"], "actors": list(set(r["name"] for r in roles))}
        ]
        return flows

    def _design_pages(self, entities: List[Dict], roles: List[Dict]) -> List[Dict]:
        role_names = [r["name"] for r in roles]
        pages = [
            {"name": "Login", "route": "/login", "allowed_roles": ["guest"], "components": ["Form"]},
            {"name": "Dashboard", "route": "/dashboard", "allowed_roles": role_names if role_names else ["user"], "components": ["StatsCard", "Table"]}
        ]
        for entity in entities:
            entity_name = entity["name"]
            lower = entity_name.lower()
            if lower.endswith("s"):
                plural = lower
            elif lower.endswith("y"):
                plural = lower[:-1] + "ies"
            else:
                plural = lower + "s"
            pages.append({"name": f"{entity_name} List", "route": f"/{plural}", "allowed_roles": role_names if role_names else ["user", "admin"], "components": ["Table", "SearchInput", "CreateButton"]})
            pages.append({"name": f"{entity_name} Form", "route": f"/{plural}/new", "allowed_roles": role_names if "admin" in role_names else role_names, "components": ["Form", "SubmitButton"]})
        return pages
