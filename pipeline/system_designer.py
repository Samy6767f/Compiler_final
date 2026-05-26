import json
import logging
import re
from typing import Dict, List, Any
import jsonschema
from pipeline.llm import generate_with_llama, review_with_model

logger = logging.getLogger("ai-compiler")

class SystemDesigner:
    # Move schema inside the class to avoid global scope issues
    SYSTEM_DESIGN_SCHEMA = {
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

If the user intent is vague or incomplete, still produce a minimal but complete design:
- Include at least one entity (e.g., "Item").
- Include at least one role (e.g., "user").
- Include at least one page (e.g., "Dashboard").
- Include at least one flow (e.g., "User Authentication").
- Never omit any of the required top‑level keys.

Required Format Shape:
{"entities":[{"name":"EntityName","fields":["id:uuid","name:string","created_at:timestamp"],"relations":[]}],"flows":[{"name":"FlowName","steps":["step1","step2"],"actors":["role"]}],"roles":[{"name":"rolename","permissions":["read","write"]}],"permissions":{"permName":["role"]},"pages":[{"name":"PageName","route":"/route","allowed_roles":["role"],"components":["Component"]}]}"""

    def design(self, intent: Dict) -> Dict:
        return self.design_llm(intent)

    def design_llm(self, intent: Dict) -> Dict:
        try:
            # Use unified generation function
            raw_content = generate_with_llama(
                json.dumps(intent),
                self.SYSTEM_DESIGN_PROMPT,
                max_tokens=1024
            )

            if "</thought>" in raw_content:
                raw_content = raw_content.split("</thought>")[-1].strip()

            draft = self._parse_and_validate(raw_content)

            # Review step (already uses review_with_model)
            draft_json = json.dumps(draft)
            corrected, was_fixed = review_with_model(
                draft_json,
                "Ensure all entities have fields (id, name, created_at), roles have permissions, pages have routes and allowed_roles, permissions mapping is correct"
            )
            if was_fixed:
                draft = self._parse_and_validate(corrected)
                logger.info("Design Phase: Architecture optimized via validation verification flow.")

            return draft

        except Exception as e:
            logger.error(f"LLM design generation failed, falling back to rule‑based: {e}")
            return self.design_rule_based(intent)

    # All other methods (_normalize_relation, _parse_and_validate, design_rule_based, etc.) remain exactly as they were.
    # (Keep your existing implementation for these methods – unchanged)
