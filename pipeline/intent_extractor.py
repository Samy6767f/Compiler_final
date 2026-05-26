import json
import re
import difflib  
import logging
from typing import Dict, List, Any, Optional
from functools import lru_cache
from pipeline.llm import generate_with_llama, review_with_model

logger = logging.getLogger("ai-compiler")

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
ENTITY_KEYWORDS = {
    'contact': 'contacts', 'user': 'users', 'customer': 'customers',
    'product': 'products', 'order': 'orders', 'invoice': 'invoices',
    'payment': 'payments', 'task': 'tasks', 'project': 'projects',
    'company': 'companies', 'lead': 'leads', 'deal': 'deals',
    'ticket': 'tickets', 'article': 'articles', 'post': 'posts',
    'comment': 'comments', 'category': 'categories', 'tag': 'tags',
    'inventory': 'inventory', 'vendor': 'vendors', 'subscription': 'subscriptions',
    'driver': 'drivers', 'vehicle': 'vehicles', 'trip': 'trips',
    'delivery': 'deliveries', 'restaurant': 'restaurants',
    'passenger': 'passengers', 'ride': 'rides',
    'rating': 'ratings', 'review': 'reviews',
    'bag': 'bags',
    'doctor': 'doctors', 'patient': 'patients', 'appointment': 'appointments',
    'clinic': 'clinics', 'prescription': 'prescriptions', 'medical_record': 'medical_records',
    'staff': 'staff', 'schedule': 'schedules',
    'map': 'maps', 'location': 'locations',
    'token': 'tokens', 'session': 'sessions',
    'notification': 'notifications', 'message': 'messages',
    'audit_log': 'audit_logs', 'audit': 'audit_logs',
}

ROLE_KEYWORDS = {
    'admin': 'admin', 'administrator': 'admin',
    'global_admin': 'global_admin', 'super_admin': 'admin',
    'manager': 'manager', 'clinic_manager': 'clinic_manager',
    'doctor': 'doctor', 'nurse': 'nurse',
    'patient': 'patient', 'customer': 'customer',
    'driver': 'driver', 'rider': 'driver',
    'user': 'user', 'guest': 'guest', 'viewer': 'guest',
    'moderator': 'moderator', 'owner': 'owner',
    'staff': 'staff', 'employee': 'staff',
    'rep': 'rep', 'sales_rep': 'rep', 'representative': 'rep',
    'receptionist': 'receptionist', 'agent': 'agent',
    'support': 'support', 'instructor': 'instructor', 'student': 'student',
    'buyer': 'buyer', 'seller': 'seller', 'vendor': 'vendor',
    'editor': 'editor', 'author': 'author', 'reader': 'reader',
}

FEATURE_KEYWORDS = {
    'login': 'Authentication', 'register': 'Registration', 'signup': 'Registration',
    'dashboard': 'Dashboard', 'analytics': 'Analytics', 'payment': 'Payments',
    'billing': 'Billing', 'search': 'Search', 'filter': 'Filtering',
    'export': 'Export', 'import': 'Import', 'chat': 'Chat', 'messaging': 'Messaging',
    'notification': 'Notifications', 'email': 'Email Notifications', 'sms': 'SMS Notifications',
    'comment': 'Comments', 'like': 'Likes', 'follow': 'Follow', 'post': 'Posts',
    'review': 'Reviews', 'rating': 'Ratings', 'cart': 'Shopping Cart',
    'checkout': 'Checkout', 'order': 'Orders', 'refund': 'Refunds',
    'booking': 'Booking System', 'appointment': 'Appointments', 'reservation': 'Reservations',
    'subscription': 'Subscription', 'multi-tenant': 'Multi-tenancy',
    'map': 'Map View', 'dark mode': 'Dark Mode', 'darkmode': 'Dark Mode',
    'insulated bag': 'Insulated Bags', '5-star': 'Rating System', '5 star': 'Rating System',
    'closest': 'Closest Driver Matching', 'auto-assign': 'Auto Assignment',
    'reject': 'Driver Rejection', 'accept': 'Accept Flow',
    '30 second': '30s Acceptance Timer', '30sec': '30s Acceptance Timer', '30s': '30s Acceptance Timer',
    'magic link': 'Magic Link Auth', 'email auth': 'Email Auth',
    'corporate': 'Corporate 2FA', 'token': '2FA Token',
    '2fa': '2FA Authentication', 'mfa': '2FA Authentication',
    'dark mode': 'Dark Mode Toggle', 'darkmode': 'Dark Mode Toggle',
    'hybrid': 'Hybrid App',
}

# Role → permissions mapping
ROLE_PERMISSIONS = {
    'admin': ["create", "read", "update", "delete", "admin"],
    'global_admin': ["create", "read", "update", "delete", "admin"],
    'super_admin': ["create", "read", "update", "delete", "admin"],
    'manager': ["create", "read", "update"],
    'clinic_manager': ["create", "read", "update"],
    'guest': ["read"],
    'viewer': ["read"],
    'moderator': ["create", "read", "update", "delete"],
    'doctor': ["create", "read", "update"],
    'patient': ["create", "read", "update"],
    'driver': ["create", "read", "update"],
    'staff': ["create", "read", "update"],
    'rep': ["create", "read", "update"],
    'agent': ["create", "read", "update"],
    'support': ["create", "read", "update"],
    'student': ["read"],
    'reader': ["read"],
    'buyer': ["read"],
    'instructor': ["create", "read", "update"],
    'author': ["create", "read", "update"],
    'seller': ["create", "read", "update"],
    'vendor': ["create", "read", "update"],
    'editor': ["create", "read", "update"],
    'default': ["create", "read", "update"],
}


VALID_APP_TYPES = {'crm', 'ecommerce', 'saas', 'dashboard', 'marketplace', 'social', 'healthcare', 'booking', 'ride', 'delivery'}


class IntentExtractor:
    def __init__(self):
      
        self.role_patterns = {keyword: re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                              for keyword in ROLE_KEYWORDS}
        self.entity_patterns = {keyword: re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                                for keyword in ENTITY_KEYWORDS}
        self._cache = {}

    @staticmethod
    def _dedupe_preserve_order(seq: List) -> List:
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    @lru_cache(maxsize=64)
    def _cached_llm_extract(self, prompt: str, system_instruction: str) -> str:
        return generate_with_llama(prompt, system_instruction, max_tokens=1024)

    def extract(self, prompt: str) -> Dict[str, Any]:
        try:
            system_instruction = (
                "You are the structural Intent Extractor frontend of an enterprise software compiler.\n"
                "Analyze the user requirements and output an un-collapsed Intermediate Representation (IR).\n\n"
                "CRITICAL COMPILER ENFORCEMENT RULES:\n"
                "1. DO NOT simplify roles. If a user asks for 'GlobalAdmin' and 'ClinicManager', do not merge them into a generic 'user' or 'admin'.\n"
                "2. Extract custom database fields, tracking keys, structural hierarchies (e.g. self-referencing relationship joins), and complex conditional validation rules entirely.\n"
                "3. Explicitly list any discovered third-party integrations under 'integrations'.\n"
                "4. Populate the 'ambiguities' or 'assumptions' tracking blocks if critical configuration details are missing.\n"
                "5. If the user intent is vague or incomplete, still produce a minimal but complete IR:\n"
                "   - Include at least one entity (e.g., 'Item').\n"
                "   - Include at least one role (e.g., 'user' with 'read' permission).\n"
                "   - Include a plausible app_type based on keywords.\n"
                "   - Never omit any of the required top-level fields.\n\n"
                "Output ONLY a raw, minified JSON object matching this exact schema structure without markdown wrappers or explanation block strings:\n"
                "{\n"
                "  \"app_name\": \"String\",\n"
                "  \"app_type\": \"String\",\n"
                "  \"features\": [\"List of unique application capabilities requiring functional layers\"],\n"
                "  \"entities\": [\"List of all domain database tables/entities parsed\"],\n"
                "  \"roles\": [{\"name\": \"role_name\", \"permissions\": [\"create\", \"read\", \"update\", \"delete\"]}],\n"
                "  \"integrations\": [\"Third-party external APIs identified\"],\n"
                "  \"ambiguities\": [],\n"
                "  \"assumptions\": []\n"
                "}"
            )
            raw_content = self._cached_llm_extract(prompt, system_instruction)
            if "</thought>" in raw_content:
                raw_content = raw_content.split("</thought>")[-1].strip()
            intent_data = json.loads(raw_content)
            return self._validate_and_heal_ir(intent_data, prompt)
        except Exception as e:
            logger.error(f"LLM Intent Extraction Compiler Fault: {e}, falling back to rule-based")
            logger.info("Using rule-based intent extraction (fallback)")
            return self.extract_rule_based(prompt)

    def _validate_and_heal_ir(self, ir: Dict[str, Any], original_prompt: str = "") -> Dict[str, Any]:
        required_keys = ["features", "entities", "roles", "integrations", "ambiguities", "assumptions"]
        for key in required_keys:
            if key not in ir or not isinstance(ir[key], list):
                ir[key] = []

        if not ir["entities"]:
            ir["entities"] = ["Item"]
            ir["assumptions"].append("Added default entity 'Item' (none extracted)")
        if not ir["roles"]:
            ir["roles"] = [{"name": "user", "permissions": ["read"]}]
            ir["assumptions"].append("Added default 'user' role (none extracted)")
        if not ir["features"]:
            ir["features"] = ["Basic CRUD Operations"]
            ir["assumptions"].append("Added basic CRUD feature (none extracted)")

        for role in ir["roles"]:
            if "read" not in role.get("permissions", []):
                role.setdefault("permissions", []).append("read")
                ir["assumptions"].append(f"Added 'read' permission to role '{role['name']}'")

        for entity in ir["entities"]:
            crud = f"{entity.title()} CRUD"
            if crud not in ir["features"]:
                ir["features"].append(crud)
                ir["assumptions"].append(f"Added CRUD feature for entity '{entity}'")

        if "app_name" not in ir or not ir["app_name"] or ir["app_name"] in ["MyApp", "App"]:
            ir["app_name"] = "GeneratedEnterpriseApp"
        if "app_type" not in ir or not ir["app_type"]:
            ir["app_type"] = "saas"
        elif ir["app_type"].lower() not in VALID_APP_TYPES:
            ir["ambiguities"].append(f"App type '{ir['app_type']}' not in standard list, using 'saas'")
            ir["app_type"] = "saas"

        if original_prompt:
            prompt_lower = original_prompt.lower()
            detected = [t for t in VALID_APP_TYPES if t in prompt_lower]
            if len(detected) > 1:
                ir["ambiguities"].append(f"Prompt suggests multiple app types: {detected}. Selected '{ir['app_type']}'.")
        return ir

    def extract_with_review(self, prompt: str) -> Dict[str, Any]:
        draft = self.extract(prompt)
        try:
            draft_json = json.dumps(draft)
            corrected, was_fixed = review_with_model(
                draft_json,
                "Ensure app_name, app_type, features[], entities[], roles[], integrations[] are all present and comprehensive"
            )
            if was_fixed:
                draft = json.loads(corrected)
                logger.info(f"Intent Refinement Complete: fixed={was_fixed}")
        except Exception as e:
            logger.warning(f"Intent verification check failed: {e}")
        return draft

    # ------------------------------------------------------------------
    # Rule‑based fallback (using precompiled patterns)
    # ------------------------------------------------------------------
    def extract_rule_based(self, prompt: str) -> Dict:
        logger.info("Running rule-based intent extraction (fallback)")
        prompt_lower = prompt.lower()
        entities = self._extract_entities(prompt_lower)
        roles = self._extract_roles(prompt_lower)
        features = self._extract_features(prompt, entities, roles)
        integrations = self._detect_integrations(prompt_lower)
        app_type = self._detect_app_type(prompt_lower)

        intent = {
            "app_name": self._generate_app_name(prompt),
            "app_type": app_type,
            "features": self._dedupe_preserve_order(features),
            "entities": self._dedupe_preserve_order(entities),
            "roles": roles,
            "integrations": self._dedupe_preserve_order(integrations),
            "ambiguities": [],
            "assumptions": []
        }

        if len(roles) == 1 and "admin" not in [r["name"] for r in roles]:
            intent["assumptions"].append("Added admin role for system management")
            intent["roles"].append({"name": "admin", "permissions": ["create", "read", "update", "delete", "admin"]})

        if "magic link" in prompt_lower or "passwordless" in prompt_lower:
            if "Magic Link Auth" not in intent["features"]:
                intent["features"].append("Magic Link Auth")
        if "2fa" in prompt_lower or "mfa" in prompt_lower or "corporate" in prompt_lower:
            if "2FA Authentication" not in intent["features"]:
                intent["features"].append("2FA Authentication")
        if "map" in prompt_lower and "Map View" not in intent["features"]:
            intent["features"].append("Map View")
        if ("dark mode" in prompt_lower or "darkmode" in prompt_lower) and "Dark Mode" not in intent["features"]:
            intent["features"].append("Dark Mode")
        return intent

    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        # Exact matching
        for keyword, entity_name in ENTITY_KEYWORDS.items():
            if self.entity_patterns[keyword].search(text):
                entities.append(entity_name)
        # If no exact matches, try fuzzy matching for misspellings
        if not entities:
            words = re.findall(r'\b\w+\b', text.lower())
            for keyword, entity_name in ENTITY_KEYWORDS.items():
                matches = difflib.get_close_matches(keyword, words, cutoff=0.8)
                if matches:
                    entities.append(entity_name)
        return self._dedupe_preserve_order(entities) if entities else ["items"]

    def _extract_roles(self, text: str) -> List[Dict]:
        roles_dict = {}
        # Exact matching
        for keyword, role_name in ROLE_KEYWORDS.items():
            if self.role_patterns[keyword].search(text):
                if role_name not in roles_dict:
                    roles_dict[role_name] = self._get_role_permissions(role_name)
        # If no exact matches, try fuzzy matching
        if not roles_dict:
            words = re.findall(r'\b\w+\b', text.lower())
            for keyword, role_name in ROLE_KEYWORDS.items():
                matches = difflib.get_close_matches(keyword, words, cutoff=0.8)
                if matches and role_name not in roles_dict:
                    roles_dict[role_name] = self._get_role_permissions(role_name)
        roles = [{"name": r, "permissions": perms} for r, perms in roles_dict.items()]
        if not roles:
            roles = [{"name": "user", "permissions": ["create", "read", "update"]}]
        if any(r["name"] == "admin" for r in roles) and not any(r["name"] == "user" for r in roles):
            roles.append({"name": "user", "permissions": ["create", "read", "update"]})
        return roles

    def _get_role_permissions(self, role: str) -> List[str]:
        return ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS['default'])

    def _extract_features(self, prompt: str, entities: List, roles: List) -> List[str]:
        features = []
        prompt_lower = prompt.lower()
        for keyword, feature in FEATURE_KEYWORDS.items():
            if keyword in prompt_lower and feature not in features:
                features.append(feature)
        for entity in entities:
            if entity not in ['items', 'auth', 'tokens', 'sessions', 'emails', 'sms', 'webhooks', 'maps', 'locations']:
                crud = f"{entity.title()} CRUD"
                if crud not in features:
                    features.append(crud)
        if 'driver' in prompt_lower:
            features.append("Driver Status (Online/Offline)")
            if any(k in prompt_lower for k in ['30 second', '30sec', '30s']):
                features.append("30s Driver Acceptance Timer")
            if 'insulated' in prompt_lower or 'bag' in prompt_lower:
                features.append("Insulated Bag Indicator")
        if 'reject' in prompt_lower or 'accept' in prompt_lower:
            features.append("Driver Accept/Reject Flow")
        if 'closest' in prompt_lower:
            features.append("Closest Driver Matching")
        if 'dark mode' in prompt_lower or 'darkmode' in prompt_lower:
            features.append("Dark Mode Toggle")
        return self._dedupe_preserve_order(features)

    def _detect_integrations(self, text: str) -> List[str]:
        integration_map = {
            'stripe': 'Stripe', 'payment': 'Stripe',
            'email': 'SendGrid', 'mail': 'SendGrid',
            'sms': 'Twilio',
            'auth': 'Auth0', 'auth0': 'Auth0',
            'analytics': 'Mixpanel',
            'slack': 'Slack',
            'github': 'GitHub',
            'google': 'Google OAuth',
            'facebook': 'Facebook', 'instagram': 'Instagram',
            'mapbox': 'Mapbox', 'maps': 'Mapbox', 'map': 'Mapbox',
            'twilio': 'Twilio', 'sendgrid': 'SendGrid',
            'razorpay': 'Razorpay',
        }
        found = [name for key, name in integration_map.items() if key in text]
        return self._dedupe_preserve_order(found)

    def _detect_app_type(self, text: str) -> str:
        type_signatures = {
            'crm': ['crm', 'customer relationship', 'contacts', 'leads', 'deals'],
            'ecommerce': ['ecommerce', 'shop', 'store', 'cart', 'checkout'],
            'saas': ['saas', 'subscription', 'multi-tenant'],
            'dashboard': ['dashboard', 'analytics', 'metrics', 'reporting'],
            'marketplace': ['marketplace', 'vendor', 'seller'],
            'social': ['social', 'post', 'like', 'comment', 'follow', 'feed'],
            'healthcare': ['patient', 'doctor', 'medical', 'health', 'clinic'],
            'booking': ['booking', 'appointment', 'reservation'],
            'ride': ['ride', 'driver', 'taxi', 'cab'],
            'delivery': ['delivery', 'food delivery', 'courier'],
        }
        for app_type, signatures in type_signatures.items():
            if any(sig in text for sig in signatures):
                return app_type
        return 'saas'

    def _generate_app_name(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        type_names = {
            'crm': 'CRM', 'ecommerce': 'ShopApp', 'shop': 'ShopApp',
            'delivery': 'DeliveryApp', 'ride': 'RideApp', 'food': 'FoodApp',
            'healthcare': 'HealthApp', 'booking': 'BookingApp',
            'social': 'SocialApp', 'dashboard': 'Dashboard', 'saas': 'SaaSApp',
        }
        for key, name in type_names.items():
            if key in prompt_lower:
                return name
        stop_words = {'crm','user','admin','build','create','make','with','that','this','the','a','an','for','and','or','but','app','application'}
        words = [w.strip('.,!?') for w in prompt.split()
                 if len(w) > 3 and w.lower() not in stop_words and not w[0].isdigit()]
        if len(words) >= 2:
            return ''.join(w.capitalize() for w in words[:2]) + 'App'
        elif words:
            return words[0].capitalize() + 'App'
        return 'GeneratedApp'
